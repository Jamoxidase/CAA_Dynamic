"""
Plots all 70 weighted CAVs (10 per behavior × 7 behaviors) in interactive 3D space
using dimensionality reduction, color-coded by behavior class.

Each weighted CAV represents the weighted combination of base CAVs for a single test AB question.

Usage:
python plot_weighted_cavs_3d.py --layer 14 --model_size 7b --method pca
"""

import torch as t
import numpy as np
import plotly.graph_objects as go
import argparse
import os
import json
from sklearn.decomposition import PCA
from pathlib import Path

# Color palette for 7 behaviors
BEHAVIOR_COLORS = {
    "envy-kindness": "#FF6B6B",          # Red
    "gluttony-temperance": "#FFA500",    # Orange
    "greed-charity": "#FFD700",          # Gold
    "lust-chastity": "#9370DB",          # Purple
    "pride-humility": "#4169E1",         # Blue
    "sloth-diligence": "#2E8B57",        # Green
    "wrath-patience": "#DC143C",         # Crimson
}

BEHAVIORS = [
    "envy-kindness",
    "gluttony-temperance",
    "greed-charity",
    "lust-chastity",
    "pride-humility",
    "sloth-diligence",
    "wrath-patience",
]

def load_weighted_cavs(layer, model_size, use_base_model=False):
    """
    Load all weighted CAVs for specified layer.

    Returns:
        cavs: numpy array of shape (70, hidden_dim)
        labels: list of behavior names (length 70)
        metadata: list of dicts with question info
    """
    vectors_dir = Path(__file__).parent / "vectors"

    all_cavs = []
    all_labels = []
    all_metadata = []

    model_suffix = "base" if use_base_model else "7b"

    for behavior in BEHAVIORS:
        # Load the weighted vectors tensor file
        pt_file = vectors_dir / f"{behavior}_weighted_vectors_layer{layer}_{model_suffix}.pt"
        json_file = vectors_dir / f"{behavior}_weighted_vectors_layer{layer}_{model_suffix}.json"

        if not pt_file.exists():
            print(f"Warning: {pt_file} not found, skipping {behavior}")
            continue

        # Load tensors (shape: [n_questions, hidden_dim])
        data_pt = t.load(pt_file, weights_only=False)
        weighted_vecs = data_pt["weighted_vectors"]

        # Load metadata
        with open(json_file) as f:
            data = json.load(f)

        # Convert to numpy
        vecs_np = weighted_vecs.numpy()

        all_cavs.append(vecs_np)
        all_labels.extend([behavior] * len(vecs_np))
        all_metadata.extend(data['data_points'])

        print(f"Loaded {behavior}: {len(vecs_np)} weighted CAVs")

    # Concatenate all
    cavs = np.vstack(all_cavs)

    print(f"\nTotal weighted CAVs loaded: {len(cavs)}")
    print(f"CAV dimension: {cavs.shape[1]}")

    return cavs, all_labels, all_metadata

def reduce_dimensions(cavs, method='pca', n_components=3):
    """
    Reduce CAVs to 3D using PCA or UMAP.
    """
    print(f"\nReducing dimensions using {method.upper()}...")

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(cavs)
        variance_explained = reducer.explained_variance_ratio_
        print(f"Variance explained: {variance_explained}")
        print(f"Total variance: {variance_explained.sum():.3f}")

    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42,
                          n_neighbors=15, min_dist=0.1)
            reduced = reducer.fit_transform(cavs)
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            print("Falling back to PCA...")
            return reduce_dimensions(cavs, method='pca', n_components=n_components)

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(cavs)

    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced

def plot_3d_interactive(reduced, labels, metadata, output_path=None, method='pca'):
    """
    Create interactive 3D scatter plot with plotly.
    """
    print("\nCreating interactive 3D plot...")

    fig = go.Figure()

    # Plot each behavior separately for color coding
    for behavior in BEHAVIORS:
        mask = np.array([l == behavior for l in labels])
        if mask.sum() > 0:
            # Get metadata for this behavior
            behavior_metadata = [m for l, m in zip(labels, metadata) if l == behavior]

            # Create hover text with question snippets
            hover_texts = []
            for i, meta in enumerate(behavior_metadata):
                question = meta['question'].split('\n')[0][:80] + "..."  # First line, truncated
                answer_match = meta['answer_matching_behavior']
                hover_texts.append(f"{behavior}<br>Q{i}: {question}<br>Answer: {answer_match}")

            fig.add_trace(go.Scatter3d(
                x=reduced[mask, 0],
                y=reduced[mask, 1],
                z=reduced[mask, 2],
                mode='markers',
                name=behavior,
                marker=dict(
                    size=6,
                    color=BEHAVIOR_COLORS[behavior],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))

    variance_text = ""
    if method == 'pca':
        # Calculate variance for title
        pca = PCA(n_components=3, random_state=42)
        pca.fit(np.vstack([reduced]))
        total_var = sum(pca.explained_variance_ratio_) * 100
        variance_text = f" ({total_var:.1f}% variance)"

    fig.update_layout(
        title=f"70 Weighted CAVs in 3D Space ({method.upper()}){variance_text}<br><sub>Color = Behavior Class | 10 points per behavior</sub>",
        scene=dict(
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            zaxis_title=f"{method.upper()} Component 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=1000,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")

    fig.show()

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot 70 weighted CAVs in 3D space")
    parser.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    parser.add_argument("--model_size", type=str, choices=["7b", "8b", "13b", "1.2b"], default="7b")
    parser.add_argument("--use_base_model", action="store_true", help="Use base model instead of chat")
    parser.add_argument("--method", type=str, choices=["pca", "umap", "tsne"], default="pca",
                       help="Dimensionality reduction method")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HTML file path")

    args = parser.parse_args()

    # Load weighted CAVs
    cavs, labels, metadata = load_weighted_cavs(
        args.layer, args.model_size, args.use_base_model
    )

    # Reduce to 3D
    reduced = reduce_dimensions(cavs, method=args.method, n_components=3)

    # Set output path
    if args.output is None:
        output_dir = Path(__file__).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        model_suffix = "base" if args.use_base_model else "chat"
        args.output = output_dir / f"weighted_cavs_3d_layer{args.layer}_{args.model_size}_{model_suffix}_{args.method}.html"

    # Plot
    plot_3d_interactive(
        reduced, labels, metadata,
        output_path=args.output,
        method=args.method
    )

    print("\nDone!")

if __name__ == "__main__":
    main()
