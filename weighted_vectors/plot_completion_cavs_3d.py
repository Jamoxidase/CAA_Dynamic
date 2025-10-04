"""
Plots all 780 completion activations (390 pos + 390 neg from generate dataset)
in interactive 3D space using dimensionality reduction, color-coded by behavior class.

Usage:
python plot_completion_cavs_3d.py --layer 15 --model_size 7b --method umap
"""

import torch as t
import numpy as np
import plotly.graph_objects as go
import argparse
import os
import sys
from sklearn.decomposition import PCA
from pathlib import Path

# Add parent directory to path to import from CAA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behaviors import ALL_BEHAVIORS, get_activations_path
from utils.helpers import get_model_path

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

def load_all_activations(layer, model_size, use_base_model=False):
    """
    Load all positive and negative activations for all behaviors.

    Returns:
        activations: numpy array of shape (780, hidden_dim)
        labels: list of behavior names (length 780)
        pos_neg: list of 'pos' or 'neg' (length 780)
    """
    model_name_path = get_model_path(model_size, is_base=use_base_model)

    all_activations = []
    all_labels = []
    all_pos_neg = []

    for behavior in ALL_BEHAVIORS:
        # Load positive activations
        pos_path = get_activations_path(behavior, layer, model_name_path, "pos")
        if not os.path.exists(pos_path):
            print(f"Warning: {pos_path} not found, skipping {behavior}")
            continue

        pos_activations = t.load(pos_path)  # Shape: (n_samples, hidden_dim)
        neg_path = get_activations_path(behavior, layer, model_name_path, "neg")
        neg_activations = t.load(neg_path)

        # Convert to numpy
        pos_np = pos_activations.cpu().numpy()
        neg_np = neg_activations.cpu().numpy()

        # Add to lists
        all_activations.append(pos_np)
        all_activations.append(neg_np)

        # Labels
        all_labels.extend([behavior] * len(pos_np))
        all_labels.extend([behavior] * len(neg_np))

        # Pos/neg markers
        all_pos_neg.extend(['pos'] * len(pos_np))
        all_pos_neg.extend(['neg'] * len(neg_np))

        print(f"Loaded {behavior}: {len(pos_np)} pos + {len(neg_np)} neg = {len(pos_np) + len(neg_np)} total")

    # Concatenate all
    activations = np.vstack(all_activations)

    print(f"\nTotal activations loaded: {len(activations)}")
    print(f"Activation dimension: {activations.shape[1]}")

    return activations, all_labels, all_pos_neg

def reduce_dimensions(activations, method='pca', n_components=3):
    """
    Reduce activations to 3D using PCA or UMAP.
    """
    print(f"\nReducing dimensions using {method.upper()}...")

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(activations)
        variance_explained = reducer.explained_variance_ratio_
        print(f"Variance explained: {variance_explained}")
        print(f"Total variance: {variance_explained.sum():.3f}")

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42,
                               n_neighbors=15, min_dist=0.1)
            reduced = reducer.fit_transform(activations)
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            print("Falling back to PCA...")
            return reduce_dimensions(activations, method='pca', n_components=n_components)

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(activations)

    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced

def plot_3d_interactive(reduced, labels, pos_neg, output_path=None, show_pos_neg=True):
    """
    Create interactive 3D scatter plot with plotly.
    """
    print("\nCreating interactive 3D plot...")

    fig = go.Figure()

    # Plot each behavior separately for color coding
    for behavior in ALL_BEHAVIORS:
        if show_pos_neg:
            # Plot positive and negative separately with different markers
            for pn, marker in [('pos', 'circle'), ('neg', 'diamond')]:
                mask = np.array([(l == behavior and p == pn) for l, p in zip(labels, pos_neg)])
                if mask.sum() > 0:
                    fig.add_trace(go.Scatter3d(
                        x=reduced[mask, 0],
                        y=reduced[mask, 1],
                        z=reduced[mask, 2],
                        mode='markers',
                        name=f"{behavior} ({pn})",
                        marker=dict(
                            size=4,
                            color=BEHAVIOR_COLORS[behavior],
                            symbol=marker,
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        text=[f"{behavior}<br>{pn}<br>({i})" for i in range(mask.sum())],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                    ))
        else:
            # Plot all together
            mask = np.array([l == behavior for l in labels])
            if mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=reduced[mask, 0],
                    y=reduced[mask, 1],
                    z=reduced[mask, 2],
                    mode='markers',
                    name=behavior,
                    marker=dict(
                        size=4,
                        color=BEHAVIOR_COLORS[behavior],
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"{behavior}<br>{pos_neg[i]}" for i in range(len(labels)) if labels[i] == behavior],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ))

    fig.update_layout(
        title="780 Completion Activations in 3D Space<br><sub>Color = Behavior Class, Shape = Pos/Neg</sub>",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=900,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")

    fig.show()

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot 780 completion activations in 3D space")
    parser.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    parser.add_argument("--model_size", type=str, choices=["7b", "8b", "13b", "1.2b"], default="7b")
    parser.add_argument("--use_base_model", action="store_true", help="Use base model instead of chat")
    parser.add_argument("--method", type=str, choices=["pca", "umap", "tsne"], default="pca",
                       help="Dimensionality reduction method")
    parser.add_argument("--output", type=str, default=None,
                       help="Output HTML file path (default: ./plots/completion_cavs_3d_layer{N}.html)")
    parser.add_argument("--no_pos_neg_split", action="store_true",
                       help="Don't show separate markers for pos/neg")

    args = parser.parse_args()

    # Load activations
    activations, labels, pos_neg = load_all_activations(
        args.layer, args.model_size, args.use_base_model
    )

    # Reduce to 3D
    reduced = reduce_dimensions(activations, method=args.method, n_components=3)

    # Set output path
    if args.output is None:
        output_dir = Path(__file__).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        model_suffix = "base" if args.use_base_model else "chat"
        args.output = output_dir / f"completion_cavs_3d_layer{args.layer}_{args.model_size}_{model_suffix}_{args.method}.html"

    # Plot
    plot_3d_interactive(
        reduced, labels, pos_neg,
        output_path=args.output,
        show_pos_neg=not args.no_pos_neg_split
    )

    print("\nDone!")

if __name__ == "__main__":
    main()
