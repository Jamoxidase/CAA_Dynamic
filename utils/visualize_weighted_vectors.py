#!/usr/bin/env python3
"""
Interactive visualization of weighted CAV vectors using dimensionality reduction.
Shows behavior clusters with positive/negative distinction.
"""

import torch
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import argparse
import os
from typing import Dict, List, Tuple

# The 7 behaviors with their color assignments
BEHAVIOR_COLORS = {
    "envy-kindness": "#2ca02c",        # Green
    "gluttony-temperance": "#ff7f0e",  # Orange
    "greed-charity": "#d62728",        # Red
    "lust-chastity": "#9467bd",        # Purple
    "pride-humility": "#1f77b4",       # Blue
    "sloth-diligence": "#8c564b",      # Brown
    "wrath-patience": "#e377c2"        # Pink
}

def load_weighted_vectors(layer: int, data_dir: str = "./weighted_cav_vectors") -> Tuple[Dict, np.ndarray, List]:
    """
    Load all weighted vectors and metadata for a layer.

    Returns:
        - vectors_dict: Dictionary with 'positive', 'negative', 'difference' tensors
        - combined_vectors: All vectors stacked for dimensionality reduction
        - metadata: List of metadata for each vector
    """

    tensor_file = os.path.join(data_dir, f"all_weighted_vectors_layer{layer}.pt")
    if not os.path.exists(tensor_file):
        raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

    print(f"Loading vectors from {tensor_file}...")
    data = torch.load(tensor_file, map_location='cpu')

    all_vectors = []
    all_metadata = []

    # Process each behavior
    for behavior in BEHAVIOR_COLORS.keys():
        json_file = os.path.join(data_dir, f"{behavior}_weighted_cavs_layer{layer}.json")
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found for {behavior}")
            continue

        # Load JSON for questions
        import json
        with open(json_file, 'r') as f:
            behavior_data = json.load(f)

        for idx, dp in enumerate(behavior_data["data_points"]):
            # Add positive vector
            if "positive_weighted_vector" in dp:
                all_vectors.append(dp["positive_weighted_vector"])
                all_metadata.append({
                    "behavior": behavior,
                    "type": "positive",
                    "index": idx,
                    "question": dp["question"][:100] + "..." if len(dp["question"]) > 100 else dp["question"],
                    "answer": dp["answer_matching_behavior"][:100] + "..."
                })

            # Add negative vector
            if "negative_weighted_vector" in dp:
                all_vectors.append(dp["negative_weighted_vector"])
                all_metadata.append({
                    "behavior": behavior,
                    "type": "negative",
                    "index": idx,
                    "question": dp["question"][:100] + "..." if len(dp["question"]) > 100 else dp["question"],
                    "answer": dp["answer_not_matching_behavior"][:100] + "..."
                })

    print(f"Loaded {len(all_vectors)} vectors")

    # Convert to numpy array
    combined_vectors = np.array(all_vectors, dtype=np.float32)

    return data, combined_vectors, all_metadata

def reduce_dimensions(vectors: np.ndarray, method: str = "pca", n_components: int = 3,
                      perplexity: int = 30, n_neighbors: int = 15) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 3D for visualization.

    Args:
        vectors: Array of shape (n_samples, n_features)
        method: 'pca', 'tsne', or 'umap'
        n_components: Number of dimensions (3 for 3D visualization)
        perplexity: For t-SNE
        n_neighbors: For UMAP

    Returns:
        Array of shape (n_samples, n_components)
    """

    print(f"Reducing dimensions using {method.upper()}...")
    print(f"Input shape: {vectors.shape}")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(vectors)
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(reducer.explained_variance_ratio_):.2%}")

    elif method == "tsne":
        # First reduce with PCA if needed
        if vectors.shape[1] > 50:
            print("Pre-reducing with PCA to 50 dimensions...")
            pca = PCA(n_components=50, random_state=42)
            vectors = pca.fit_transform(vectors)

        reducer = TSNE(n_components=n_components, perplexity=perplexity,
                      random_state=42, n_iter=1000)
        reduced = reducer.fit_transform(vectors)

    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           random_state=42, min_dist=0.1)
        reduced = reducer.fit_transform(vectors)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Output shape: {reduced.shape}")
    return reduced

def create_interactive_plot(reduced_vectors: np.ndarray, metadata: List[Dict],
                           method: str, layer: int) -> go.Figure:
    """
    Create an interactive 3D scatter plot with Plotly.
    """

    # Create DataFrame for easier handling
    df = pd.DataFrame(metadata)
    df['x'] = reduced_vectors[:, 0]
    df['y'] = reduced_vectors[:, 1]
    df['z'] = reduced_vectors[:, 2] if reduced_vectors.shape[1] > 2 else 0

    # Create the figure
    fig = go.Figure()

    # Add traces for each behavior and type combination
    for behavior in BEHAVIOR_COLORS.keys():
        behavior_df = df[df['behavior'] == behavior]

        # Positive points (dots)
        pos_df = behavior_df[behavior_df['type'] == 'positive']
        if not pos_df.empty:
            fig.add_trace(go.Scatter3d(
                x=pos_df['x'],
                y=pos_df['y'],
                z=pos_df['z'],
                mode='markers',
                name=f"{behavior} (+)",
                marker=dict(
                    color=BEHAVIOR_COLORS[behavior],
                    size=6,
                    symbol='circle',
                    line=dict(color='white', width=0.5)
                ),
                text=[f"<b>{behavior} (positive)</b><br>"
                      f"Index: {row['index']}<br>"
                      f"Q: {row['question']}<br>"
                      f"A: {row['answer']}"
                      for _, row in pos_df.iterrows()],
                hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}',
                legendgroup=behavior,
                showlegend=True
            ))

        # Negative points (crosses)
        neg_df = behavior_df[behavior_df['type'] == 'negative']
        if not neg_df.empty:
            fig.add_trace(go.Scatter3d(
                x=neg_df['x'],
                y=neg_df['y'],
                z=neg_df['z'],
                mode='markers',
                name=f"{behavior} (-)",
                marker=dict(
                    color=BEHAVIOR_COLORS[behavior],
                    size=6,
                    symbol='cross',
                    line=dict(color='black', width=0.5)
                ),
                text=[f"<b>{behavior} (negative)</b><br>"
                      f"Index: {row['index']}<br>"
                      f"Q: {row['question']}<br>"
                      f"A: {row['answer']}"
                      for _, row in neg_df.iterrows()],
                hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}',
                legendgroup=behavior,
                showlegend=True
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Weighted CAV Vectors - Layer {layer} ({method.upper()})",
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemsizing='constant'
        ),
        width=1200,
        height=800,
        hovermode='closest',
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Add annotations explaining the symbols
    fig.add_annotation(
        text="● = Positive (matching behavior)<br>✕ = Negative (not matching)",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )

    return fig

def create_comparison_plot(reduced_vectors: np.ndarray, metadata: List[Dict],
                          layer: int) -> go.Figure:
    """
    Create a comparison plot with multiple dimensionality reduction methods.
    """

    methods = ["pca", "tsne", "umap"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m.upper() for m in methods],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # Load vectors once
    vectors = reduced_vectors

    for col, method in enumerate(methods, 1):
        # Reduce dimensions
        if method == "pca":
            reduced = reduce_dimensions(vectors, method="pca")
        elif method == "tsne":
            reduced = reduce_dimensions(vectors, method="tsne", perplexity=min(30, len(vectors)//4))
        else:  # umap
            reduced = reduce_dimensions(vectors, method="umap", n_neighbors=min(15, len(vectors)//2))

        # Add traces for this subplot
        df = pd.DataFrame(metadata)
        df['x'] = reduced[:, 0]
        df['y'] = reduced[:, 1]
        df['z'] = reduced[:, 2] if reduced.shape[1] > 2 else 0

        for behavior in BEHAVIOR_COLORS.keys():
            behavior_df = df[df['behavior'] == behavior]

            # Positive points
            pos_df = behavior_df[behavior_df['type'] == 'positive']
            if not pos_df.empty:
                fig.add_trace(go.Scatter3d(
                    x=pos_df['x'],
                    y=pos_df['y'],
                    z=pos_df['z'],
                    mode='markers',
                    name=f"{behavior} (+)",
                    marker=dict(
                        color=BEHAVIOR_COLORS[behavior],
                        size=4,
                        symbol='circle'
                    ),
                    showlegend=(col == 1),
                    legendgroup=behavior
                ), row=1, col=col)

            # Negative points
            neg_df = behavior_df[behavior_df['type'] == 'negative']
            if not neg_df.empty:
                fig.add_trace(go.Scatter3d(
                    x=neg_df['x'],
                    y=neg_df['y'],
                    z=neg_df['z'],
                    mode='markers',
                    name=f"{behavior} (-)",
                    marker=dict(
                        color=BEHAVIOR_COLORS[behavior],
                        size=4,
                        symbol='cross'
                    ),
                    showlegend=(col == 1),
                    legendgroup=behavior
                ), row=1, col=col)

    fig.update_layout(
        title=f"Weighted CAV Vectors - Layer {layer} (Method Comparison)",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10)
        )
    )

    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize weighted CAV vectors')
    parser.add_argument('--layer', type=int, default=15,
                        help='Layer number to visualize')
    parser.add_argument('--method', type=str, default='pca',
                        choices=['pca', 'tsne', 'umap', 'all'],
                        help='Dimensionality reduction method')
    parser.add_argument('--data_dir', type=str, default='./weighted_cav_vectors',
                        help='Directory containing weighted vector data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file (default: weighted_vectors_layer{N}_{method}.html)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='Perplexity for t-SNE')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP')
    parser.add_argument('--no_browser', action='store_true',
                        help='Do not open browser after creating visualization')

    args = parser.parse_args()

    # Load data
    try:
        data_dict, vectors, metadata = load_weighted_vectors(args.layer, args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you've run create_weighted_cav_vectors.py for layer {args.layer} first.")
        return

    if args.method == 'all':
        # Create comparison plot
        fig = create_comparison_plot(vectors, metadata, args.layer)
        output_file = args.output or f"weighted_vectors_layer{args.layer}_comparison.html"
    else:
        # Reduce dimensions
        reduced = reduce_dimensions(
            vectors,
            method=args.method,
            n_components=3,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors
        )

        # Create interactive plot
        fig = create_interactive_plot(reduced, metadata, args.method, args.layer)
        output_file = args.output or f"weighted_vectors_layer{args.layer}_{args.method}.html"

    # Save and optionally open
    fig.write_html(output_file)
    print(f"\nVisualization saved to: {output_file}")

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        print("Opening in browser...")

    # Print some statistics
    print("\n" + "="*60)
    print("CLUSTERING STATISTICS")
    print("="*60)

    df = pd.DataFrame(metadata)
    for behavior in BEHAVIOR_COLORS.keys():
        behavior_df = df[df['behavior'] == behavior]
        n_pos = len(behavior_df[behavior_df['type'] == 'positive'])
        n_neg = len(behavior_df[behavior_df['type'] == 'negative'])
        print(f"{behavior}: {n_pos} positive, {n_neg} negative")

    print(f"\nTotal vectors: {len(metadata)}")
    print(f"Dimensions: {vectors.shape[1]} -> 3")

if __name__ == "__main__":
    main()