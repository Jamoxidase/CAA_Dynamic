#!/usr/bin/env python3
"""
Visualize the 7-dimensional behavior vectors (not the weighted CAVs).
These vectors show the composition of behaviors in each data point.
"""

import numpy as np
import json
import plotly.graph_objs as go
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

def load_behavior_vectors(layer: int, data_dir: str = "./behavior_vectors") -> Tuple[np.ndarray, List]:
    """
    Load the 7-dimensional behavior vectors from the behavior_vectors directory.

    Returns:
        - vectors: Array of 7D vectors
        - metadata: List of metadata for each vector
    """

    all_vectors = []
    all_metadata = []

    for behavior in BEHAVIOR_COLORS.keys():
        json_file = os.path.join(data_dir, f"{behavior}_behavior_vectors_layer{layer}.json")
        if not os.path.exists(json_file):
            print(f"Warning: File not found: {json_file}")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        for dp in data["data_points"]:
            # Add positive vector
            if "positive_behavior_vector" in dp:
                all_vectors.append(dp["positive_behavior_vector"])
                all_metadata.append({
                    "behavior": behavior,
                    "type": "positive",
                    "index": dp["index"],
                    "question": dp["question"][:100] + "..." if len(dp["question"]) > 100 else dp["question"],
                    "answer": dp["answer_matching_behavior"][:100] + "..."
                })

            # Add negative vector
            if "negative_behavior_vector" in dp:
                all_vectors.append(dp["negative_behavior_vector"])
                all_metadata.append({
                    "behavior": behavior,
                    "type": "negative",
                    "index": dp["index"],
                    "question": dp["question"][:100] + "..." if len(dp["question"]) > 100 else dp["question"],
                    "answer": dp["answer_not_matching_behavior"][:100] + "..."
                })

    print(f"Loaded {len(all_vectors)} 7-dimensional behavior vectors")

    vectors = np.array(all_vectors, dtype=np.float32)
    print(f"Vector shape: {vectors.shape}")

    return vectors, all_metadata

def create_3d_plot(vectors: np.ndarray, metadata: List[Dict], layer: int, method: str = "PCA", skip_first_pc: bool = True, show_direction_lines: bool = False) -> go.Figure:
    """
    Create interactive 3D plot of the behavior vectors.

    Args:
        vectors: 7D behavior vectors
        metadata: Metadata for each vector
        layer: Layer number
        method: Dimensionality reduction method (for title only, since 7D->3D is always PCA)
        skip_first_pc: If True, skip PC1 (which just separates pos/neg) and use PC2,3,4
        show_direction_lines: If True, show lines from mean negative to mean positive for each behavior
    """

    # Apply PCA - get more components than we need
    n_components_to_compute = 4 if skip_first_pc else 3
    print(f"Computing PCA with {n_components_to_compute} components...")
    pca = PCA(n_components=n_components_to_compute, random_state=42)
    vectors_pca = pca.fit_transform(vectors)

    print(f"All explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    if skip_first_pc:
        print(f"\nSkipping PC1 (explains {pca.explained_variance_ratio_[0]:.1%} - likely just pos/neg split)")
        print(f"Using PC2, PC3, PC4 instead")
        vectors_3d = vectors_pca[:, 1:4]  # Use components 2, 3, 4
        pc_indices = [1, 2, 3]
        pc_labels = ["PC2", "PC3", "PC4"]
        variance_used = pca.explained_variance_ratio_[1:4]
    else:
        vectors_3d = vectors_pca[:, :3]  # Use components 1, 2, 3
        pc_indices = [0, 1, 2]
        pc_labels = ["PC1", "PC2", "PC3"]
        variance_used = pca.explained_variance_ratio_[:3]

    print(f"Variance explained by components used: {variance_used}")
    print(f"Total variance in plot: {sum(variance_used):.2%}")

    # Create DataFrame
    df = pd.DataFrame(metadata)
    df['x'] = vectors_3d[:, 0]
    df['y'] = vectors_3d[:, 1]
    df['z'] = vectors_3d[:, 2]

    # Create figure
    fig = go.Figure()

    # Add traces for each behavior
    for behavior in BEHAVIOR_COLORS.keys():
        behavior_df = df[df['behavior'] == behavior]

        # Positive and negative dataframes
        pos_df = behavior_df[behavior_df['type'] == 'positive']
        neg_df = behavior_df[behavior_df['type'] == 'negative']

        # Add direction lines if requested
        if show_direction_lines and not pos_df.empty and not neg_df.empty:
            mean_neg = np.array([neg_df['x'].mean(), neg_df['y'].mean(), neg_df['z'].mean()])
            mean_pos = np.array([pos_df['x'].mean(), pos_df['y'].mean(), pos_df['z'].mean()])

            # Calculate direction vector
            direction = mean_pos - mean_neg
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)  # Normalize

                # Get plot bounds (will be set properly after all data is added)
                # For now, estimate from data
                all_x = pd.concat([pos_df['x'], neg_df['x']])
                all_y = pd.concat([pos_df['y'], neg_df['y']])
                all_z = pd.concat([pos_df['z'], neg_df['z']])

                x_range = all_x.max() - all_x.min()
                y_range = all_y.max() - all_y.min()
                z_range = all_z.max() - all_z.min()
                max_range = max(x_range, y_range, z_range)

                # Extend line to cover the plot
                t_values = np.linspace(-max_range, max_range, 100)
                line_points = mean_neg[:, np.newaxis] + direction[:, np.newaxis] * t_values

                fig.add_trace(go.Scatter3d(
                    x=line_points[0, :],
                    y=line_points[1, :],
                    z=line_points[2, :],
                    mode='lines',
                    name=f"{behavior} (→)",
                    line=dict(
                        color=BEHAVIOR_COLORS[behavior],
                        width=3,
                        dash='solid'
                    ),
                    opacity=0.6,
                    hovertemplate=(f'<b>{behavior} Direction Vector</b><br>'
                                 f'Direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]<br>'
                                 f'(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})'),
                    legendgroup=behavior,
                    showlegend=True
                ))

                # Add markers at the mean points
                fig.add_trace(go.Scatter3d(
                    x=[mean_neg[0], mean_pos[0]],
                    y=[mean_neg[1], mean_pos[1]],
                    z=[mean_neg[2], mean_pos[2]],
                    mode='markers',
                    name=f"{behavior} means",
                    marker=dict(
                        size=[8, 12],
                        symbol=['circle-open', 'diamond'],
                        color=BEHAVIOR_COLORS[behavior],
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=(f'<b>{behavior} Mean Points</b><br>'
                                 f'%{{text}}<br>'
                                 f'(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})'),
                    text=['Negative mean', 'Positive mean'],
                    legendgroup=behavior,
                    showlegend=False
                ))

        # Positive points (dots)
        if not pos_df.empty:
            fig.add_trace(go.Scatter3d(
                x=pos_df['x'],
                y=pos_df['y'],
                z=pos_df['z'],
                mode='markers',
                name=f"{behavior} (+)",
                marker=dict(
                    color=BEHAVIOR_COLORS[behavior],
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=0.5),
                    opacity=0.9
                ),
                text=[f"<b>{behavior} (positive)</b><br>"
                      f"Index: {row['index']}<br>"
                      f"Q: {row['question']}<br>"
                      f"A: {row['answer']}"
                      for _, row in pos_df.iterrows()],
                hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}',
                legendgroup=behavior
            ))

        # Negative points (crosses)
        if not neg_df.empty:
            fig.add_trace(go.Scatter3d(
                x=neg_df['x'],
                y=neg_df['y'],
                z=neg_df['z'],
                mode='markers',
                name=f"{behavior} (-)",
                marker=dict(
                    color=BEHAVIOR_COLORS[behavior],
                    size=8,
                    symbol='cross',
                    line=dict(color='black', width=0.5),
                    opacity=0.9
                ),
                text=[f"<b>{behavior} (negative)</b><br>"
                      f"Index: {row['index']}<br>"
                      f"Q: {row['question']}<br>"
                      f"A: {row['answer']}"
                      for _, row in neg_df.iterrows()],
                hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}',
                legendgroup=behavior
            ))

    # Update layout
    title_text = f"7D Behavior Vectors - Layer {layer}<br>"
    if skip_first_pc:
        title_text += (f"<sub>Skipping PC1 ({pca.explained_variance_ratio_[0]:.1%} variance - pos/neg split)<br>"
                      f"Showing {pc_labels[0]}-{pc_labels[2]}: "
                      f"{variance_used[0]:.1%} + {variance_used[1]:.1%} + {variance_used[2]:.1%} = "
                      f"{sum(variance_used):.1%} variance</sub>")
    else:
        title_text += (f"<sub>PCA: {variance_used[0]:.1%} + "
                      f"{variance_used[1]:.1%} + {variance_used[2]:.1%} = "
                      f"{sum(variance_used):.1%} variance explained</sub>")

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title=f"{pc_labels[0]} ({variance_used[0]:.1%})",
            yaxis_title=f"{pc_labels[1]} ({variance_used[1]:.1%})",
            zaxis_title=f"{pc_labels[2]} ({variance_used[2]:.1%})",
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
        hovermode='closest'
    )

    # Add annotation
    fig.add_annotation(
        text="● = Positive (matching behavior)<br>✕ = Negative (not matching)",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.95,
        xanchor="left",
        yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )

    return fig

def analyze_clustering(vectors: np.ndarray, metadata: List[Dict]) -> None:
    """
    Analyze and print clustering statistics.
    """

    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)

    df = pd.DataFrame(metadata)

    # Count by behavior and type
    print("\nData distribution:")
    for behavior in BEHAVIOR_COLORS.keys():
        behavior_df = df[df['behavior'] == behavior]
        n_pos = len(behavior_df[behavior_df['type'] == 'positive'])
        n_neg = len(behavior_df[behavior_df['type'] == 'negative'])
        print(f"  {behavior}: {n_pos} positive, {n_neg} negative")

    # Analyze separation between positive and negative
    print("\nMean vectors by type:")
    pos_vectors = vectors[df['type'] == 'positive']
    neg_vectors = vectors[df['type'] == 'negative']

    print(f"  Positive mean: {pos_vectors.mean(axis=0).round(3)}")
    print(f"  Negative mean: {neg_vectors.mean(axis=0).round(3)}")

    # Distance between means
    pos_mean = pos_vectors.mean(axis=0)
    neg_mean = neg_vectors.mean(axis=0)
    distance = np.linalg.norm(pos_mean - neg_mean)
    print(f"  Distance between positive and negative means: {distance:.4f}")

    # Analyze by behavior
    print("\nMean vectors by behavior (difference vectors):")
    for behavior in BEHAVIOR_COLORS.keys():
        behavior_df = df[df['behavior'] == behavior]
        behavior_indices = behavior_df.index.tolist()
        if behavior_indices:
            behavior_vectors = vectors[behavior_indices]
            print(f"  {behavior}: {behavior_vectors.mean(axis=0).round(3)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize 7D behavior vectors')
    parser.add_argument('--layer', type=int, default=15,
                        help='Layer number to visualize')
    parser.add_argument('--data_dir', type=str, default='./behavior_vectors',
                        help='Directory containing behavior vector data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file')
    parser.add_argument('--no_browser', action='store_true',
                        help='Do not open browser automatically')
    parser.add_argument('--include_pc1', action='store_true',
                        help='Include PC1 (which mostly captures pos/neg split)')
    parser.add_argument('--show_directions', action='store_true',
                        help='Show direction lines from mean negative to mean positive for each behavior')

    args = parser.parse_args()

    # Load data
    try:
        vectors, metadata = load_behavior_vectors(args.layer, args.data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure you've run create_behavior_vectors.py for layer {args.layer} first.")
        return

    # Create visualization
    fig = create_3d_plot(vectors, metadata, args.layer,
                        skip_first_pc=not args.include_pc1,
                        show_direction_lines=args.show_directions)

    # Save
    output_file = args.output or f"behavior_vectors_7d_layer{args.layer}.html"
    fig.write_html(output_file)
    print(f"\nVisualization saved to: {output_file}")

    # Open in browser
    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        print("Opening in browser...")

    # Analyze clustering
    analyze_clustering(vectors, metadata)

if __name__ == "__main__":
    main()