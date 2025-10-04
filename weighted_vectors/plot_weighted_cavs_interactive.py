"""
Interactive 3D visualization of weighted CAVs with parameter sliders.
Uses Plotly Dash for live parameter adjustment.

Usage:
python plot_weighted_cavs_interactive.py --layer 14 --model_size 7b
Then open browser to http://localhost:8050
"""

import torch as t
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import argparse
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from pathlib import Path

# Color palette for 7 behaviors
BEHAVIOR_COLORS = {
    "envy-kindness": "#FF6B6B",
    "gluttony-temperance": "#FFA500",
    "greed-charity": "#FFD700",
    "lust-chastity": "#9370DB",
    "pride-humility": "#4169E1",
    "sloth-diligence": "#2E8B57",
    "wrath-patience": "#DC143C",
}

BEHAVIORS = [
    "envy-kindness", "gluttony-temperance", "greed-charity",
    "lust-chastity", "pride-humility", "sloth-diligence", "wrath-patience"
]

def load_weighted_cavs(layer, model_size, use_base_model=False):
    """Load all weighted CAVs for specified layer."""
    vectors_dir = Path(__file__).parent / "vectors"

    all_cavs = []
    all_labels = []
    all_metadata = []

    model_suffix = "base" if use_base_model else "7b"

    for behavior in BEHAVIORS:
        pt_file = vectors_dir / f"{behavior}_weighted_vectors_layer{layer}_{model_suffix}.pt"
        json_file = vectors_dir / f"{behavior}_weighted_vectors_layer{layer}_{model_suffix}.json"

        if not pt_file.exists():
            continue

        data_pt = t.load(pt_file, weights_only=False)
        weighted_vecs = data_pt["weighted_vectors"]

        with open(json_file) as f:
            data = json.load(f)

        vecs_np = weighted_vecs.numpy()

        all_cavs.append(vecs_np)
        all_labels.extend([behavior] * len(vecs_np))
        all_metadata.extend(data['data_points'])

    cavs = np.vstack(all_cavs)
    return cavs, all_labels, all_metadata

def reduce_dimensions(cavs, method='pca', params=None):
    """Reduce dimensions with specified method and parameters."""
    params = params or {}

    if method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        reduced = reducer.fit_transform(cavs)
        info = f"Variance: {reducer.explained_variance_ratio_.sum():.2%}"

    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(
                n_components=3,
                n_neighbors=params.get('n_neighbors', 15),
                min_dist=params.get('min_dist', 0.1),
                random_state=42
            )
            reduced = reducer.fit_transform(cavs)
            info = f"n_neighbors={params.get('n_neighbors', 15)}, min_dist={params.get('min_dist', 0.1)}"
        except ImportError:
            return reduce_dimensions(cavs, 'pca')

    elif method == 'tsne':
        reducer = TSNE(
            n_components=3,
            perplexity=params.get('perplexity', 30),
            learning_rate=params.get('learning_rate', 200),
            random_state=42
        )
        reduced = reducer.fit_transform(cavs)
        info = f"perplexity={params.get('perplexity', 30)}, lr={params.get('learning_rate', 200)}"

    elif method == 'mds':
        reducer = MDS(
            n_components=3,
            metric=params.get('metric', True),
            random_state=42
        )
        reduced = reducer.fit_transform(cavs)
        info = f"metric={'Yes' if params.get('metric', True) else 'No'}"

    elif method == 'isomap':
        reducer = Isomap(
            n_components=3,
            n_neighbors=params.get('n_neighbors', 15)
        )
        reduced = reducer.fit_transform(cavs)
        info = f"n_neighbors={params.get('n_neighbors', 15)}"

    elif method == 'lle':
        reducer = LocallyLinearEmbedding(
            n_components=3,
            n_neighbors=params.get('n_neighbors', 15),
            random_state=42
        )
        reduced = reducer.fit_transform(cavs)
        info = f"n_neighbors={params.get('n_neighbors', 15)}"

    return reduced, info

def create_plot(reduced, labels, metadata, method='PCA', info=''):
    """Create 3D scatter plot."""
    fig = go.Figure()

    for behavior in BEHAVIORS:
        mask = np.array([l == behavior for l in labels])
        if mask.sum() > 0:
            behavior_metadata = [m for l, m in zip(labels, metadata) if l == behavior]

            hover_texts = []
            for i, meta in enumerate(behavior_metadata):
                question = meta['question'].split('\n')[0][:60] + "..."
                answer = meta['answer_matching_behavior']
                hover_texts.append(
                    f"<b>{behavior}</b> (Q{i})<br>"
                    f"Question: {question}<br>"
                    f"<b>Matching: {answer}</b><br>"
                    f"Not matching: {meta['answer_not_matching_behavior']}"
                )

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
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title=f"70 Weighted CAVs - {method.upper()}<br><sub>{info}</sub>",
        scene=dict(
            xaxis_title=f"{method} 1",
            yaxis_title=f"{method} 2",
            zaxis_title=f"{method} 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1400,
        height=900,
        hovermode='closest',
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )

    return fig

def create_app(cavs, labels, metadata):
    """Create Dash app with interactive controls."""
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Interactive Weighted CAV Visualization", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label("Reduction Method:"),
                dcc.Dropdown(
                    id='method-dropdown',
                    options=[
                        {'label': 'PCA (Fast, Linear)', 'value': 'pca'},
                        {'label': 'UMAP (Balanced)', 'value': 'umap'},
                        {'label': 't-SNE (Clusters)', 'value': 'tsne'},
                        {'label': 'MDS (Distances)', 'value': 'mds'},
                        {'label': 'Isomap (Manifold)', 'value': 'isomap'},
                        {'label': 'LLE (Local)', 'value': 'lle'},
                    ],
                    value='pca',
                    style={'width': '300px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # Parameter sliders (visibility controlled by method)
        html.Div(id='parameter-controls', children=[
            html.Div([
                html.Label("n_neighbors (UMAP/Isomap/LLE):"),
                dcc.Slider(id='n-neighbors-slider', min=5, max=50, step=5, value=15,
                          marks={i: str(i) for i in range(5, 51, 10)},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], id='n-neighbors-div', style={'marginBottom': '20px'}),

            html.Div([
                html.Label("min_dist (UMAP):"),
                dcc.Slider(id='min-dist-slider', min=0.0, max=1.0, step=0.1, value=0.1,
                          marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], id='min-dist-div', style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Perplexity (t-SNE):"),
                dcc.Slider(id='perplexity-slider', min=5, max=50, step=5, value=30,
                          marks={i: str(i) for i in range(5, 51, 10)},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], id='perplexity-div', style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Learning Rate (t-SNE):"),
                dcc.Slider(id='learning-rate-slider', min=10, max=1000, step=10, value=200,
                          marks={i: str(i) for i in range(0, 1001, 200)},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], id='learning-rate-div', style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Metric (MDS):"),
                dcc.RadioItems(
                    id='metric-radio',
                    options=[
                        {'label': 'Metric (preserves distances)', 'value': 'true'},
                        {'label': 'Non-metric (preserves order)', 'value': 'false'}
                    ],
                    value='true',
                    inline=True
                ),
            ], id='metric-div', style={'marginBottom': '20px'}),
        ], style={'width': '80%', 'margin': 'auto'}),

        html.Button('Update Visualization', id='update-button', n_clicks=0,
                   style={'display': 'block', 'margin': '20px auto',
                         'padding': '10px 30px', 'fontSize': '16px'}),

        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id='3d-plot', style={'height': '900px'})
        ),

        html.Div(id='method-info', style={'textAlign': 'center', 'marginTop': '20px'}),

    ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

    @app.callback(
        [Output('n-neighbors-div', 'style'),
         Output('min-dist-div', 'style'),
         Output('perplexity-div', 'style'),
         Output('learning-rate-div', 'style'),
         Output('metric-div', 'style')],
        Input('method-dropdown', 'value')
    )
    def toggle_controls(method):
        """Show/hide relevant parameter controls."""
        base_style = {'marginBottom': '20px'}
        hidden_style = {'display': 'none'}

        n_neighbors_style = base_style if method in ['umap', 'isomap', 'lle'] else hidden_style
        min_dist_style = base_style if method == 'umap' else hidden_style
        perplexity_style = base_style if method == 'tsne' else hidden_style
        learning_rate_style = base_style if method == 'tsne' else hidden_style
        metric_style = base_style if method == 'mds' else hidden_style

        return n_neighbors_style, min_dist_style, perplexity_style, learning_rate_style, metric_style

    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('method-info', 'children')],
        Input('update-button', 'n_clicks'),
        [State('method-dropdown', 'value'),
         State('n-neighbors-slider', 'value'),
         State('min-dist-slider', 'value'),
         State('perplexity-slider', 'value'),
         State('learning-rate-slider', 'value'),
         State('metric-radio', 'value')]
    )
    def update_plot(n_clicks, method, n_neighbors, min_dist, perplexity, learning_rate, metric):
        """Update plot with new parameters."""
        params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'metric': metric == 'true'
        }

        reduced, info = reduce_dimensions(cavs, method, params)
        fig = create_plot(reduced, labels, metadata, method, info)

        info_text = f"Current method: {method.upper()} | {info}"
        return fig, info_text

    return app

def main():
    parser = argparse.ArgumentParser(description="Interactive weighted CAV visualization")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    print(f"Loading weighted CAVs from layer {args.layer}...")
    cavs, labels, metadata = load_weighted_cavs(args.layer, args.model_size, args.use_base_model)
    print(f"Loaded {len(cavs)} weighted CAVs")

    print(f"\nStarting interactive dashboard on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    app = create_app(cavs, labels, metadata)
    app.run_server(debug=True, port=args.port)

if __name__ == "__main__":
    main()
