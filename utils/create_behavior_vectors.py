#!/usr/bin/env python3
"""
Create behavior vectors for each data point where each component represents
the weighted presence of each behavior based on normalized CAV projections.
"""

import torch
import json
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# The 7 behaviors we're working with
TARGET_BEHAVIORS = [
    "envy-kindness",
    "gluttony-temperance",
    "greed-charity",
    "lust-chastity",
    "pride-humility",
    "sloth-diligence",
    "wrath-patience"
]

def load_projection_data(behavior: str, layer: int, projection_dir: str = "./cav_projections") -> Dict:
    """Load projection data for a specific behavior and layer."""
    filepath = os.path.join(projection_dir, f"{behavior}_cav_projections_layer{layer}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Could not find {filepath}")
        return None

def create_behavior_vector(projections: Dict[str, float], normalize: bool = True) -> np.ndarray:
    """
    Create a 7-dimensional behavior vector from CAV projections.
    Each component represents the strength of that behavior.

    Args:
        projections: Dictionary mapping behavior names to projection values
        normalize: Whether to normalize the vector to unit length

    Returns:
        7-dimensional numpy array ordered by TARGET_BEHAVIORS
    """
    vector = np.zeros(len(TARGET_BEHAVIORS), dtype=np.float64)

    for i, behavior in enumerate(TARGET_BEHAVIORS):
        if behavior in projections:
            vector[i] = projections[behavior]

    if normalize and np.linalg.norm(vector) > 0:
        vector = vector / np.linalg.norm(vector)

    return vector

def process_all_behaviors(layer: int, projection_dir: str = "./cav_projections",
                         output_dir: str = "./behavior_vectors") -> Dict:
    """
    Process all behaviors and create behavior vectors for each data point.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results = {}
    all_vectors_pos = []
    all_vectors_neg = []
    all_vectors_diff = []
    all_metadata = []

    for behavior in tqdm(TARGET_BEHAVIORS, desc="Processing behaviors"):
        # Load projection data
        data = load_projection_data(behavior, layer, projection_dir)
        if data is None:
            continue

        behavior_results = {
            "behavior": behavior,
            "layer": layer,
            "data_points": []
        }

        for dp in data["data_points"]:
            result = {
                "index": dp["index"],
                "question": dp["question"],
                "answer_matching_behavior": dp["answer_matching_behavior"],
                "answer_not_matching_behavior": dp["answer_not_matching_behavior"]
            }

            # Create behavior vectors for positive case
            if "positive_case" in dp and "cav_projections" in dp["positive_case"]:
                pos_vector = create_behavior_vector(dp["positive_case"]["cav_projections"], normalize=True)
                result["positive_behavior_vector"] = pos_vector.tolist()
                result["positive_behavior_vector_norm"] = np.linalg.norm(
                    create_behavior_vector(dp["positive_case"]["cav_projections"], normalize=False)
                )
                all_vectors_pos.append(pos_vector)

                # Also store raw projections for reference
                result["positive_raw_projections"] = dp["positive_case"]["cav_projections"]

            # Create behavior vectors for negative case
            if "negative_case" in dp and "cav_projections" in dp["negative_case"]:
                neg_vector = create_behavior_vector(dp["negative_case"]["cav_projections"], normalize=True)
                result["negative_behavior_vector"] = neg_vector.tolist()
                result["negative_behavior_vector_norm"] = np.linalg.norm(
                    create_behavior_vector(dp["negative_case"]["cav_projections"], normalize=False)
                )
                all_vectors_neg.append(neg_vector)

                # Also store raw projections
                result["negative_raw_projections"] = dp["negative_case"]["cav_projections"]

            # Create behavior vector for difference
            if "difference" in dp and "cav_projections" in dp["difference"]:
                diff_vector = create_behavior_vector(dp["difference"]["cav_projections"], normalize=True)
                result["difference_behavior_vector"] = diff_vector.tolist()
                result["difference_behavior_vector_norm"] = np.linalg.norm(
                    create_behavior_vector(dp["difference"]["cav_projections"], normalize=False)
                )
                all_vectors_diff.append(diff_vector)

                # Store raw projections
                result["difference_raw_projections"] = dp["difference"]["cav_projections"]

            behavior_results["data_points"].append(result)

            # Store metadata for analysis
            all_metadata.append({
                "source_behavior": behavior,
                "index": dp["index"],
                "question": dp["question"][:100] + "..."
            })

        all_results[behavior] = behavior_results

        # Save individual behavior file
        output_file = os.path.join(output_dir, f"{behavior}_behavior_vectors_layer{layer}.json")
        with open(output_file, 'w') as f:
            json.dump(behavior_results, f, indent=2)
        print(f"Saved behavior vectors for {behavior}")

    # Compute statistics
    stats = compute_statistics(all_vectors_pos, all_vectors_neg, all_vectors_diff, all_metadata)

    # Save statistics
    stats_file = os.path.join(output_dir, f"behavior_vector_statistics_layer{layer}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved statistics to {stats_file}")

    return all_results, stats

def compute_statistics(vectors_pos: List[np.ndarray], vectors_neg: List[np.ndarray],
                       vectors_diff: List[np.ndarray], metadata: List[Dict]) -> Dict:
    """
    Compute comprehensive statistics on behavior vectors.
    """

    stats = {
        "metadata": {
            "num_datapoints": len(metadata),
            "behaviors": TARGET_BEHAVIORS,
            "timestamp": datetime.now().isoformat()
        },
        "positive_case": {},
        "negative_case": {},
        "difference": {}
    }

    # Convert lists to numpy arrays for easier computation
    if vectors_pos:
        pos_array = np.array(vectors_pos)  # Shape: (N, 7)
        stats["positive_case"] = {
            "mean_vector": pos_array.mean(axis=0).tolist(),
            "std_vector": pos_array.std(axis=0).tolist(),
            "component_means": {behavior: float(pos_array[:, i].mean())
                               for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "component_stds": {behavior: float(pos_array[:, i].std())
                              for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "correlation_matrix": compute_correlation_matrix(pos_array),
            "dominant_behavior_counts": count_dominant_behaviors(pos_array)
        }

    if vectors_neg:
        neg_array = np.array(vectors_neg)
        stats["negative_case"] = {
            "mean_vector": neg_array.mean(axis=0).tolist(),
            "std_vector": neg_array.std(axis=0).tolist(),
            "component_means": {behavior: float(neg_array[:, i].mean())
                               for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "component_stds": {behavior: float(neg_array[:, i].std())
                              for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "correlation_matrix": compute_correlation_matrix(neg_array),
            "dominant_behavior_counts": count_dominant_behaviors(neg_array)
        }

    if vectors_diff:
        diff_array = np.array(vectors_diff)
        stats["difference"] = {
            "mean_vector": diff_array.mean(axis=0).tolist(),
            "std_vector": diff_array.std(axis=0).tolist(),
            "component_means": {behavior: float(diff_array[:, i].mean())
                               for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "component_stds": {behavior: float(diff_array[:, i].std())
                              for i, behavior in enumerate(TARGET_BEHAVIORS)},
            "correlation_matrix": compute_correlation_matrix(diff_array),
            "dominant_behavior_counts": count_dominant_behaviors(diff_array),
            "behavior_separability": compute_behavior_separability(diff_array, metadata)
        }

    return stats

def compute_correlation_matrix(vectors: np.ndarray) -> Dict:
    """
    Compute correlation between behavior components across all vectors.
    """
    corr_matrix = np.corrcoef(vectors.T)  # Shape: (7, 7)

    result = {}
    for i, behavior_i in enumerate(TARGET_BEHAVIORS):
        result[behavior_i] = {}
        for j, behavior_j in enumerate(TARGET_BEHAVIORS):
            result[behavior_i][behavior_j] = float(corr_matrix[i, j])

    return result

def count_dominant_behaviors(vectors: np.ndarray) -> Dict[str, int]:
    """
    Count how often each behavior is the dominant component in vectors.
    """
    dominant_indices = np.argmax(np.abs(vectors), axis=1)
    counts = {behavior: 0 for behavior in TARGET_BEHAVIORS}

    for idx in dominant_indices:
        counts[TARGET_BEHAVIORS[idx]] += 1

    return counts

def compute_behavior_separability(diff_vectors: np.ndarray, metadata: List[Dict]) -> Dict:
    """
    Compute how well behavior vectors separate different source behaviors.
    This measures whether vectors from the same behavior cluster together.
    """

    # Group vectors by source behavior
    behavior_vectors = {behavior: [] for behavior in TARGET_BEHAVIORS}

    for i, meta in enumerate(metadata):
        if i < len(diff_vectors):
            behavior_vectors[meta["source_behavior"]].append(diff_vectors[i])

    # Compute mean vector for each behavior
    behavior_means = {}
    for behavior, vecs in behavior_vectors.items():
        if vecs:
            behavior_means[behavior] = np.mean(vecs, axis=0)

    # Compute inter-behavior distances (between different behaviors)
    inter_distances = []
    for b1 in behavior_means:
        for b2 in behavior_means:
            if b1 < b2:  # Avoid duplicates
                dist = np.linalg.norm(behavior_means[b1] - behavior_means[b2])
                inter_distances.append(dist)

    # Compute intra-behavior distances (within same behavior)
    intra_distances = []
    for behavior, vecs in behavior_vectors.items():
        if len(vecs) > 1 and behavior in behavior_means:
            mean_vec = behavior_means[behavior]
            for vec in vecs:
                dist = np.linalg.norm(vec - mean_vec)
                intra_distances.append(dist)

    separability = {
        "mean_inter_behavior_distance": float(np.mean(inter_distances)) if inter_distances else 0,
        "mean_intra_behavior_distance": float(np.mean(intra_distances)) if intra_distances else 0,
        "separability_ratio": float(np.mean(inter_distances) / np.mean(intra_distances))
                             if inter_distances and intra_distances else 0,
        "interpretation": "Higher ratio means behaviors are better separated"
    }

    return separability

def create_visualization(stats: Dict, layer: int, output_dir: str = "./behavior_vectors"):
    """
    Create visualization of behavior vector statistics.
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Behavior Vector Analysis - Layer {layer}', fontsize=16)

    # Plot 1: Mean behavior vectors
    ax = axes[0, 0]
    x = np.arange(len(TARGET_BEHAVIORS))
    width = 0.25

    if "positive_case" in stats and "mean_vector" in stats["positive_case"]:
        ax.bar(x - width, stats["positive_case"]["mean_vector"], width, label='Positive', alpha=0.7)
    if "negative_case" in stats and "mean_vector" in stats["negative_case"]:
        ax.bar(x, stats["negative_case"]["mean_vector"], width, label='Negative', alpha=0.7)
    if "difference" in stats and "mean_vector" in stats["difference"]:
        ax.bar(x + width, stats["difference"]["mean_vector"], width, label='Difference', alpha=0.7)

    ax.set_xlabel('Behavior')
    ax.set_ylabel('Mean Component Value')
    ax.set_title('Mean Behavior Vectors')
    ax.set_xticks(x)
    ax.set_xticklabels([b.split('-')[0] for b in TARGET_BEHAVIORS], rotation=45)
    ax.legend()

    # Plot 2: Correlation matrix for difference vectors
    ax = axes[0, 1]
    if "difference" in stats and "correlation_matrix" in stats["difference"]:
        corr_data = []
        for b1 in TARGET_BEHAVIORS:
            row = []
            for b2 in TARGET_BEHAVIORS:
                row.append(stats["difference"]["correlation_matrix"][b1][b2])
            corr_data.append(row)

        im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(TARGET_BEHAVIORS)))
        ax.set_yticks(range(len(TARGET_BEHAVIORS)))
        ax.set_xticklabels([b.split('-')[0] for b in TARGET_BEHAVIORS], rotation=45)
        ax.set_yticklabels([b.split('-')[0] for b in TARGET_BEHAVIORS])
        ax.set_title('Behavior Correlation Matrix (Diff)')
        plt.colorbar(im, ax=ax)

    # Plot 3: Dominant behavior counts
    ax = axes[0, 2]
    if "difference" in stats and "dominant_behavior_counts" in stats["difference"]:
        counts = stats["difference"]["dominant_behavior_counts"]
        behaviors_short = [b.split('-')[0] for b in TARGET_BEHAVIORS]
        values = [counts[b] for b in TARGET_BEHAVIORS]

        ax.bar(behaviors_short, values)
        ax.set_xlabel('Behavior')
        ax.set_ylabel('Count')
        ax.set_title('Dominant Behavior Frequency')
        ax.set_xticklabels(behaviors_short, rotation=45)

    # Plot 4: Component standard deviations
    ax = axes[1, 0]
    if "difference" in stats and "component_stds" in stats["difference"]:
        stds = [stats["difference"]["component_stds"][b] for b in TARGET_BEHAVIORS]
        behaviors_short = [b.split('-')[0] for b in TARGET_BEHAVIORS]

        ax.bar(behaviors_short, stds)
        ax.set_xlabel('Behavior')
        ax.set_ylabel('Std Dev')
        ax.set_title('Component Variability (Diff)')
        ax.set_xticklabels(behaviors_short, rotation=45)

    # Plot 5: Separability metrics
    ax = axes[1, 1]
    if "difference" in stats and "behavior_separability" in stats["difference"]:
        sep = stats["difference"]["behavior_separability"]
        labels = ['Inter-behavior\nDistance', 'Intra-behavior\nDistance']
        values = [sep["mean_inter_behavior_distance"], sep["mean_intra_behavior_distance"]]

        ax.bar(labels, values)
        ax.set_ylabel('Distance')
        ax.set_title(f'Behavior Separability\n(Ratio: {sep["separability_ratio"]:.2f})')

    # Plot 6: Text summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"Layer {layer} Summary\n\n"
    summary_text += f"Total datapoints: {stats['metadata']['num_datapoints']}\n"

    if "difference" in stats and "behavior_separability" in stats["difference"]:
        sep = stats["difference"]["behavior_separability"]
        summary_text += f"\nSeparability ratio: {sep['separability_ratio']:.2f}\n"
        summary_text += "(Higher = better behavior separation)\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f"behavior_vectors_analysis_layer{layer}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved visualization to {fig_path}")

def main():
    parser = argparse.ArgumentParser(description='Create behavior vectors from CAV projections')
    parser.add_argument('--layers', nargs='+', type=int, default=[15],
                        help='Layer numbers to process (e.g., --layers 10 15 20 25)')
    parser.add_argument('--projection_dir', type=str, default='./cav_projections',
                        help='Directory containing projection data')
    parser.add_argument('--output_dir', type=str, default='./behavior_vectors',
                        help='Output directory for behavior vectors')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')

    args = parser.parse_args()

    print(f"Creating behavior vectors for layers: {args.layers}")

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)

        results, stats = process_all_behaviors(layer, args.projection_dir, args.output_dir)

        # Print summary statistics
        print(f"\n{'-'*40}")
        print(f"Layer {layer} Statistics Summary:")
        print(f"{'-'*40}")

        if "difference" in stats:
            print("\nDominant behaviors in difference vectors:")
            for behavior, count in stats["difference"]["dominant_behavior_counts"].items():
                print(f"  {behavior}: {count}")

            if "behavior_separability" in stats["difference"]:
                sep = stats["difference"]["behavior_separability"]
                print(f"\nBehavior separability:")
                print(f"  Inter-behavior distance: {sep['mean_inter_behavior_distance']:.4f}")
                print(f"  Intra-behavior distance: {sep['mean_intra_behavior_distance']:.4f}")
                print(f"  Separability ratio: {sep['separability_ratio']:.2f}")

        if args.visualize:
            create_visualization(stats, layer, args.output_dir)

if __name__ == "__main__":
    main()