#!/usr/bin/env python3
"""
Create weighted CAV vectors for each data point.
Each vector is a weighted combination of all 7 CAVs based on their projection strengths.
"""

import torch
import json
import numpy as np
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime

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

def load_normalized_cavs(layer: int, model_name: str = "Llama-2-7b-chat-hf") -> Dict[str, torch.Tensor]:
    """Load all normalized CAVs for a layer."""
    cavs = {}
    for behavior in TARGET_BEHAVIORS:
        cav_path = f"../normalized_vectors/{behavior}/vec_layer_{layer}_{model_name}.pt"
        if os.path.exists(cav_path):
            cavs[behavior] = torch.load(cav_path, map_location='cpu').to(torch.float64)
        else:
            print(f"Warning: CAV not found for {behavior} at layer {layer}")
    return cavs

def load_projection_data(behavior: str, layer: int, projection_dir: str = "./cav_projections") -> Dict:
    """Load projection data for a specific behavior and layer."""
    filepath = os.path.join(projection_dir, f"{behavior}_cav_projections_layer{layer}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Could not find {filepath}")
        return None

def create_weighted_cav_vector(projections: Dict[str, float], cavs: Dict[str, torch.Tensor],
                              normalize_weights: bool = True) -> torch.Tensor:
    """
    Create a single weighted CAV vector by combining all CAVs weighted by their projections.

    Args:
        projections: Dictionary mapping behavior names to projection values
        cavs: Dictionary mapping behavior names to CAV vectors
        normalize_weights: Whether to normalize the projection weights

    Returns:
        A single vector (same dimension as CAVs) that is the weighted sum of all CAVs
    """

    # Get the dimension from any CAV
    dim = next(iter(cavs.values())).shape[0]
    weighted_vector = torch.zeros(dim, dtype=torch.float64)

    # Collect weights
    weights = []
    behaviors_used = []

    for behavior in TARGET_BEHAVIORS:
        if behavior in projections and behavior in cavs:
            weights.append(projections[behavior])
            behaviors_used.append(behavior)

    if not weights:
        return weighted_vector

    weights = np.array(weights, dtype=np.float64)

    # Optionally normalize weights to sum to 1 (preserving sign)
    if normalize_weights and np.sum(np.abs(weights)) > 0:
        weights = weights / np.sum(np.abs(weights))

    # Create weighted combination
    for i, behavior in enumerate(behaviors_used):
        weighted_vector += weights[i] * cavs[behavior]

    return weighted_vector, dict(zip(behaviors_used, weights))

def process_all_data(layer: int, projection_dir: str = "./cav_projections",
                    output_dir: str = "./weighted_cav_vectors") -> Dict:
    """
    Process all behaviors and create weighted CAV vectors for each data point.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, load all CAVs
    print(f"Loading CAVs for layer {layer}...")
    cavs = load_normalized_cavs(layer)

    if not cavs:
        print("Error: No CAVs could be loaded!")
        return

    cav_dim = next(iter(cavs.values())).shape[0]
    print(f"Loaded {len(cavs)} CAVs, dimension: {cav_dim}")

    all_results = {}

    # Store all weighted vectors for analysis
    all_weighted_pos = []
    all_weighted_neg = []
    all_weighted_diff = []

    for source_behavior in tqdm(TARGET_BEHAVIORS, desc="Processing behaviors"):
        # Load projection data
        proj_data = load_projection_data(source_behavior, layer, projection_dir)
        if proj_data is None:
            continue

        behavior_results = {
            "behavior": source_behavior,
            "layer": layer,
            "cav_dimension": cav_dim,
            "normalized_cavs": {b: cav.tolist() for b, cav in cavs.items()},  # Include CAVs
            "data_points": []
        }

        for dp in proj_data["data_points"]:
            result = {
                "index": dp["index"],
                "question": dp["question"],
                "answer_matching_behavior": dp["answer_matching_behavior"],
                "answer_not_matching_behavior": dp["answer_not_matching_behavior"]
            }

            # Create weighted vector for positive case
            if "positive_case" in dp and "cav_projections" in dp["positive_case"]:
                weighted_vec, weights = create_weighted_cav_vector(
                    dp["positive_case"]["cav_projections"],
                    cavs,
                    normalize_weights=True
                )

                result["positive_weighted_vector"] = weighted_vec.tolist()
                result["positive_weighted_vector_norm"] = weighted_vec.norm().item()
                result["positive_weights"] = weights
                result["positive_raw_projections"] = dp["positive_case"]["cav_projections"]

                all_weighted_pos.append(weighted_vec)

            # Create weighted vector for negative case
            if "negative_case" in dp and "cav_projections" in dp["negative_case"]:
                weighted_vec, weights = create_weighted_cav_vector(
                    dp["negative_case"]["cav_projections"],
                    cavs,
                    normalize_weights=True
                )

                result["negative_weighted_vector"] = weighted_vec.tolist()
                result["negative_weighted_vector_norm"] = weighted_vec.norm().item()
                result["negative_weights"] = weights
                result["negative_raw_projections"] = dp["negative_case"]["cav_projections"]

                all_weighted_neg.append(weighted_vec)

            # Create weighted vector for difference
            if "difference" in dp and "cav_projections" in dp["difference"]:
                weighted_vec, weights = create_weighted_cav_vector(
                    dp["difference"]["cav_projections"],
                    cavs,
                    normalize_weights=True
                )

                result["difference_weighted_vector"] = weighted_vec.tolist()
                result["difference_weighted_vector_norm"] = weighted_vec.norm().item()
                result["difference_weights"] = weights
                result["difference_raw_projections"] = dp["difference"]["cav_projections"]

                # Compute similarity to source behavior's CAV
                if source_behavior in cavs:
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        weighted_vec.unsqueeze(0),
                        cavs[source_behavior].unsqueeze(0)
                    ).item()
                    result["difference_similarity_to_own_cav"] = cosine_sim

                all_weighted_diff.append(weighted_vec)

            behavior_results["data_points"].append(result)

        # Save individual behavior file
        output_file = os.path.join(output_dir, f"{source_behavior}_weighted_cavs_layer{layer}.json")

        # Convert to JSON-serializable format (excluding the full vectors for readability)
        save_data = {
            "behavior": behavior_results["behavior"],
            "layer": behavior_results["layer"],
            "cav_dimension": behavior_results["cav_dimension"],
            "num_datapoints": len(behavior_results["data_points"]),
            "data_points": behavior_results["data_points"]
        }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved weighted CAV vectors for {source_behavior}")
        all_results[source_behavior] = behavior_results

    # Compute and save statistics
    stats = compute_statistics(all_weighted_pos, all_weighted_neg, all_weighted_diff, cavs)

    stats_file = os.path.join(output_dir, f"weighted_cav_statistics_layer{layer}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved statistics to {stats_file}")

    # Also save a compact tensor file with all weighted vectors for easy loading
    tensor_file = os.path.join(output_dir, f"all_weighted_vectors_layer{layer}.pt")
    torch.save({
        "positive": torch.stack(all_weighted_pos) if all_weighted_pos else None,
        "negative": torch.stack(all_weighted_neg) if all_weighted_neg else None,
        "difference": torch.stack(all_weighted_diff) if all_weighted_diff else None,
        "cavs": cavs,
        "behaviors": TARGET_BEHAVIORS
    }, tensor_file)

    print(f"Saved tensor file to {tensor_file}")

    return all_results, stats

def compute_statistics(weighted_pos: List[torch.Tensor], weighted_neg: List[torch.Tensor],
                       weighted_diff: List[torch.Tensor], cavs: Dict[str, torch.Tensor]) -> Dict:
    """
    Compute statistics on the weighted CAV vectors.
    """

    stats = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_vectors_positive": len(weighted_pos),
            "num_vectors_negative": len(weighted_neg),
            "num_vectors_difference": len(weighted_diff),
            "vector_dimension": cavs[TARGET_BEHAVIORS[0]].shape[0] if cavs else 0
        },
        "positive_case": {},
        "negative_case": {},
        "difference": {}
    }

    # Compute statistics for each case
    if weighted_pos:
        stacked_pos = torch.stack(weighted_pos)
        stats["positive_case"] = {
            "mean_norm": float(torch.norm(stacked_pos, dim=1).mean()),
            "std_norm": float(torch.norm(stacked_pos, dim=1).std()),
            "mean_vector_norm": float(stacked_pos.mean(dim=0).norm()),
        }

        # Similarity to each CAV
        for behavior, cav in cavs.items():
            sims = [torch.nn.functional.cosine_similarity(v.unsqueeze(0), cav.unsqueeze(0)).item()
                   for v in weighted_pos]
            stats["positive_case"][f"mean_similarity_to_{behavior}"] = float(np.mean(sims))
            stats["positive_case"][f"std_similarity_to_{behavior}"] = float(np.std(sims))

    if weighted_neg:
        stacked_neg = torch.stack(weighted_neg)
        stats["negative_case"] = {
            "mean_norm": float(torch.norm(stacked_neg, dim=1).mean()),
            "std_norm": float(torch.norm(stacked_neg, dim=1).std()),
            "mean_vector_norm": float(stacked_neg.mean(dim=0).norm()),
        }

        # Similarity to each CAV
        for behavior, cav in cavs.items():
            sims = [torch.nn.functional.cosine_similarity(v.unsqueeze(0), cav.unsqueeze(0)).item()
                   for v in weighted_neg]
            stats["negative_case"][f"mean_similarity_to_{behavior}"] = float(np.mean(sims))
            stats["negative_case"][f"std_similarity_to_{behavior}"] = float(np.std(sims))

    if weighted_diff:
        stacked_diff = torch.stack(weighted_diff)
        stats["difference"] = {
            "mean_norm": float(torch.norm(stacked_diff, dim=1).mean()),
            "std_norm": float(torch.norm(stacked_diff, dim=1).std()),
            "mean_vector_norm": float(stacked_diff.mean(dim=0).norm()),
        }

        # Similarity to each CAV
        for behavior, cav in cavs.items():
            sims = [torch.nn.functional.cosine_similarity(v.unsqueeze(0), cav.unsqueeze(0)).item()
                   for v in weighted_diff]
            stats["difference"][f"mean_similarity_to_{behavior}"] = float(np.mean(sims))
            stats["difference"][f"std_similarity_to_{behavior}"] = float(np.std(sims))

        # Check orthogonality between CAVs
        orthogonality = {}
        for b1 in TARGET_BEHAVIORS:
            for b2 in TARGET_BEHAVIORS:
                if b1 < b2 and b1 in cavs and b2 in cavs:
                    sim = torch.nn.functional.cosine_similarity(
                        cavs[b1].unsqueeze(0), cavs[b2].unsqueeze(0)
                    ).item()
                    orthogonality[f"{b1}_vs_{b2}"] = float(sim)

        stats["cav_orthogonality"] = orthogonality

    return stats

def main():
    parser = argparse.ArgumentParser(description='Create weighted CAV vectors')
    parser.add_argument('--layers', nargs='+', type=int, default=[15],
                        help='Layer numbers to process (e.g., --layers 10 15 20 25)')
    parser.add_argument('--projection_dir', type=str, default='./cav_projections',
                        help='Directory containing projection data')
    parser.add_argument('--output_dir', type=str, default='./weighted_cav_vectors',
                        help='Output directory for weighted vectors')

    args = parser.parse_args()

    print(f"Creating weighted CAV vectors for layers: {args.layers}")

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)

        results, stats = process_all_data(layer, args.projection_dir, args.output_dir)

        # Print summary
        if stats and "difference" in stats:
            print(f"\n{'-'*40}")
            print(f"Layer {layer} Summary:")
            print(f"{'-'*40}")
            print(f"Vector dimension: {stats['metadata']['vector_dimension']}")
            print(f"Number of weighted vectors: {stats['metadata']['num_vectors_difference']}")

            if "mean_norm" in stats["difference"]:
                print(f"Mean norm of weighted difference vectors: {stats['difference']['mean_norm']:.4f}")

            print("\nMean similarity to each CAV (difference vectors):")
            for behavior in TARGET_BEHAVIORS:
                key = f"mean_similarity_to_{behavior}"
                if key in stats["difference"]:
                    print(f"  {behavior}: {stats['difference'][key]:.4f}")

if __name__ == "__main__":
    main()