#!/usr/bin/env python3
"""
Compute the projection of all training data activations onto all CAVs.
This shows how much each behavior's signal is present in each activation.
"""

import torch
import json
import os
import numpy as np
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

def load_normalized_cav(behavior: str, layer: int, model_name: str = "Llama-2-7b-chat-hf") -> torch.Tensor:
    """Load a normalized CAV for a behavior."""
    cav_path = f"../normalized_vectors/{behavior}/vec_layer_{layer}_{model_name}.pt"
    if os.path.exists(cav_path):
        return torch.load(cav_path, map_location='cpu').to(torch.float64)  # Use float64 for precision
    else:
        print(f"Warning: CAV not found at {cav_path}")
        return None

def load_training_data_and_activations(behavior: str, layer: int, model_name: str = "Llama-2-7b-chat-hf") -> Tuple[List, torch.Tensor, torch.Tensor]:
    """Load training data and activations for a behavior."""

    # Load training data
    data_paths = [
        f"../datasets/generate/{behavior}/generate_dataset.json",
        f"../datasets/generate/{behavior}/generate_dataset.json"
    ]

    training_data = None
    for path in data_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                training_data = json.load(f)
            break

    if training_data is None:
        print(f"Warning: Could not find training data for {behavior}")
        return None, None, None

    # Load activations
    pos_path = f"../activations/{behavior}/activations_pos_{layer}_{model_name}.pt"
    neg_path = f"../activations/{behavior}/activations_neg_{layer}_{model_name}.pt"

    pos_activations = None
    neg_activations = None

    if os.path.exists(pos_path):
        pos_activations = torch.load(pos_path, map_location='cpu').to(torch.float64)
    else:
        print(f"Warning: Positive activations not found for {behavior}")

    if os.path.exists(neg_path):
        neg_activations = torch.load(neg_path, map_location='cpu').to(torch.float64)
    else:
        print(f"Warning: Negative activations not found for {behavior}")

    return training_data, pos_activations, neg_activations

def compute_projection(activation: torch.Tensor, cav: torch.Tensor) -> float:
    """
    Compute the projection of an activation onto a CAV.
    This is the dot product, which measures how much the CAV is present in the activation.
    """
    return torch.dot(activation, cav).item()

def compute_all_projections(layer: int, model_name: str = "Llama-2-7b-chat-hf", output_dir: str = "./cav_projections"):
    """
    Compute projections of all activations onto all CAVs.
    """

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, load all CAVs
    cavs = {}
    for behavior in TARGET_BEHAVIORS:
        cav = load_normalized_cav(behavior, layer, model_name)
        if cav is not None:
            cavs[behavior] = cav
            print(f"Loaded CAV for {behavior}: shape {cav.shape}, norm {cav.norm().item():.6f}")

    if not cavs:
        print("Error: No CAVs could be loaded!")
        return

    # Process each behavior's training data
    all_results = {}

    for source_behavior in tqdm(TARGET_BEHAVIORS, desc="Processing behaviors"):
        print(f"\nProcessing {source_behavior}...")

        # Load data and activations
        training_data, pos_activations, neg_activations = load_training_data_and_activations(
            source_behavior, layer, model_name
        )

        if training_data is None:
            print(f"Skipping {source_behavior} - no data found")
            continue

        behavior_results = {
            "behavior": source_behavior,
            "layer": layer,
            "model": model_name,
            "normalized_cav": cavs.get(source_behavior).tolist() if source_behavior in cavs else None,
            "cav_norm": cavs.get(source_behavior).norm().item() if source_behavior in cavs else None,
            "data_points": []
        }

        # Process each training example
        for idx, example in enumerate(training_data):
            data_point = {
                "index": idx,
                "question": example["question"],
                "answer_matching_behavior": example["answer_matching_behavior"],
                "answer_not_matching_behavior": example["answer_not_matching_behavior"],
                "positive_case": {},
                "negative_case": {}
            }

            # Compute projections for positive activation
            if pos_activations is not None and idx < len(pos_activations):
                pos_act = pos_activations[idx]
                data_point["positive_case"]["activation_norm"] = pos_act.norm().item()

                projections = {}
                for cav_behavior, cav in cavs.items():
                    projection = compute_projection(pos_act, cav)
                    projections[cav_behavior] = projection

                data_point["positive_case"]["cav_projections"] = projections

                # Also compute cosine similarity for normalized comparison
                cosine_sims = {}
                pos_act_normalized = pos_act / (pos_act.norm() + 1e-10)
                for cav_behavior, cav in cavs.items():
                    cav_normalized = cav / (cav.norm() + 1e-10)
                    cosine_sim = torch.dot(pos_act_normalized, cav_normalized).item()
                    cosine_sims[cav_behavior] = cosine_sim

                data_point["positive_case"]["cav_cosine_similarities"] = cosine_sims

            # Compute projections for negative activation
            if neg_activations is not None and idx < len(neg_activations):
                neg_act = neg_activations[idx]
                data_point["negative_case"]["activation_norm"] = neg_act.norm().item()

                projections = {}
                for cav_behavior, cav in cavs.items():
                    projection = compute_projection(neg_act, cav)
                    projections[cav_behavior] = projection

                data_point["negative_case"]["cav_projections"] = projections

                # Also compute cosine similarity
                cosine_sims = {}
                neg_act_normalized = neg_act / (neg_act.norm() + 1e-10)
                for cav_behavior, cav in cavs.items():
                    cav_normalized = cav / (cav.norm() + 1e-10)
                    cosine_sim = torch.dot(neg_act_normalized, cav_normalized).item()
                    cosine_sims[cav_behavior] = cosine_sim

                data_point["negative_case"]["cav_cosine_similarities"] = cosine_sims

            # Compute difference projections (pos - neg)
            if pos_activations is not None and neg_activations is not None:
                if idx < len(pos_activations) and idx < len(neg_activations):
                    diff_act = pos_activations[idx] - neg_activations[idx]
                    data_point["difference"] = {
                        "activation_norm": diff_act.norm().item(),
                        "cav_projections": {},
                        "cav_cosine_similarities": {}
                    }

                    for cav_behavior, cav in cavs.items():
                        projection = compute_projection(diff_act, cav)
                        data_point["difference"]["cav_projections"][cav_behavior] = projection

                        # Cosine similarity
                        diff_act_normalized = diff_act / (diff_act.norm() + 1e-10)
                        cav_normalized = cav / (cav.norm() + 1e-10)
                        cosine_sim = torch.dot(diff_act_normalized, cav_normalized).item()
                        data_point["difference"]["cav_cosine_similarities"][cav_behavior] = cosine_sim

            behavior_results["data_points"].append(data_point)

        # Save results for this behavior
        output_file = os.path.join(output_dir, f"{source_behavior}_cav_projections_layer{layer}.json")
        with open(output_file, 'w') as f:
            json.dump(behavior_results, f, indent=2)

        print(f"Saved projections for {source_behavior} to {output_file}")
        all_results[source_behavior] = behavior_results

    # Also save a combined file with summary statistics
    summary = {
        "metadata": {
            "layer": layer,
            "model": model_name,
            "behaviors": TARGET_BEHAVIORS,
            "timestamp": datetime.now().isoformat()
        },
        "statistics": {}
    }

    for behavior in all_results:
        behavior_stats = {
            "num_datapoints": len(all_results[behavior]["data_points"]),
            "mean_projections_positive": {},
            "mean_projections_negative": {},
            "mean_projections_difference": {},
            "std_projections_positive": {},
            "std_projections_negative": {},
            "std_projections_difference": {}
        }

        # Collect all projections for statistics
        for cav_behavior in TARGET_BEHAVIORS:
            pos_projs = []
            neg_projs = []
            diff_projs = []

            for dp in all_results[behavior]["data_points"]:
                if "positive_case" in dp and "cav_projections" in dp["positive_case"]:
                    if cav_behavior in dp["positive_case"]["cav_projections"]:
                        pos_projs.append(dp["positive_case"]["cav_projections"][cav_behavior])

                if "negative_case" in dp and "cav_projections" in dp["negative_case"]:
                    if cav_behavior in dp["negative_case"]["cav_projections"]:
                        neg_projs.append(dp["negative_case"]["cav_projections"][cav_behavior])

                if "difference" in dp and "cav_projections" in dp["difference"]:
                    if cav_behavior in dp["difference"]["cav_projections"]:
                        diff_projs.append(dp["difference"]["cav_projections"][cav_behavior])

            if pos_projs:
                behavior_stats["mean_projections_positive"][cav_behavior] = float(np.mean(pos_projs))
                behavior_stats["std_projections_positive"][cav_behavior] = float(np.std(pos_projs))

            if neg_projs:
                behavior_stats["mean_projections_negative"][cav_behavior] = float(np.mean(neg_projs))
                behavior_stats["std_projections_negative"][cav_behavior] = float(np.std(neg_projs))

            if diff_projs:
                behavior_stats["mean_projections_difference"][cav_behavior] = float(np.mean(diff_projs))
                behavior_stats["std_projections_difference"][cav_behavior] = float(np.std(diff_projs))

        summary["statistics"][behavior] = behavior_stats

    # Save summary
    summary_file = os.path.join(output_dir, f"projection_summary_layer{layer}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_file}")

    # Print a quick analysis
    print("\n" + "="*60)
    print("PROJECTION ANALYSIS SUMMARY")
    print("="*60)

    for behavior in summary["statistics"]:
        stats = summary["statistics"][behavior]
        print(f"\n{behavior.upper()}:")
        print(f"  Number of datapoints: {stats['num_datapoints']}")

        # Find strongest mean projection in difference (should be self)
        if stats["mean_projections_difference"]:
            sorted_projs = sorted(stats["mean_projections_difference"].items(),
                                key=lambda x: x[1], reverse=True)
            print(f"  Top 3 CAV projections (difference):")
            for i, (cav_behavior, proj) in enumerate(sorted_projs[:3]):
                print(f"    {i+1}. {cav_behavior}: {proj:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Compute CAV projections for all training data')
    parser.add_argument('--layers', nargs='+', type=int, default=[15],
                        help='Layer numbers to analyze (e.g., --layers 10 15 20 25)')
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf',
                        help='Model name')
    parser.add_argument('--output_dir', type=str, default='./cav_projections',
                        help='Output directory for projection data')

    args = parser.parse_args()

    print(f"Computing CAV projections for layers: {args.layers}")

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)
        compute_all_projections(layer, args.model, args.output_dir)

if __name__ == "__main__":
    main()