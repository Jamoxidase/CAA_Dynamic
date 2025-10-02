#!/usr/bin/env python3
"""
Create weighted CAV vectors for each test data point.
Each vector is a weighted combination of all 7 CAVs based on their projection strengths.

Usage:
python create_weighted_test_vectors.py --behaviors pride-humility envy-kindness --layers 15 --model_size 7b
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
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

def load_normalized_cavs(layer: int, model_size: str, use_base_model: bool) -> Dict[str, torch.Tensor]:
    """Load all normalized CAVs for a layer."""
    from utils.helpers import get_model_path

    model_name_path = get_model_path(model_size, is_base=use_base_model)
    model_name = os.path.basename(model_name_path)

    cavs = {}
    for behavior in TARGET_BEHAVIORS:
        cav_path = f"../normalized_vectors/{behavior}/vec_layer_{layer}_{model_name}.pt"
        if os.path.exists(cav_path):
            cavs[behavior] = torch.load(cav_path, map_location='cpu').to(torch.float64)
        else:
            print(f"Warning: CAV not found for {behavior} at layer {layer}")
    return cavs

def load_projection_data(
    behavior: str,
    layer: int,
    model_size: str,
    use_base_model: bool,
    projection_dir: str = "./projections"
) -> Dict:
    """Load projection data for a specific behavior and layer."""
    base_str = "base_" if use_base_model else ""
    filepath = os.path.join(
        projection_dir,
        f"{behavior}_test_projections_layer{layer}_{base_str}{model_size}.json"
    )
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Could not find {filepath}")
        return None

def create_weighted_cav_vector(
    projections: Dict[str, float],
    cavs: Dict[str, torch.Tensor],
    normalize_weights: bool = True,
    normalize_to_mean_norm: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Create a single weighted CAV vector by combining all CAVs weighted by their projections.

    Args:
        projections: Dictionary mapping behavior names to projection values
        cavs: Dictionary mapping behavior names to CAV vectors
        normalize_weights: Whether to normalize the projection weights to sum to 1
        normalize_to_mean_norm: Whether to normalize result to mean CAV norm (matching CAV standards)

    Returns:
        weighted_vector: The weighted combination vector
        weights: Dictionary of weights used
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
        return weighted_vector, {}

    weights = np.array(weights, dtype=np.float64)

    # Normalize weights to sum to 1 (preserving sign/proportionality)
    if normalize_weights and np.sum(np.abs(weights)) > 0:
        weights = weights / np.sum(np.abs(weights))

    # Create weighted combination
    for i, behavior in enumerate(behaviors_used):
        weighted_vector += weights[i] * cavs[behavior]

    # Normalize to mean CAV norm (to match CAV normalization standards)
    if normalize_to_mean_norm:
        cav_norms = [cav.norm().item() for cav in cavs.values()]
        mean_norm = np.mean(cav_norms)
        current_norm = weighted_vector.norm().item()
        if current_norm > 0:
            weighted_vector = weighted_vector * (mean_norm / current_norm)

    return weighted_vector, dict(zip(behaviors_used, weights))

def process_all_data(
    behaviors: List[str],
    layer: int,
    model_size: str,
    use_base_model: bool,
    projection_dir: str = "./projections",
    output_dir: str = "./vectors"
):
    """
    Process all behaviors and create weighted CAV vectors for each test data point.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all CAVs
    print(f"Loading CAVs for layer {layer}...")
    cavs = load_normalized_cavs(layer, model_size, use_base_model)

    if not cavs:
        print("Error: No CAVs could be loaded!")
        return

    cav_dim = next(iter(cavs.values())).shape[0]
    print(f"Loaded {len(cavs)} CAVs, dimension: {cav_dim}")

    for source_behavior in tqdm(behaviors, desc="Processing behaviors"):
        # Load projection data
        proj_data = load_projection_data(
            source_behavior, layer, model_size, use_base_model, projection_dir
        )
        if proj_data is None:
            continue

        behavior_results = {
            "behavior": source_behavior,
            "layer": layer,
            "model_size": model_size,
            "use_base_model": use_base_model,
            "cav_dimension": cav_dim,
            "data_points": []
        }

        # Store all weighted vectors for saving
        all_weighted_vectors = []
        all_weights = []
        all_questions = []

        for dp in proj_data["data_points"]:
            result = {
                "index": dp["index"],
                "question": dp["question"],
                "answer_matching_behavior": dp["answer_matching_behavior"],
                "answer_not_matching_behavior": dp["answer_not_matching_behavior"]
            }

            # Create weighted vector using difference projections
            if "difference" in dp and "cav_projections" in dp["difference"]:
                weighted_vec, weights = create_weighted_cav_vector(
                    dp["difference"]["cav_projections"],
                    cavs,
                    normalize_weights=True,
                    normalize_to_mean_norm=True
                )

                result["weighted_vector_norm"] = weighted_vec.norm().item()
                result["weights"] = weights
                result["raw_projections"] = dp["difference"]["cav_projections"]

                # Store for tensor file
                all_weighted_vectors.append(weighted_vec)
                all_weights.append(weights)
                all_questions.append(dp["question"])

            behavior_results["data_points"].append(result)

        # Save individual behavior JSON file
        base_str = "base_" if use_base_model else ""
        output_file = os.path.join(
            output_dir,
            f"{source_behavior}_weighted_vectors_layer{layer}_{base_str}{model_size}.json"
        )

        save_data = {
            "behavior": behavior_results["behavior"],
            "layer": behavior_results["layer"],
            "model_size": behavior_results["model_size"],
            "use_base_model": behavior_results["use_base_model"],
            "cav_dimension": behavior_results["cav_dimension"],
            "num_datapoints": len(behavior_results["data_points"]),
            "data_points": behavior_results["data_points"]
        }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved weighted CAV vectors for {source_behavior}")

        # Save tensor file with all weighted vectors for easy loading during steering
        tensor_file = os.path.join(
            output_dir,
            f"{source_behavior}_weighted_vectors_layer{layer}_{base_str}{model_size}.pt"
        )
        torch.save({
            "behavior": source_behavior,
            "layer": layer,
            "model_size": model_size,
            "use_base_model": use_base_model,
            "weighted_vectors": torch.stack(all_weighted_vectors) if all_weighted_vectors else None,
            "weights": all_weights,
            "questions": all_questions,
            "cavs": cavs,
            "cav_behaviors": list(cavs.keys())
        }, tensor_file)

        print(f"Saved tensor file to {tensor_file}")

def main():
    parser = argparse.ArgumentParser(description='Create weighted CAV vectors from test projections')
    parser.add_argument(
        '--behaviors',
        type=str,
        nargs='+',
        default=TARGET_BEHAVIORS,
        help='Behaviors to process (defaults to all seven deadly sins)'
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        default=[15],
        help='Layer numbers to process (e.g., --layers 10 15 20 25)'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        choices=["7b", "8b", "13b", "1.2b"],
        default="7b",
        help='Model size'
    )
    parser.add_argument(
        '--use_base_model',
        action='store_true',
        default=False,
        help='Use base model instead of chat/instruct model'
    )
    parser.add_argument(
        '--projection_dir',
        type=str,
        default='./projections',
        help='Directory containing projection data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./vectors',
        help='Output directory for weighted vectors'
    )

    args = parser.parse_args()

    print(f"Creating weighted CAV vectors for behaviors: {args.behaviors}")
    print(f"Layers: {args.layers}")

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)

        process_all_data(
            args.behaviors,
            layer,
            args.model_size,
            args.use_base_model,
            args.projection_dir,
            args.output_dir
        )

if __name__ == "__main__":
    main()
