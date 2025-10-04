#!/usr/bin/env python3
"""
Compute projections of AB data activations onto all CAVs.
This analyzes each question to determine how much each behavior's signal is present.

Usage:
python compute_test_projections.py --behaviors pride-humility envy-kindness --layers 15 --model_size 7b --dataset train
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
from dotenv import load_dotenv

from llama_wrapper import LlamaWrapper
from lfm2_wrapper import LFM2Wrapper
from behaviors import get_ab_data_path, ALL_BEHAVIORS
from utils.helpers import get_model_path

load_dotenv()

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

def load_normalized_cav(behavior: str, layer: int, model_name: str) -> torch.Tensor:
    """Load a normalized CAV for a behavior."""
    cav_path = f"../normalized_vectors/{behavior}/vec_layer_{layer}_{model_name}.pt"
    if os.path.exists(cav_path):
        return torch.load(cav_path, map_location='cpu').to(torch.float64)
    else:
        print(f"Warning: CAV not found at {cav_path}")
        return None

def compute_projection(activation: torch.Tensor, cav: torch.Tensor) -> float:
    """
    Compute the projection of an activation onto a CAV.
    This is the dot product, which measures how much the CAV is present in the activation.
    """
    return torch.dot(activation, cav).item()

def get_test_activations(
    model: LlamaWrapper,
    question: str,
    answer_a: str,
    answer_b: str,
    layer: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get activations for answer A and answer B for a test question.
    Returns: (activation_a, activation_b)
    """
    # Construct full prompts
    prompt_a = f"{question}\n{answer_a}"
    prompt_b = f"{question}\n{answer_b}"

    # Get activations at position -2 (before final token)
    model.reset_all()
    model.get_logits_from_text(user_input=prompt_a, model_output="")
    activation_a = model.get_last_activations(layer)

    # Handle different model types (LFM2 vs Llama)
    if isinstance(model, LFM2Wrapper):
        # LFM2 activations are [batch, seq, hidden]
        activation_a = activation_a[0, -2, :].to(torch.float64)
    else:
        # Llama activations are [seq, hidden]
        activation_a = activation_a[-2, :].to(torch.float64)

    model.reset_all()
    model.get_logits_from_text(user_input=prompt_b, model_output="")
    activation_b = model.get_last_activations(layer)

    if isinstance(model, LFM2Wrapper):
        activation_b = activation_b[0, -2, :].to(torch.float64)
    else:
        activation_b = activation_b[-2, :].to(torch.float64)

    return activation_a, activation_b

def compute_all_projections(
    behaviors: List[str],
    layer: int,
    model_size: str,
    use_base_model: bool,
    dataset: str,
    output_dir: str = "./projections"
):
    """
    Compute projections of all AB activations onto all CAVs.
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get model path
    model_name_path = get_model_path(model_size, is_base=use_base_model)
    model_name = os.path.basename(model_name_path)

    # Load model
    print(f"Loading model: {model_name_path}")
    hf_token = os.getenv("HF_TOKEN")

    if model_size == "1.2b":
        model = LFM2Wrapper(hf_token=hf_token, size=model_size)
    else:
        model = LlamaWrapper(hf_token=hf_token, size=model_size, use_chat=not use_base_model)

    # First, load all CAVs
    cavs = {}
    for behavior in TARGET_BEHAVIORS:
        cav = load_normalized_cav(behavior, layer, model_name)
        if cav is not None:
            cavs[behavior] = cav.to(model.device)
            print(f"Loaded CAV for {behavior}: shape {cav.shape}, norm {cav.norm().item():.6f}")

    if not cavs:
        print("Error: No CAVs could be loaded!")
        return

    # Process each behavior's AB data
    for source_behavior in tqdm(behaviors, desc="Processing behaviors"):
        print(f"\nProcessing {dataset} data for {source_behavior}...")

        # Load AB data
        try:
            with open(get_ab_data_path(source_behavior, test=(dataset == "test")), 'r') as f:
                ab_data = json.load(f)
        except Exception as e:
            print(f"Skipping {source_behavior} - could not load {dataset} data: {e}")
            continue

        behavior_results = {
            "behavior": source_behavior,
            "layer": layer,
            "model": model_name,
            "model_size": model_size,
            "use_base_model": use_base_model,
            "dataset": dataset,
            "normalized_cavs_used": list(cavs.keys()),
            "data_points": []
        }

        # Process each example
        for idx, example in enumerate(tqdm(ab_data, desc=f"Processing {source_behavior}")):
            question = example["question"]
            answer_matching = example["answer_matching_behavior"]
            answer_not_matching = example["answer_not_matching_behavior"]

            # Get activations for both answers
            try:
                activation_a, activation_b = get_test_activations(
                    model, question, answer_matching, answer_not_matching, layer
                )
            except Exception as e:
                print(f"\nError getting activations for question {idx}: {e}")
                continue

            # Compute difference (matching - not_matching)
            diff_activation = activation_a - activation_b

            data_point = {
                "index": idx,
                "question": question,
                "answer_matching_behavior": answer_matching,
                "answer_not_matching_behavior": answer_not_matching,
                "matching_case": {
                    "activation_norm": activation_a.norm().item(),
                    "cav_projections": {},
                    "cav_cosine_similarities": {}
                },
                "not_matching_case": {
                    "activation_norm": activation_b.norm().item(),
                    "cav_projections": {},
                    "cav_cosine_similarities": {}
                },
                "difference": {
                    "activation_norm": diff_activation.norm().item(),
                    "cav_projections": {},
                    "cav_cosine_similarities": {}
                }
            }

            # Compute projections onto all CAVs for matching answer
            for cav_behavior, cav in cavs.items():
                projection = compute_projection(activation_a, cav)
                data_point["matching_case"]["cav_projections"][cav_behavior] = projection

                # Cosine similarity
                act_norm = activation_a / (activation_a.norm() + 1e-10)
                cav_norm = cav / (cav.norm() + 1e-10)
                cosine_sim = torch.dot(act_norm, cav_norm).item()
                data_point["matching_case"]["cav_cosine_similarities"][cav_behavior] = cosine_sim

            # Compute projections for not matching answer
            for cav_behavior, cav in cavs.items():
                projection = compute_projection(activation_b, cav)
                data_point["not_matching_case"]["cav_projections"][cav_behavior] = projection

                act_norm = activation_b / (activation_b.norm() + 1e-10)
                cav_norm = cav / (cav.norm() + 1e-10)
                cosine_sim = torch.dot(act_norm, cav_norm).item()
                data_point["not_matching_case"]["cav_cosine_similarities"][cav_behavior] = cosine_sim

            # Compute projections for difference
            for cav_behavior, cav in cavs.items():
                projection = compute_projection(diff_activation, cav)
                data_point["difference"]["cav_projections"][cav_behavior] = projection

                diff_norm = diff_activation / (diff_activation.norm() + 1e-10)
                cav_norm = cav / (cav.norm() + 1e-10)
                cosine_sim = torch.dot(diff_norm, cav_norm).item()
                data_point["difference"]["cav_cosine_similarities"][cav_behavior] = cosine_sim

            behavior_results["data_points"].append(data_point)

        # Save results for this behavior
        base_str = "base_" if use_base_model else ""
        output_file = os.path.join(
            output_dir,
            f"{source_behavior}_{dataset}_projections_layer{layer}_{base_str}{model_size}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(behavior_results, f, indent=2)

        print(f"Saved projections for {source_behavior} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compute CAV projections for AB data')
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
        help='Layer numbers to analyze (e.g., --layers 10 15 20 25)'
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
        '--dataset',
        type=str,
        choices=["train", "test"],
        default="train",
        help='Dataset to use: train (generate) or test'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./projections',
        help='Output directory for projection data'
    )

    args = parser.parse_args()

    print(f"Computing projections for behaviors: {args.behaviors}")
    print(f"Dataset: {args.dataset}")
    print(f"Layers: {args.layers}")
    print(f"Model: {args.model_size} (base={args.use_base_model})")

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)
        compute_all_projections(
            args.behaviors,
            layer,
            args.model_size,
            args.use_base_model,
            args.dataset,
            args.output_dir
        )

if __name__ == "__main__":
    main()
