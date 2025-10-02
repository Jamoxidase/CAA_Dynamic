#!/usr/bin/env python3
"""
Run AB and open-ended tests using weighted CAV vectors for steering.
Each test question is steered with its own custom weighted vector.

Usage:
python prompting_with_weighted_steering.py --behaviors pride-humility --layers 15 --multipliers -1 0 1 --model_size 7b
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from typing import List, Dict, Optional
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

from llama_wrapper import LlamaWrapper
from lfm2_wrapper import LFM2Wrapper
from behaviors import get_ab_test_data, get_open_ended_test_data
from utils.helpers import get_a_b_probs, get_model_path
from utils.llama_tokenize import E_INST, EOT, IM_END

load_dotenv()

TARGET_BEHAVIORS = [
    "envy-kindness",
    "gluttony-temperance",
    "greed-charity",
    "lust-chastity",
    "pride-humility",
    "sloth-diligence",
    "wrath-patience"
]

def load_weighted_vectors(
    behavior: str,
    layer: int,
    model_size: str,
    use_base_model: bool,
    vectors_dir: str = "./vectors"
) -> Dict:
    """Load weighted vectors for a behavior."""
    base_str = "base_" if use_base_model else ""
    filepath = os.path.join(
        vectors_dir,
        f"{behavior}_weighted_vectors_layer{layer}_{base_str}{model_size}.pt"
    )
    if os.path.exists(filepath):
        return torch.load(filepath, map_location='cpu')
    else:
        raise FileNotFoundError(f"Weighted vectors not found: {filepath}")

def process_item_ab_weighted(
    item: Dict[str, str],
    weighted_vector: torch.Tensor,
    model: LlamaWrapper,
    layer: int,
    multiplier: float,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    """Process AB test with weighted steering vector."""
    question = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]

    # Apply weighted steering vector
    model.reset_all()
    model.set_add_activations(layer, multiplier * weighted_vector)

    # Get probabilities
    model_output = model.get_logits_from_text(
        user_input=question, model_output="("
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)

    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def process_item_open_ended_weighted(
    item: Dict[str, str],
    weighted_vector: torch.Tensor,
    model: LlamaWrapper,
    layer: int,
    multiplier: float,
) -> Dict[str, str]:
    """Process open-ended test with weighted steering vector."""
    question = item["question"]

    # Apply weighted steering vector
    model.reset_all()
    model.set_add_activations(layer, multiplier * weighted_vector)

    # Generate response
    model_output = model.generate_text(
        user_input=question, system_prompt=None, max_new_tokens=100
    )

    # Parse output based on model type
    if hasattr(model, 'is_llama3') and model.is_llama3:
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_marker in model_output:
            text_after_marker = model_output.split(assistant_marker)[-1]
            if EOT in text_after_marker:
                parsed_output = text_after_marker.split(EOT)[0].strip()
            else:
                parsed_output = text_after_marker.strip()
        else:
            parsed_output = model_output.split(EOT)[-1].strip()
    elif hasattr(model, 'size') and model.size == "1.2b":  # LFM2
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in model_output:
            text_after_marker = model_output.split(assistant_marker)[-1]
            if IM_END in text_after_marker:
                parsed_output = text_after_marker.split(IM_END)[0].strip()
            else:
                parsed_output = text_after_marker.strip()
        else:
            parsed_output = ""
    else:
        # Llama 2
        parsed_output = model_output.split(E_INST)[-1].strip()

    return {
        "question": question,
        "model_output": parsed_output,
        "full_output": model_output
    }

def run_weighted_steering(
    behavior: str,
    layer: int,
    multipliers: List[float],
    model_size: str,
    use_base_model: bool,
    vectors_dir: str = "./vectors",
    output_dir: str = "./results"
):
    """Run AB and open-ended tests with weighted steering for a behavior."""

    # Load weighted vectors
    print(f"\nLoading weighted vectors for {behavior}, layer {layer}...")
    weighted_data = load_weighted_vectors(behavior, layer, model_size, use_base_model, vectors_dir)

    weighted_vectors = weighted_data["weighted_vectors"]
    questions_list = weighted_data["questions"]
    weights_list = weighted_data["weights"]

    if weighted_vectors is None or len(weighted_vectors) == 0:
        print(f"No weighted vectors found for {behavior}")
        return

    print(f"Loaded {len(weighted_vectors)} weighted vectors")

    # Load model
    model_name_path = get_model_path(model_size, is_base=use_base_model)
    hf_token = os.getenv("HF_TOKEN")

    print(f"Loading model: {model_name_path}")
    if model_size == "1.2b":
        model = LFM2Wrapper(hf_token=hf_token, size=model_size)
    else:
        model = LlamaWrapper(hf_token=hf_token, size=model_size, use_chat=not use_base_model)

    # Get token IDs for A and B
    a_token_id = model.tokenizer.encode("A", add_special_tokens=False)[0]
    b_token_id = model.tokenizer.encode("B", add_special_tokens=False)[0]

    # Load test data
    ab_test_data = get_ab_test_data(behavior)
    open_ended_test_data = get_open_ended_test_data(behavior)

    # Ensure same number of questions
    if len(ab_test_data) != len(weighted_vectors):
        print(f"Warning: Mismatch in data sizes - AB test: {len(ab_test_data)}, weighted vectors: {len(weighted_vectors)}")
        min_len = min(len(ab_test_data), len(weighted_vectors))
        ab_test_data = ab_test_data[:min_len]
        weighted_vectors = weighted_vectors[:min_len]

    # Process each multiplier
    for multiplier in multipliers:
        print(f"\n{'='*60}")
        print(f"Processing multiplier {multiplier}")
        print('='*60)

        # AB Test Results
        ab_results = []
        for idx, (ab_item, weighted_vec) in enumerate(tqdm(
            zip(ab_test_data, weighted_vectors),
            total=len(ab_test_data),
            desc=f"AB test (mult={multiplier})"
        )):
            result = process_item_ab_weighted(
                ab_item, weighted_vec, model, layer, multiplier, a_token_id, b_token_id
            )
            # Add metadata
            result["index"] = idx
            result["weighted_vector_weights"] = weights_list[idx]
            ab_results.append(result)

        # Save AB results
        base_str = "base_" if use_base_model else ""
        ab_output_file = os.path.join(
            output_dir, "ab",
            f"{behavior}_weighted_ab_layer{layer}_mult{multiplier}_{base_str}{model_size}.json"
        )
        os.makedirs(os.path.dirname(ab_output_file), exist_ok=True)
        with open(ab_output_file, 'w') as f:
            json.dump(ab_results, f, indent=2)
        print(f"Saved AB results to {ab_output_file}")

        # Open-Ended Test Results
        open_ended_results = []
        for idx, (oe_item, weighted_vec) in enumerate(tqdm(
            zip(open_ended_test_data, weighted_vectors),
            total=len(open_ended_test_data),
            desc=f"Open-ended (mult={multiplier})"
        )):
            result = process_item_open_ended_weighted(
                oe_item, weighted_vec, model, layer, multiplier
            )
            # Add metadata
            result["index"] = idx
            result["weighted_vector_weights"] = weights_list[idx]
            result["corresponding_ab_question"] = ab_test_data[idx]["question"]
            open_ended_results.append(result)

        # Save open-ended results
        oe_output_file = os.path.join(
            output_dir, "open_ended",
            f"{behavior}_weighted_open_ended_layer{layer}_mult{multiplier}_{base_str}{model_size}.json"
        )
        os.makedirs(os.path.dirname(oe_output_file), exist_ok=True)
        with open(oe_output_file, 'w') as f:
            json.dump(open_ended_results, f, indent=2)
        print(f"Saved open-ended results to {oe_output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run weighted CAV steering tests')
    parser.add_argument(
        '--behaviors',
        type=str,
        nargs='+',
        default=TARGET_BEHAVIORS,
        help='Behaviors to test (defaults to all seven deadly sins)'
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        required=True,
        help='Layer numbers to test'
    )
    parser.add_argument(
        '--multipliers',
        nargs='+',
        type=float,
        required=True,
        help='Steering multipliers to test'
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
        '--vectors_dir',
        type=str,
        default='./vectors',
        help='Directory containing weighted vectors'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for test results'
    )

    args = parser.parse_args()

    print(f"Running weighted steering tests for behaviors: {args.behaviors}")
    print(f"Layers: {args.layers}")
    print(f"Multipliers: {args.multipliers}")

    for behavior in args.behaviors:
        for layer in args.layers:
            print(f"\n{'#'*60}")
            print(f"# Processing {behavior}, layer {layer}")
            print(f"{'#'*60}")

            run_weighted_steering(
                behavior,
                layer,
                args.multipliers,
                args.model_size,
                args.use_base_model,
                args.vectors_dir,
                args.output_dir
            )

    print("\n" + "="*60)
    print("ALL WEIGHTED STEERING TESTS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
