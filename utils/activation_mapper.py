#!/usr/bin/env python3
"""
Utility functions for working with activation-data mappings.
"""

import torch
import json
import os
from typing import Dict, List, Optional, Tuple


class ActivationDataMapper:
    """
    A class to manage the mapping between activations and training data.
    """

    def __init__(self, behavior: str, layer: int, model_name: str = "Llama-2-7b-chat-hf"):
        self.behavior = behavior
        self.layer = layer
        self.model_name = model_name
        self.training_data = None
        self.pos_activations = None
        self.neg_activations = None
        self._load_data()

    def _load_data(self):
        """Load training data and activations."""
        # Try different data paths
        data_paths = [
            f"./CAA_data_v0/datasets/generate/{self.behavior}/generate_dataset.json",
            f"./CAA/datasets/generate/{self.behavior}/generate_dataset.json"
        ]

        for path in data_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.training_data = json.load(f)
                print(f"Loaded {len(self.training_data)} training examples from {path}")
                break

        if self.training_data is None:
            print(f"Warning: Could not find training data for {self.behavior}")
            return

        # Load activations
        pos_path = f"./CAA/activations/{self.behavior}/activations_pos_{self.layer}_{self.model_name}.pt"
        neg_path = f"./CAA/activations/{self.behavior}/activations_neg_{self.layer}_{self.model_name}.pt"

        if os.path.exists(pos_path):
            self.pos_activations = torch.load(pos_path, map_location='cpu')
            print(f"Loaded positive activations: {self.pos_activations.shape}")

        if os.path.exists(neg_path):
            self.neg_activations = torch.load(neg_path, map_location='cpu')
            print(f"Loaded negative activations: {self.neg_activations.shape}")

    def get_example(self, idx: int) -> Dict:
        """
        Get a specific training example with its activations.

        Args:
            idx: Index of the example (0-based)

        Returns:
            Dictionary containing question, answers, and activations
        """
        if self.training_data is None or idx >= len(self.training_data):
            return None

        example = self.training_data[idx]
        result = {
            'index': idx,
            'question': example['question'],
            'answer_matching': example['answer_matching_behavior'],
            'answer_not_matching': example['answer_not_matching_behavior']
        }

        if self.pos_activations is not None:
            result['pos_activation'] = self.pos_activations[idx]

        if self.neg_activations is not None:
            result['neg_activation'] = self.neg_activations[idx]

        if self.pos_activations is not None and self.neg_activations is not None:
            result['difference'] = self.pos_activations[idx] - self.neg_activations[idx]
            result['difference_norm'] = result['difference'].norm().item()

        return result

    def find_examples_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Find all examples containing a keyword in the question.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching examples with their indices
        """
        matches = []
        if self.training_data is None:
            return matches

        for idx, example in enumerate(self.training_data):
            if keyword.lower() in example['question'].lower():
                matches.append({
                    'index': idx,
                    'question': example['question'],
                    'has_activations': self.pos_activations is not None
                })

        return matches

    def get_activation_for_text(self, text_snippet: str, use_positive: bool = True) -> Optional[torch.Tensor]:
        """
        Get the activation vector for a specific text snippet.

        Args:
            text_snippet: Part of the question or answer to search for
            use_positive: If True, return positive activation; if False, return negative

        Returns:
            The activation tensor if found, None otherwise
        """
        if self.training_data is None:
            return None

        activations = self.pos_activations if use_positive else self.neg_activations
        if activations is None:
            return None

        for idx, example in enumerate(self.training_data):
            if text_snippet in example['question']:
                return activations[idx]
            if use_positive and text_snippet in example['answer_matching_behavior']:
                return activations[idx]
            if not use_positive and text_snippet in example['answer_not_matching_behavior']:
                return activations[idx]

        return None

    def compute_mean_difference(self) -> Optional[torch.Tensor]:
        """
        Compute the mean difference vector (should match the steering vector).

        Returns:
            Mean difference vector or None if activations not available
        """
        if self.pos_activations is None or self.neg_activations is None:
            return None

        differences = self.pos_activations - self.neg_activations
        return differences.mean(dim=0)

    def get_strongest_examples(self, n: int = 5) -> List[Dict]:
        """
        Get the n examples with the strongest activation differences.

        Args:
            n: Number of examples to return

        Returns:
            List of examples sorted by difference norm (descending)
        """
        if self.pos_activations is None or self.neg_activations is None:
            return []

        differences = self.pos_activations - self.neg_activations
        norms = differences.norm(dim=1)
        top_indices = norms.argsort(descending=True)[:n]

        results = []
        for idx in top_indices:
            idx = idx.item()
            example = self.get_example(idx)
            example['rank'] = len(results) + 1
            results.append(example)

        return results

    def verify_steering_vector(self) -> bool:
        """
        Verify that the computed mean matches the saved steering vector.

        Returns:
            True if they match, False otherwise
        """
        computed_mean = self.compute_mean_difference()
        if computed_mean is None:
            print("Cannot compute mean difference - activations not loaded")
            return False

        vector_path = f"./CAA/vectors/{self.behavior}/vec_layer_{self.layer}_{self.model_name}.pt"
        if not os.path.exists(vector_path):
            print(f"Steering vector not found at {vector_path}")
            return False

        steering_vector = torch.load(vector_path, map_location='cpu')

        matches = torch.allclose(computed_mean, steering_vector, rtol=1e-5)
        if matches:
            print(f"✓ Computed mean matches steering vector (norm: {steering_vector.norm().item():.4f})")
        else:
            print(f"✗ Mismatch between computed mean and steering vector")
            print(f"  Computed norm: {computed_mean.norm().item():.4f}")
            print(f"  Steering norm: {steering_vector.norm().item():.4f}")

        return matches


# Example usage functions
def demo_usage():
    """Demonstrate how to use the ActivationDataMapper."""

    print("="*60)
    print("ACTIVATION DATA MAPPER DEMO")
    print("="*60)

    # Initialize mapper
    mapper = ActivationDataMapper("pride-humility", layer=15)

    # Get a specific example
    print("\n1. Getting example #0:")
    example = mapper.get_example(0)
    if example:
        print(f"   Question: {example['question'][:100]}...")
        if 'difference_norm' in example:
            print(f"   Activation difference norm: {example['difference_norm']:.4f}")

    # Find examples by keyword
    print("\n2. Finding examples with 'team':")
    matches = mapper.find_examples_by_keyword("team")
    for match in matches[:3]:
        print(f"   Example {match['index']}: {match['question'][:80]}...")

    # Get strongest examples
    print("\n3. Top 3 examples with strongest differences:")
    strong = mapper.get_strongest_examples(3)
    for ex in strong:
        print(f"   Rank {ex['rank']}: Index {ex['index']}, Norm: {ex.get('difference_norm', 'N/A'):.4f}")

    # Verify steering vector
    print("\n4. Verifying steering vector:")
    mapper.verify_steering_vector()


if __name__ == "__main__":
    # Run demo
    demo_usage()

    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("""
# Initialize mapper for your behavior and layer
from utils.activation_mapper import ActivationDataMapper

mapper = ActivationDataMapper("pride-humility", layer=15)

# Get activation for the 5th training example
example = mapper.get_example(4)
pos_activation = example['pos_activation']  # Shape: [4096]

# Find all examples about "promotion"
promotion_examples = mapper.find_examples_by_keyword("promotion")

# Get the examples that create the strongest steering signal
strongest = mapper.get_strongest_examples(10)

# Compute the mean steering vector from activations
steering_vector = mapper.compute_mean_difference()

# Verify your steering vector matches the saved one
is_valid = mapper.verify_steering_vector()
""")