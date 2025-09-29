#!/usr/bin/env python3
"""
Data loader for training an adapter model using weighted CAV vectors.
This module loads and prepares data for training where:
- Input: Text embeddings from your embedding model (to be added)
- Output: Weighted CAV vectors (already computed)
"""

import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import random
from tqdm import tqdm


class WeightedCAVDataset(Dataset):
    """
    Dataset class for loading weighted CAV vectors paired with text data.

    The dataset provides:
    - Text (question/answer pairs) for generating embeddings
    - Pre-computed weighted CAV vectors as training targets
    - Support for positive, negative, or difference vectors
    """

    def __init__(
        self,
        data_dir: str = "./weighted_cav_vectors",
        behaviors: Optional[List[str]] = None,
        layer: int = 15,
        vector_type: str = "difference",  # "positive", "negative", or "difference"
        embedding_fn: Optional[Callable] = None,
        max_samples_per_behavior: Optional[int] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing weighted CAV vector JSON files
            behaviors: List of behaviors to include (None for all)
            layer: Layer number to load data for
            vector_type: Which vectors to use as targets
            embedding_fn: Function to generate embeddings from text (you'll provide this)
            max_samples_per_behavior: Limit samples per behavior for debugging
        """
        self.data_dir = Path(data_dir)
        self.layer = layer
        self.vector_type = vector_type
        self.embedding_fn = embedding_fn

        # Default behaviors if not specified
        if behaviors is None:
            behaviors = [
                "envy-kindness", "gluttony-temperance", "greed-charity",
                "lust-chastity", "pride-humility", "sloth-diligence",
                "wrath-patience"
            ]
        self.behaviors = behaviors

        # Load all data
        self.data_points = []
        self._load_data(max_samples_per_behavior)

        print(f"Loaded {len(self.data_points)} data points from {len(behaviors)} behaviors")

    def _load_data(self, max_samples: Optional[int] = None):
        """Load data from JSON files."""

        for behavior in self.behaviors:
            file_path = self.data_dir / f"{behavior}_weighted_cavs_layer{self.layer}.json"

            if not file_path.exists():
                print(f"Warning: File not found {file_path}")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract data points
            points = data.get('data_points', [])
            if max_samples:
                points = points[:max_samples]

            for dp in points:
                # Create data point with all necessary information
                point = {
                    'behavior': behavior,
                    'index': dp['index'],
                    'question': dp['question'],
                    'answer_positive': dp['answer_matching_behavior'],
                    'answer_negative': dp['answer_not_matching_behavior'],
                }

                # Add the appropriate weighted vector based on vector_type
                vector_key = f"{self.vector_type}_weighted_vector"
                if vector_key in dp:
                    point['target_vector'] = torch.tensor(dp[vector_key], dtype=torch.float32)
                    point['vector_norm'] = dp.get(f"{self.vector_type}_weighted_vector_norm", 0)
                    point['weights'] = dp.get(f"{self.vector_type}_weights", {})
                    point['raw_projections'] = dp.get(f"{self.vector_type}_raw_projections", {})

                    self.data_points.append(point)
                else:
                    print(f"Warning: {vector_key} not found in data point")

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        """
        Get a single data point.

        Returns:
            Dict containing:
            - text: Combined question and answer text
            - embedding: Text embedding (if embedding_fn provided)
            - target_vector: Weighted CAV vector
            - metadata: Additional information
        """
        point = self.data_points[idx]

        # Prepare text based on vector type
        if self.vector_type == "positive":
            text = f"{point['question']} {point['answer_positive']}"
        elif self.vector_type == "negative":
            text = f"{point['question']} {point['answer_negative']}"
        else:  # difference
            # For difference vectors, you might want both answers
            text = f"{point['question']} [POS] {point['answer_positive']} [NEG] {point['answer_negative']}"

        result = {
            'text': text,
            'target_vector': point['target_vector'],
            'behavior': point['behavior'],
            'vector_norm': point['vector_norm'],
            'weights': point['weights']
        }

        # Generate embedding if function provided
        if self.embedding_fn is not None:
            result['embedding'] = self.embedding_fn(text)

        return result


class WeightedCAVBatchSampler:
    """
    Custom batch sampler that ensures balanced representation of behaviors.
    """

    def __init__(self, dataset: WeightedCAVDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by behavior
        self.behavior_indices = {}
        for idx, point in enumerate(dataset.data_points):
            behavior = point['behavior']
            if behavior not in self.behavior_indices:
                self.behavior_indices[behavior] = []
            self.behavior_indices[behavior].append(idx)

    def __iter__(self):
        # Create batches with balanced behavior representation
        batches = []

        # Shuffle indices within each behavior if needed
        if self.shuffle:
            for behavior in self.behavior_indices:
                random.shuffle(self.behavior_indices[behavior])

        # Create mixed batches
        all_indices = []
        for behavior, indices in self.behavior_indices.items():
            all_indices.extend(indices)

        if self.shuffle:
            random.shuffle(all_indices)

        # Create batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) > 0:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def load_precomputed_tensors(layer: int = 15, data_dir: str = "./weighted_cav_vectors"):
    """
    Load the precomputed tensor file containing all weighted vectors.

    This is useful for bulk operations or when you want to work with
    the vectors directly without the text data.
    """
    tensor_file = Path(data_dir) / f"all_weighted_vectors_layer{layer}.pt"

    if not tensor_file.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

    data = torch.load(tensor_file, map_location='cpu')

    return {
        'positive_vectors': data.get('positive'),  # Shape: (n_samples, vector_dim)
        'negative_vectors': data.get('negative'),  # Shape: (n_samples, vector_dim)
        'difference_vectors': data.get('difference'),  # Shape: (n_samples, vector_dim)
        'cavs': data.get('cavs'),  # Dict of behavior -> CAV vector
        'behaviors': data.get('behaviors')  # List of behavior names
    }


def create_adapter_training_data(
    embedding_model,  # Your embedding model
    tokenizer,  # Your tokenizer
    layer: int = 15,
    batch_size: int = 32,
    vector_type: str = "difference"
):
    """
    Example function showing how to create training data for an adapter model.

    You would replace the embedding_model and tokenizer with your actual models.
    """

    # Define embedding function using your model
    def generate_embedding(text: str) -> torch.Tensor:
        """
        This is where you'd use your embedding model.
        Replace this with your actual embedding generation code.
        """
        # Example placeholder - replace with your actual embedding model
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            outputs = embedding_model(**inputs)
            # You might use pooled output, last hidden state mean, etc.
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding

    # Create dataset
    dataset = WeightedCAVDataset(
        layer=layer,
        vector_type=vector_type,
        embedding_fn=generate_embedding
    )

    # Create dataloader with custom batch sampler for balanced training
    sampler = WeightedCAVBatchSampler(dataset, batch_size=batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,  # Set to 0 if using embedding function
        pin_memory=True
    )

    return dataloader, dataset


def example_training_loop():
    """
    Example showing how you might use this data in a training loop.
    """

    # Load precomputed vectors for analysis
    vectors_data = load_precomputed_tensors(layer=15)
    print(f"Loaded vectors with shapes:")
    print(f"  Positive: {vectors_data['positive_vectors'].shape if vectors_data['positive_vectors'] is not None else 'None'}")
    print(f"  Negative: {vectors_data['negative_vectors'].shape if vectors_data['negative_vectors'] is not None else 'None'}")
    print(f"  Difference: {vectors_data['difference_vectors'].shape if vectors_data['difference_vectors'] is not None else 'None'}")

    # Create dataset (without embedding function for this example)
    dataset = WeightedCAVDataset(layer=15, vector_type="difference")

    # Create simple dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example training loop structure
    for epoch in range(1):
        for batch_idx, batch in enumerate(dataloader):
            # Extract data
            texts = batch['text']
            target_vectors = batch['target_vector']
            behaviors = batch['behavior']

            # Here you would:
            # 1. Generate embeddings from texts using your model
            # 2. Pass embeddings through your adapter
            # 3. Compute loss between adapter output and target_vectors
            # 4. Backpropagate and update adapter weights

            if batch_idx == 0:
                print(f"\nBatch {batch_idx}:")
                print(f"  Batch size: {len(texts)}")
                print(f"  Target vector shape: {target_vectors.shape}")
                print(f"  Behaviors in batch: {set(behaviors)}")
                print(f"  First text sample: {texts[0][:100]}...")

        break  # Just show first epoch for example


if __name__ == "__main__":
    # Run example
    print("WeightedCAV Data Loader Example")
    print("="*60)

    example_training_loop()

    print("\n" + "="*60)
    print("Data structure summary:")
    print("- Each data point contains text (question + answer)")
    print("- Target is a weighted CAV vector (linear combination of 7 CAVs)")
    print("- Weights are normalized based on projection strengths")
    print("- You can choose positive, negative, or difference vectors")
    print("\nTo use with your embedding model:")
    print("1. Provide your embedding function to the dataset")
    print("2. The dataset will generate embeddings on-the-fly")
    print("3. Train your adapter to map embeddings -> weighted CAV vectors")