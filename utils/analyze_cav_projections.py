#!/usr/bin/env python3
"""
Analyze the CAV projection data to find interesting patterns.
"""

import json
import numpy as np
import pandas as pd
import argparse
from typing import Dict, List
import os

def load_projection_data(projection_dir: str, layer: int) -> Dict:
    """Load all projection data for a given layer."""

    all_data = {}
    for behavior in [
        "envy-kindness",
        "gluttony-temperance",
        "greed-charity",
        "lust-chastity",
        "pride-humility",
        "sloth-diligence",
        "wrath-patience"
    ]:
        filepath = os.path.join(projection_dir, f"{behavior}_cav_projections_layer{layer}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data[behavior] = json.load(f)
        else:
            print(f"Warning: Could not find {filepath}")

    return all_data

def create_projection_matrix(projection_data: Dict) -> pd.DataFrame:
    """
    Create a matrix showing mean projections of each behavior's data onto each CAV.
    Rows: source behavior (data), Columns: CAV behavior (projection target)
    """

    behaviors = list(projection_data.keys())
    matrix = pd.DataFrame(index=behaviors, columns=behaviors, dtype=float)

    for source_behavior, data in projection_data.items():
        mean_projs = {}

        # Calculate mean projection for difference (pos - neg)
        for dp in data["data_points"]:
            if "difference" in dp and "cav_projections" in dp["difference"]:
                for cav_behavior, proj in dp["difference"]["cav_projections"].items():
                    if cav_behavior not in mean_projs:
                        mean_projs[cav_behavior] = []
                    mean_projs[cav_behavior].append(proj)

        for cav_behavior, projs in mean_projs.items():
            if cav_behavior in behaviors:
                matrix.loc[source_behavior, cav_behavior] = np.mean(projs)

    return matrix

def find_cross_behavior_patterns(projection_data: Dict, threshold: float = 0.5) -> Dict:
    """
    Find examples where multiple behavior CAVs are strongly present.
    """

    patterns = {}

    for source_behavior, data in projection_data.items():
        patterns[source_behavior] = []

        for dp in data["data_points"]:
            if "difference" not in dp:
                continue

            # Get cosine similarities for this example
            cosine_sims = dp["difference"].get("cav_cosine_similarities", {})

            # Find behaviors with high similarity
            strong_signals = {k: v for k, v in cosine_sims.items() if abs(v) > threshold}

            if len(strong_signals) > 1:  # Multiple behaviors present
                patterns[source_behavior].append({
                    "index": dp["index"],
                    "question": dp["question"][:100] + "...",
                    "signals": strong_signals
                })

    return patterns

def analyze_behavior_relationships(projection_data: Dict) -> pd.DataFrame:
    """
    Compute correlation between different behavior CAVs based on their projections.
    """

    behaviors = list(projection_data.keys())

    # Collect all projection values
    projection_vectors = {b: [] for b in behaviors}

    # For each data point across all behaviors
    for source_behavior, data in projection_data.items():
        for dp in data["data_points"]:
            if "difference" in dp and "cav_projections" in dp["difference"]:
                for cav_behavior in behaviors:
                    if cav_behavior in dp["difference"]["cav_projections"]:
                        projection_vectors[cav_behavior].append(
                            dp["difference"]["cav_projections"][cav_behavior]
                        )

    # Create DataFrame and compute correlations
    df = pd.DataFrame(projection_vectors)
    return df.corr()

def find_outliers(projection_data: Dict, std_threshold: float = 2.0) -> Dict:
    """
    Find data points with unusually high or low projections.
    """

    outliers = {}

    for source_behavior, data in projection_data.items():
        behavior_outliers = []

        # Collect self-projections (should typically be highest)
        self_projections = []
        for dp in data["data_points"]:
            if "difference" in dp and "cav_projections" in dp["difference"]:
                if source_behavior in dp["difference"]["cav_projections"]:
                    self_projections.append(dp["difference"]["cav_projections"][source_behavior])

        if not self_projections:
            continue

        mean_proj = np.mean(self_projections)
        std_proj = np.std(self_projections)

        # Find outliers
        for i, dp in enumerate(data["data_points"]):
            if "difference" not in dp or "cav_projections" not in dp["difference"]:
                continue

            self_proj = dp["difference"]["cav_projections"].get(source_behavior, 0)
            z_score = (self_proj - mean_proj) / (std_proj + 1e-10)

            if abs(z_score) > std_threshold:
                behavior_outliers.append({
                    "index": dp["index"],
                    "question": dp["question"][:100] + "...",
                    "self_projection": self_proj,
                    "z_score": z_score,
                    "mean": mean_proj,
                    "std": std_proj
                })

        outliers[source_behavior] = behavior_outliers

    return outliers

def main():
    parser = argparse.ArgumentParser(description='Analyze CAV projections')
    parser.add_argument('--layer', type=int, default=15,
                        help='Layer number to analyze')
    parser.add_argument('--projection_dir', type=str, default='./cav_projections',
                        help='Directory containing projection data')

    args = parser.parse_args()

    print(f"Loading projection data for layer {args.layer}...")
    projection_data = load_projection_data(args.projection_dir, args.layer)

    if not projection_data:
        print("No projection data found!")
        return

    print(f"Loaded data for {len(projection_data)} behaviors")

    # Create projection matrix
    print("\n" + "="*60)
    print("PROJECTION MATRIX (Mean Difference Projections)")
    print("Rows: Source behavior (training data)")
    print("Columns: CAV behavior (projection target)")
    print("="*60)

    matrix = create_projection_matrix(projection_data)
    print("\n", matrix.round(4))

    # Diagonal should be strongest (self-projection)
    print("\nDiagonal values (self-projections):")
    for behavior in matrix.index:
        if behavior in matrix.columns:
            print(f"  {behavior}: {matrix.loc[behavior, behavior]:.6f}")

    # Find off-diagonal patterns
    print("\n" + "="*60)
    print("STRONGEST CROSS-BEHAVIOR PROJECTIONS")
    print("="*60)

    for source in matrix.index:
        row = matrix.loc[source]
        # Sort and exclude self
        sorted_projs = row.drop(source).sort_values(ascending=False)
        print(f"\n{source} data projects most strongly onto:")
        for i, (target, value) in enumerate(sorted_projs.head(3).items()):
            print(f"  {i+1}. {target}: {value:.6f}")

    # Find behavior correlations
    print("\n" + "="*60)
    print("BEHAVIOR CAV CORRELATIONS")
    print("="*60)

    correlations = analyze_behavior_relationships(projection_data)
    print("\n", correlations.round(3))

    # Find interesting cross-behavior examples
    print("\n" + "="*60)
    print("EXAMPLES WITH MULTIPLE STRONG CAV SIGNALS")
    print("="*60)

    patterns = find_cross_behavior_patterns(projection_data, threshold=0.3)

    for behavior, examples in patterns.items():
        if examples:
            print(f"\n{behavior}:")
            for ex in examples[:3]:  # Show top 3
                print(f"  Example {ex['index']}: {ex['question']}")
                print(f"    Signals: {', '.join([f'{k}:{v:.3f}' for k, v in ex['signals'].items()])}")

    # Find outliers
    print("\n" + "="*60)
    print("OUTLIER EXAMPLES (|z-score| > 2)")
    print("="*60)

    outliers = find_outliers(projection_data, std_threshold=2.0)

    for behavior, outlier_list in outliers.items():
        if outlier_list:
            print(f"\n{behavior}:")
            for out in outlier_list[:3]:  # Show top 3
                print(f"  Example {out['index']}: z={out['z_score']:.2f}")
                print(f"    Question: {out['question']}")
                print(f"    Projection: {out['self_projection']:.6f} (mean: {out['mean']:.6f})")

if __name__ == "__main__":
    main()