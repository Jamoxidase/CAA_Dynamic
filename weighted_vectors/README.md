# Weighted CAV Vectors Pipeline

This directory contains the complete pipeline for creating and testing **weighted behavioral steering vectors** that capture multi-dimensional behavioral signals from activations.

## Overview

Unlike standard CAA which uses single "pure" behavioral vectors, this approach:

1. **Re-analyzes test data activations** to decompose them into proportional contributions from ALL 7 behavioral CAVs
2. **Creates custom steering vectors** for each test question that represent the weighted combination of all behaviors present in that activation
3. **Tests with 1:1 mapping** - each custom vector is only applied to its corresponding test question (AB and open-ended versions)

This preserves the subtle multi-dimensional behavioral signals present in individual data points.

## Pipeline Steps

### 1. Compute Test Projections

**Script:** `compute_test_projections.py`

Analyzes test AB data to compute how much each of the 7 behavioral CAVs is represented in each test question's activation.

```bash
python compute_test_projections.py \
  --behaviors pride-humility envy-kindness \
  --layers 15 20 \
  --model_size 7b \
  --output_dir ./projections
```

**What it does:**
- Loads test AB data for specified behaviors
- Gets model activations for answer A (matching) and answer B (not matching)
- Computes difference: `diff = activation_A - activation_B`
- Projects `diff` onto all 7 normalized CAVs
- Saves projection strengths for each CAV per test question

**Output:** `projections/{behavior}_test_projections_layer{N}_{model}.json`

### 2. Create Weighted Vectors

**Script:** `create_weighted_test_vectors.py`

Creates custom steering vectors by combining all 7 CAVs weighted by their projection strengths.

```bash
python create_weighted_test_vectors.py \
  --behaviors pride-humility envy-kindness \
  --layers 15 20 \
  --model_size 7b \
  --projection_dir ./projections \
  --output_dir ./vectors
```

**What it does:**
- Loads projection data from step 1
- For each test question:
  - Weights = normalized projections (sum to 1, preserving proportionality)
  - Custom vector = `w1×CAV1 + w2×CAV2 + ... + w7×CAV7`
  - Normalizes to mean CAV norm (matching CAV normalization standards)
- Saves both JSON (metadata) and PT (tensors) files

**Output:**
- `vectors/{behavior}_weighted_vectors_layer{N}_{model}.json` (metadata)
- `vectors/{behavior}_weighted_vectors_layer{N}_{model}.pt` (tensors for steering)

### 3. Run Weighted Steering Tests

**Script:** `prompting_with_weighted_steering.py`

Runs AB and open-ended tests using the custom weighted vectors for steering.

```bash
python prompting_with_weighted_steering.py \
  --behaviors pride-humility envy-kindness \
  --layers 15 20 \
  --multipliers -5 -3 -1 0 1 3 5 \
  --model_size 7b \
  --vectors_dir ./vectors \
  --output_dir ./results
```

**What it does:**
- Loads weighted vectors for each behavior
- For each test question:
  - Applies its custom weighted vector at various multipliers
  - Runs AB test (measures A vs B probabilities)
  - Runs corresponding open-ended test (generates response)
- **1:1 mapping**: Vector from question N only steers question N
- Saves results with full metadata (weights, projections, etc.)

**Output:**
- `results/ab/{behavior}_weighted_ab_layer{N}_mult{M}_{model}.json`
- `results/open_ended/{behavior}_weighted_open_ended_layer{N}_mult{M}_{model}.json`

## Key Design Decisions

### Normalization Strategy

Weighted vectors are normalized to **mean CAV norm** (not unit norm), matching the standard from `normalize_vectors.py`:

```python
# CAVs are normalized so all have the same mean norm
cav_norms = [cav.norm().item() for cav in cavs.values()]
mean_norm = np.mean(cav_norms)

# Weighted vector is scaled to match this
weighted_vector = weighted_vector * (mean_norm / current_norm)
```

This preserves:
- Proportionality of underlying behavioral signals (via normalized weights)
- Consistency with existing CAV standards
- Comparable magnitude across all steering vectors

### Weight Calculation

Weights are computed from projection strengths:

```python
weights = projections_to_all_cavs  # Raw dot products
weights = weights / np.sum(np.abs(weights))  # Normalize to sum to 1
```

This ensures the weighted combination reflects the **proportional representation** of each behavior in the original activation.

### 1:1 Question Mapping

Each custom vector is **only** applied to its source question:
- AB test question → custom vector → AB test with that vector
- Same question's open-ended version → same custom vector → open-ended test

This prevents cross-contamination and preserves the specificity of each activation's behavioral signature.

## Directory Structure

```
weighted_vectors/
├── README.md                          # This file
├── compute_test_projections.py        # Step 1: Compute projections
├── create_weighted_test_vectors.py    # Step 2: Create weighted vectors
├── prompting_with_weighted_steering.py # Step 3: Run steering tests
├── projections/                       # Projection data (step 1 output)
│   └── {behavior}_test_projections_layer{N}_{model}.json
├── vectors/                           # Weighted vectors (step 2 output)
│   ├── {behavior}_weighted_vectors_layer{N}_{model}.json
│   └── {behavior}_weighted_vectors_layer{N}_{model}.pt
└── results/                           # Test results (step 3 output)
    ├── ab/
    │   └── {behavior}_weighted_ab_layer{N}_mult{M}_{model}.json
    └── open_ended/
        └── {behavior}_weighted_open_ended_layer{N}_mult{M}_{model}.json
```

## Example Usage

### Full Pipeline for Pride-Humility

```bash
# Step 1: Compute projections for test data
python compute_test_projections.py \
  --behaviors pride-humility \
  --layers 15 \
  --model_size 7b

# Step 2: Create weighted vectors
python create_weighted_test_vectors.py \
  --behaviors pride-humility \
  --layers 15 \
  --model_size 7b

# Step 3: Run steering tests
python prompting_with_weighted_steering.py \
  --behaviors pride-humility \
  --layers 15 \
  --multipliers -5 -4 -3 -2 -1 0 1 2 3 4 5 \
  --model_size 7b
```

### All Seven Deadly Sins

```bash
BEHAVIORS="pride-humility envy-kindness gluttony-temperance greed-charity lust-chastity sloth-diligence wrath-patience"

# Run full pipeline
python compute_test_projections.py --behaviors $BEHAVIORS --layers 15 --model_size 7b
python create_weighted_test_vectors.py --behaviors $BEHAVIORS --layers 15 --model_size 7b
python prompting_with_weighted_steering.py --behaviors $BEHAVIORS --layers 15 --multipliers -1 0 1 --model_size 7b
```

### Using Base Models

Add `--use_base_model` flag to all scripts:

```bash
python compute_test_projections.py \
  --behaviors pride-humility \
  --layers 15 \
  --model_size 7b \
  --use_base_model
```

## Comparison with Standard CAA

| Aspect | Standard CAA | Weighted CAA |
|--------|-------------|--------------|
| **Vector per behavior** | 1 pure CAV | ~128 custom vectors (one per test question) |
| **Signal representation** | Single behavior only | Weighted combination of all 7 behaviors |
| **Application** | Same vector for all test questions | 1:1 mapping: vector N only for question N |
| **Captures** | Primary behavioral direction | Multi-dimensional behavioral signature |
| **Use case** | General steering toward single behavior | Precise steering preserving activation's full behavioral profile |

## Notes

- All scripts support `--behaviors` parameter to process specific behaviors
- Compatible with all model sizes: 7b, 8b, 13b, 1.2b (LFM2)
- Base model support via `--use_base_model` flag
- Results include full metadata: weights, projections, questions, etc.
- Designed to work alongside standard CAA pipeline without conflicts
