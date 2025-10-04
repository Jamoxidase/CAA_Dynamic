# Weighted Vectors - CLI Usage

## Requirements
- Normalized CAVs for all 7 behaviors at `normalized_vectors/{behavior}/vec_layer_{N}_{model}.pt`
- Train AB data at `datasets/generate/{behavior}/generate_dataset.json` (default)
- Test AB data at `datasets/test/{behavior}/test_dataset_ab.json`
- Test open-ended data at `datasets/test/{behavior}/test_dataset_open_ended.json`

## Pipeline

### Step 1: Compute Projections
```bash
python compute_test_projections.py \
  --behaviors pride-humility envy-kindness gluttony-temperance greed-charity lust-chastity sloth-diligence wrath-patience \
  --layers 15 20 \
  --model_size 7b \
  [--dataset train] \
  [--use_base_model] \
  [--output_dir ./projections]
```

**Options:**
- `--behaviors`: Space-separated list (default: all 7)
- `--layers`: Space-separated layer numbers
- `--model_size`: `7b`, `8b`, `13b`, or `1.2b`
- `--dataset`: `train` or `test` (default: `train`)
- `--use_base_model`: Use base model instead of chat/instruct
- `--output_dir`: Where to save projections (default: `./projections`)

**Output:** `projections/{behavior}_{dataset}_projections_layer{N}_{model}.json`

---

### Step 2: Create Weighted Vectors
```bash
python create_weighted_test_vectors.py \
  --behaviors pride-humility envy-kindness gluttony-temperance greed-charity lust-chastity sloth-diligence wrath-patience \
  --layers 15 20 \
  --model_size 7b \
  [--dataset train] \
  [--use_base_model] \
  [--projection_dir ./projections] \
  [--output_dir ./vectors]
```

**Options:**
- `--behaviors`: Space-separated list (default: all 7)
- `--layers`: Space-separated layer numbers
- `--model_size`: `7b`, `8b`, `13b`, or `1.2b`
- `--dataset`: `train` or `test` (default: `train`)
- `--use_base_model`: Use base model instead of chat/instruct
- `--projection_dir`: Where projections are (default: `./projections`)
- `--output_dir`: Where to save vectors (default: `./vectors`)

**Output:**
- `vectors/{behavior}_{dataset}_weighted_vectors_layer{N}_{model}.json`
- `vectors/{behavior}_{dataset}_weighted_vectors_layer{N}_{model}.pt`

---

### Step 3: Run Weighted Steering Tests
```bash
python prompting_with_weighted_steering.py \
  --behaviors pride-humility envy-kindness gluttony-temperance greed-charity lust-chastity sloth-diligence wrath-patience \
  --layers 15 20 \
  --multipliers -5 -4 -3 -2 -1 0 1 2 3 4 5 \
  --model_size 7b \
  [--use_base_model] \
  [--vectors_dir ./vectors] \
  [--output_dir ./results]
```

**Options:**
- `--behaviors`: Space-separated list (default: all 7)
- `--layers`: Space-separated layer numbers (required)
- `--multipliers`: Space-separated multipliers (required)
- `--model_size`: `7b`, `8b`, `13b`, or `1.2b`
- `--use_base_model`: Use base model instead of chat/instruct
- `--vectors_dir`: Where vectors are (default: `./vectors`)
- `--output_dir`: Where to save results (default: `./results`)

**Output:**
- `results/ab/{behavior}_weighted_ab_layer{N}_mult{M}_{model}.json`
- `results/open_ended/{behavior}_weighted_open_ended_layer{N}_mult{M}_{model}.json`

---

## Quick Examples

### All 7 behaviors, single layer, Llama 7b chat (train data - default)
```bash
python compute_test_projections.py --layers 15 --model_size 7b
python create_weighted_test_vectors.py --layers 15 --model_size 7b
python prompting_with_weighted_steering.py --layers 15 --multipliers -1 0 1 --model_size 7b
```

### Generate CAV dataset from train data (for CAV dataset creation)
```bash
python compute_test_projections.py --layers 15 --model_size 7b --dataset train
python create_weighted_test_vectors.py --layers 15 --model_size 7b --dataset train
# Output: weighted CAVs with corresponding AB question pairs for all train data
```

### Using test data
```bash
python compute_test_projections.py --layers 15 --model_size 7b --dataset test
python create_weighted_test_vectors.py --layers 15 --model_size 7b --dataset test
python prompting_with_weighted_steering.py --layers 15 --multipliers -1 0 1 --model_size 7b
```

### Specific behaviors, multiple layers, Llama 7b base
```bash
python compute_test_projections.py --behaviors pride-humility envy-kindness --layers 10 15 20 --model_size 7b --use_base_model
python create_weighted_test_vectors.py --behaviors pride-humility envy-kindness --layers 10 15 20 --model_size 7b --use_base_model
python prompting_with_weighted_steering.py --behaviors pride-humility envy-kindness --layers 10 15 20 --multipliers -5 -3 -1 0 1 3 5 --model_size 7b --use_base_model
```

### LFM2 1.2b
```bash
python compute_test_projections.py --layers 2 5 --model_size 1.2b
python create_weighted_test_vectors.py --layers 2 5 --model_size 1.2b
python prompting_with_weighted_steering.py --layers 2 5 --multipliers -1 0 1 --model_size 1.2b
```

### Llama 3 8b (Daredevil)
```bash
python compute_test_projections.py --layers 15 20 25 --model_size 8b
python create_weighted_test_vectors.py --layers 15 20 25 --model_size 8b
python prompting_with_weighted_steering.py --layers 15 20 25 --multipliers -5 -1 0 1 5 --model_size 8b
```
