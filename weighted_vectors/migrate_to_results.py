#!/usr/bin/env python3
"""
Migrate weighted vector results to base results format.
Renames files to match SteeringSettings expectations while preserving data.
"""

import os
import json
import shutil
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
SOURCE_AB = SCRIPT_DIR / "results/ab"
SOURCE_OE = SCRIPT_DIR / "results/open_ended"
DEST_BASE = SCRIPT_DIR.parent / "results/weighted_vectors"

def parse_filename(filename):
    """
    Parse: sloth-diligence_weighted_ab_layer16_mult-3.0_7b.json
           sloth-diligence_weighted_open_ended_layer16_mult-3.0_7b.json
    Returns: (behavior, type, layer, multiplier, model_size)
    """
    name = filename.replace(".json", "")
    parts = name.split("_")

    # Find "weighted" position
    weighted_idx = parts.index("weighted")

    # Behavior is everything before "weighted"
    behavior = "_".join(parts[:weighted_idx])

    # Type is after "weighted" - could be "ab" or "open" followed by "ended"
    if parts[weighted_idx + 1] == "open" and parts[weighted_idx + 2] == "ended":
        test_type = "open_ended"
        type_end_idx = weighted_idx + 3
    else:
        test_type = parts[weighted_idx + 1]
        type_end_idx = weighted_idx + 2

    # Layer (format: layerN)
    layer_part = [p for p in parts if p.startswith("layer")][0]
    layer = int(layer_part.replace("layer", ""))

    # Multiplier (format: multX.X or mult-X.X)
    mult_part = [p for p in parts if p.startswith("mult")][0]
    multiplier = float(mult_part.replace("mult", ""))

    # Model size is last part
    model_size = parts[-1]

    return behavior, test_type, layer, multiplier, model_size

def create_base_filename(behavior, test_type, layer, multiplier, model_size):
    """Create filename matching base format"""
    return (
        f"results_layer={layer}_multiplier={multiplier}_"
        f"behavior={behavior}_type={test_type}_"
        f"use_base_model=False_model_size={model_size}_"
        f"weighted_vector=True.json"
    )

def migrate(dry_run=False):
    """
    Migrate weighted vector results to base format.

    Args:
        dry_run: If True, print what would be done without actually copying files
    """
    migrated_count = 0
    errors = []

    for test_type, source_dir in [("ab", SOURCE_AB), ("open_ended", SOURCE_OE)]:
        if not source_dir.exists():
            print(f"WARNING: Source directory not found: {source_dir}")
            continue

        for filename in sorted(os.listdir(source_dir)):
            if not filename.endswith(".json"):
                continue

            try:
                # Parse original filename
                behavior, parsed_type, layer, multiplier, model_size = parse_filename(filename)

                if parsed_type != test_type:
                    errors.append(f"Type mismatch in {filename}: {parsed_type} != {test_type}")
                    continue

                # Create destination filename
                new_filename = create_base_filename(behavior, test_type, layer, multiplier, model_size)

                # Create behavior directory
                dest_dir = DEST_BASE / behavior

                # Source and dest paths
                source_path = source_dir / filename
                dest_path = dest_dir / new_filename

                if dry_run:
                    print(f"[DRY RUN] Would copy:")
                    print(f"  FROM: {source_path}")
                    print(f"  TO:   {dest_path}")
                else:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    print(f"✓ {behavior}/{new_filename}")

                migrated_count += 1

            except Exception as e:
                errors.append(f"Error processing {filename}: {e}")

    print(f"\n{'[DRY RUN] Would migrate' if dry_run else 'Migrated'}: {migrated_count} files")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    return migrated_count, errors

def verify():
    """Verify migration completed successfully"""
    print("\n=== VERIFICATION ===")

    # Count source files
    source_ab_count = len([f for f in os.listdir(SOURCE_AB) if f.endswith(".json")])
    source_oe_count = len([f for f in os.listdir(SOURCE_OE) if f.endswith(".json")])
    source_total = source_ab_count + source_oe_count

    print(f"Source files: {source_total} ({source_ab_count} AB + {source_oe_count} open-ended)")

    # Count destination files
    dest_total = 0
    if DEST_BASE.exists():
        for behavior_dir in DEST_BASE.iterdir():
            if behavior_dir.is_dir():
                count = len([f for f in os.listdir(behavior_dir) if f.endswith(".json")])
                dest_total += count
                print(f"  {behavior_dir.name}: {count} files")

    print(f"Destination files: {dest_total}")

    if source_total == dest_total:
        print("✓ Counts match!")

        # Check data integrity on sample files
        print("\nSpot-checking data integrity...")
        sample_count = 0
        for behavior_dir in DEST_BASE.iterdir():
            if behavior_dir.is_dir():
                files = [f for f in os.listdir(behavior_dir) if f.endswith(".json")]
                if files:
                    sample_file = behavior_dir / files[0]
                    with open(sample_file) as f:
                        data = json.load(f)

                    # Check first item
                    item = data[0]
                    has_a_prob = "a_prob" in item
                    has_b_prob = "b_prob" in item
                    has_question = "question" in item

                    if "_type=ab_" in str(sample_file):
                        has_matching = "answer_matching_behavior" in item
                        print(f"  {sample_file.name}: a_prob={has_a_prob}, b_prob={has_b_prob}, question={has_question}, answer_matching_behavior={has_matching}")
                    else:
                        has_output = "model_output" in item
                        print(f"  {sample_file.name}: question={has_question}, model_output={has_output}")

                    sample_count += 1
                    if sample_count >= 3:
                        break

        print("\n✓ Migration verified successfully!")
        return True
    else:
        print(f"\n✗ Count mismatch! Expected {source_total}, got {dest_total}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate weighted vector results to base format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    args = parser.parse_args()

    print("=== WEIGHTED VECTOR MIGRATION ===\n")

    migrated, errors = migrate(dry_run=args.dry_run)

    if not args.dry_run and args.verify:
        verify()
    elif args.dry_run:
        print("\nRun without --dry-run to actually perform migration")
