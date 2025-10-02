#!/bin/bash

# Script to merge data from more_data into main CAA project
# Only copies files that don't already exist

MORE_DATA_DIR="/Users/jlarbale/foopoo1/more_data"
CAA_DIR="/Users/jlarbale/foopoo1/CAA"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Merging more_data into CAA project ===${NC}"
echo ""

# Counter for new files
new_results=0
new_vectors=0

# Process each behavior directory
for behavior_dir in "$MORE_DATA_DIR"/*; do
    if [ -d "$behavior_dir" ]; then
        behavior=$(basename "$behavior_dir")

        # Skip non-behavior directories
        if [[ "$behavior" == ".ipynb_checkpoints" ]] || [[ "$behavior" == "open_ended_scores" ]]; then
            continue
        fi

        echo -e "${YELLOW}Processing: ${behavior}${NC}"

        # Create target directories if they don't exist
        mkdir -p "$CAA_DIR/results/$behavior"
        mkdir -p "$CAA_DIR/vectors/$behavior"

        # Copy results (JSON files)
        for file in "$behavior_dir"/*.json; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                target="$CAA_DIR/results/$behavior/$filename"

                if [ ! -f "$target" ]; then
                    cp "$file" "$target"
                    ((new_results++))
                    echo "  + Added result: $filename"
                fi
            fi
        done

        # Copy vectors (PT files)
        for file in "$behavior_dir"/*.pt; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                target="$CAA_DIR/vectors/$behavior/$filename"

                if [ ! -f "$target" ]; then
                    cp "$file" "$target"
                    ((new_vectors++))
                    echo "  + Added vector: $filename"
                fi
            fi
        done
    fi
done

echo ""
echo -e "${GREEN}=== Merge Complete ===${NC}"
echo -e "${GREEN}New result files added: $new_results${NC}"
echo -e "${GREEN}New vector files added: $new_vectors${NC}"

# Summary of 8b model data
echo ""
echo -e "${BLUE}=== 8b Model Data Summary ===${NC}"
count_8b=$(find "$CAA_DIR/results" -name "*model_size=8b*" | wc -l | tr -d ' ')
echo -e "Total 8b model files: $count_8b"

# Show unique layers for 8b
echo "8b model layers:"
find "$CAA_DIR/results/envy-kindness" -name "*model_size=8b*" -type f | \
    grep -o "layer=[0-9]*" | cut -d= -f2 | sort -n | uniq | tr '\n' ' '
echo ""
