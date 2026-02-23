#!/bin/bash

# Enhanced model averaging script for Wespeaker
# Usage: ./average_models.sh [exp_dir] [num_avg] [specific_epochs]
# Example 1 (Last 5): ./average_models.sh exp/wav2vec2_xlsr_8epochB16_3 5
# Example 2 (Specific): ./average_models.sh exp/wav2vec2_xlsr_8epochB16_3 0 "3 6 9"

exp_dir=${1:-exp/wav2vec2_xlsr_8epochB16_3}
num_avg=${2:-5}
specific_epochs=$3
avg_model=$exp_dir/models/avg_model.pt

# Ensure we are in the correct directory (examples/tidyvocie)
if [ ! -d "wespeaker" ] && [ -d "../../wespeaker" ]; then
    bin_path="../../wespeaker/bin/average_model.py"
    project_root="../.."
else
    bin_path="wespeaker/bin/average_model.py"
    project_root="."
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/$project_root

if [ -n "$specific_epochs" ]; then
    echo "Averaging SPECIFIC epochs: $specific_epochs"
    
    # Create a temporary directory to hold symlinks
    tmp_dir=$(mktemp -d)
    count=0
    
    for epoch in $specific_epochs; do
        src_model="$exp_dir/models/model_$epoch.pt"
        if [ -f "$src_model" ]; then
            ln -s "$(realpath $src_model)" "$tmp_dir/model_$epoch.pt"
            count=$((count + 1))
        else
            echo "Warning: Model for epoch $epoch not found at $src_model, skipping."
        fi
    done
    
    if [ $count -eq 0 ]; then
        echo "Error: No valid models found for the specified epochs."
        rm -rf "$tmp_dir"
        exit 1
    fi
    
    python3 $bin_path \
      --dst_model $avg_model \
      --src_path "$tmp_dir" \
      --num $count
    
    rm -rf "$tmp_dir"
else
    echo "Averaging LAST $num_avg models in: $exp_dir"
    python3 $bin_path \
      --dst_model $avg_model \
      --src_path $exp_dir/models \
      --num ${num_avg}
fi

if [ $? -eq 0 ]; then
    echo "Successfully created: $avg_model"
else
    echo "Error occurred during model averaging."
    exit 1
fi
