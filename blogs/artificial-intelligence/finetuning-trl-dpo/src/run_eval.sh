#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <folder_path1> [folder_path2 ... folder_pathN]"
  exit 1
fi

for folder_path in "$@"
do
  echo "Running evaluation for folder: $folder_path"

  # Extract just the folder name from the path to use in output folder name
  folder_name=$(basename "$folder_path")

  accelerate launch -m lm_eval --model hf --model_args pretrained=${folder_path},attn_implementation=flash_attention_2 --tasks openllm --batch_size 64 --output_path eval_results/${folder_name}

  echo "Finished evaluation for $folder_path"
done

