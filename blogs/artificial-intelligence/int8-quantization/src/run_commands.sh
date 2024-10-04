#!/bin/bash

# Log file
LOGFILE="output_4_models.log"

# Clear the log file if it exists
> $LOGFILE

# Array of model repositories
MODEL_REPOS=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Meta-Llama-3-8B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-chat-hf")

# Run commands and log output
{
    echo "Processing models"
    
    # Loop through the model repositories
    for MODEL_REPO in "${MODEL_REPOS[@]}"; do
        # Prepare/download the model
        ./scripts/prepare.sh $MODEL_REPO

        echo -e "\n**************Running baseline with $MODEL_REPO..."
        python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
        echo -e "\n**************Running torch.compile with $MODEL_REPO..."
        python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"

        echo "Setting DEVICE to cuda..."
        export DEVICE=cuda

        echo -e "\n**************Quantizing and running commands with $MODEL_REPO..."
        python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
        echo -e "\n**************Running int8 with $MODEL_REPO..."
        python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
    done
} &> $LOGFILE
