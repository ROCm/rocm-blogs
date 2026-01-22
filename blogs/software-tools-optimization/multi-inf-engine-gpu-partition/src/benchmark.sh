#!/bin/bash

LOG_DIR="$1"
shift
# Check if folder name was provided
if [ -z "$LOG_DIR" ]; then
    echo "Error: Please provide a folder name to save logs."
    echo "Usage: $0 <log_folder_name> gpu_count1 gpu_count2 ...ls"
    exit 1
fi

# Create the log folder if it doesn't exist
mkdir -p "$LOG_DIR"

SERVER_NUM=("$@")
VU_NUM_LIST=(1024)

MODEL="superbigtree/Mistral-Nemo-Instruct-2407-FP8_aq"

for vu in "${VU_NUM_LIST[@]}"; do
    for server in "${SERVER_NUM[@]}"; do
        LOG_FILE="$LOG_DIR/log_${server}_servers_${vu}_vusers.txt"
        echo "Running ---> benchmark_core.sh $server $MODEL $vu $LOG_FILE" | tee "$LOG_FILE"
        bash benchmark_core.sh "$server" "$MODEL" "$vu" "$LOG_FILE" &> "$LOG_FILE"
    done
done
