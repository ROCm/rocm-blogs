#!/bin/bash

NUM_CONTAINERS=$1  # Set how many containers you want to launch
MODEL_DIR=$2
START_GPU_ID=0
START_PORT=8081
SCRIPT="./launch_vllm_server_template.sh"
LOG_FILE=$3

for ((i=0; i < NUM_CONTAINERS; i++)); do
    docker stop vllm_$i
    docker rm vllm_$i
done

for ((i=0; i < NUM_CONTAINERS; i++)); do
    GPU_ID=$((START_GPU_ID + i))
    PORT=$((START_PORT + i))
    bash "$SCRIPT" "$GPU_ID" "$PORT" "$PWD" "$MODEL_DIR"
    # Wait for the server to start by monitoring the log  // we change the server launch in a sequential way
    # Wait until (i+1) servers have started
    EXPECTED_COUNT=$((i + 1))
    echo "Waiting for $EXPECTED_COUNT vLLM server(s) to start..."

    while true; do
        COUNT=$(grep -c "Starting vLLM API server" "$LOG_FILE")
        if (( COUNT >= EXPECTED_COUNT )); then
            echo "$EXPECTED_COUNT server(s) have started."
            break
        fi
        sleep 1
    done
done
