SERVER_NUM=$1
MODEL=$2
VU_NUM=$3
LOG_FILE=$4

# Step 1: Launch servers
bash launch_vllm_servers.sh $SERVER_NUM $MODEL $LOG_FILE

# Step 2: Generate nginx config
bash generate_nginx_conf.sh $SERVER_NUM

# Step 3: Restart nginx container
docker stop nginx-lb
docker rm nginx-lb
docker run -itd -p 8040:80 --network vllm_nginx \
    -v $(pwd)/nginx_conf/nginx.conf:/etc/nginx/nginx.conf:ro \
    --name nginx-lb nginx:latest

# Step 4: Wait for all the servers up
PATTERN="Starting vLLM API server"
REQUIRED_COUNT=$SERVER_NUM

echo "Waiting for $REQUIRED_COUNT vLLM server(s) to start..."

while true; do
    count=$(grep -c "$PATTERN" "$LOG_FILE" 2>/dev/null || echo 0)
    if (( count >= REQUIRED_COUNT )); then
        echo "✅ Detected $count servers ready. Proceeding to send requests..."
        break
    else
        echo "⏳ Detected $count/$REQUIRED_COUNT servers ready. Waiting..."
        sleep 5
    fi
done

# Step 5: Send requests
echo "Starting to generate requests for benchmarking"
docker run \
    --rm -it --net host \
    -v $(pwd):/opt \
    ghcr.io/huggingface/inference-benchmarker:latest \
    inference-benchmarker \
    --tokenizer-name $MODEL \
    --model-name $MODEL \
    --url http://localhost:8040 \
    --prompt-options "num_tokens=512,max_tokens=1024,min_tokens=16,variance=25000" \
    --decode-options "num_tokens=512,max_tokens=1024,min_tokens=16,variance=25000" \
    --no-console \
    --max-vus $VU_NUM
