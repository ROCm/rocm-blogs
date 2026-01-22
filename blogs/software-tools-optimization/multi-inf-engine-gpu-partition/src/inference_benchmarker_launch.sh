MODEL="${1:-"superbigtree/Mistral-Nemo-Instruct-2407-FP8_aq"}"
VU_NUM=1024 # Number of virtual users
DOCKER_SPEC='ghcr.io/huggingface/inference-benchmarker:latest'

echo "'$0' with model '${MODEL}'; virtual user count = ${VU_NUM}"

docker run \
    --name inference-benchmarker-vllm \
    --rm -it --net host \
    -v $(pwd):/opt \
    "${DOCKER_SPEC}" \
    inference-benchmarker \
    --tokenizer-name $MODEL \
    --model-name $MODEL \
    --url http://localhost:8040 \
    --prompt-options "num_tokens=512,max_tokens=1024,min_tokens=16,variance=25000" \
    --decode-options "num_tokens=512,max_tokens=1024,min_tokens=16,variance=25000" \
    --no-console \
    --max-vus $VU_NUM
