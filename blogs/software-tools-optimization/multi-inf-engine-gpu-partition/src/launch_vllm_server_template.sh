# vLLM instances
GPU_ID=$1
PORT=$2
WORKSPACE_PATH=$3
MODEL_PATH=$4
DEFAULT_HF_HOME="$HOME/.cache/huggingface"
HF_HOME="${5:-$DEFAULT_HF_HOME}"

echo "Mounting '${HF_HOME}' as HF_HOME".

docker stop vllm_$GPU_ID
docker rm vllm_$GPU_ID
docker run -d -it \
    --name vllm_$GPU_ID \
    --privileged \
    --shm-size 32G \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    -v "${HF_HOME}":/root/.cache/huggingface \
    -v "$WORKSPACE_PATH":/workspace \
    -p $PORT:8000 \
    --network vllm_nginx \
    --ipc=host \
    rocm/vllm:latest

docker exec vllm_$GPU_ID bash /workspace/launch_vllm_core.sh $GPU_ID $MODEL_PATH &
