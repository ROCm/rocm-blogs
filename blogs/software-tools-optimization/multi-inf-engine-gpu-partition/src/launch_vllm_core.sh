
GPU_ID=$1
MODEL_PATH=$2
export HIP_VISIBLE_DEVICES="$GPU_ID"
cd /root
vllm serve $MODEL_PATH --port 8000
