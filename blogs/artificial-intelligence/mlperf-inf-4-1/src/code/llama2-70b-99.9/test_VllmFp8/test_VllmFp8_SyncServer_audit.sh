#!/bin/bash

set -xeu

N_SAMPLES=${N_SAMPLES:-24576}
TP=1
DP=${DP:-8}

export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1

export HARNESS_DISABLE_VLLM_LOGS=1
export VLLM_LOGGING_LEVEL=ERROR

MODEL_PATH=/data/llm/llama2-70b-chat/
DATASET_PATH=/data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
QUANTIZED_WEIGHTS_PATH=quantized/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors/llama.safetensors
QUANTIZATION_PARAM_PATH=/app/kv_cache_scales.json

MLPERF_CONF=/app/mlperf_inference/mlperf.conf
USER_CONF="${USER_CONF:-/lab-mlperf-inference/code/llama2-70b-99.9/mlperf_config_VllmFp8/user.conf}"
AUDIT_CONF=/lab-mlperf-inference/code/llama2-70b-99.9/VllmFp8/audit.config

SUBMISSION=${SUBMISSION:-0}
if [ "$SUBMISSION" -eq "0" ]; then
    BASE_LOG_DIR="${BASE_LOG_DIR:-${LAB_CLOG}/server/`date +%m%d-%H%M%S`}"
    LOG_DIR=${BASE_LOG_DIR}/audit
else
    TS_NOW=`date +%m%d-%H%M%S`
    TS_RESULTS="${TS_START_BENCHMARKS:-${TS_NOW}}"

    BASE_LOG_DIR="${BASE_LOG_DIR:-${LAB_CLOG}/${TS_RESULTS}/Server}"
    LOG_DIR=${BASE_LOG_DIR}/audit
fi
mkdir -p $LOG_DIR

env | sort >> ${LOG_DIR}/ct-env.txt
cp $USER_CONF ${LOG_DIR}/user.conf

cp /app/mlperf_inference/compliance/nvidia/TEST06/audit.config $AUDIT_CONF

cd /lab-mlperf-inference/code/llama2-70b-99.9/VllmFp8
python3 /lab-mlperf-inference/code/llama2-70b-99.9/VllmFp8/mainVllmFp8_SyncServer.py \
    --scenario Server \
    --output-log-dir ${LOG_DIR} \
    --model-path $MODEL_PATH \
    --mlperf-conf $MLPERF_CONF \
    --user-conf $USER_CONF \
    --total-sample-count $N_SAMPLES \
    --dataset-path $DATASET_PATH \
    --dtype float16 \
    --backend vllm \
    --device cuda:0 \
    --kv-cache-dtype fp8 \
    -tp ${TP} \
    -dp ${DP} \
    --quantization fp8 \
    --quantized-weights-path ${QUANTIZED_WEIGHTS_PATH} \
    --quantization-param-path ${QUANTIZATION_PARAM_PATH} \
    --enable-warm-up \
    --enable-batcher \
    2>&1 | tee ${LOG_DIR}/audit.server.DP${DP}TP${TP}.${N_SAMPLES}.log


python /app/mlperf_inference/compliance/nvidia/TEST06/run_verification.py -s Server -c ${LOG_DIR} -o ${LOG_DIR}/compliance -d int32


# Clean
rm $AUDIT_CONF
