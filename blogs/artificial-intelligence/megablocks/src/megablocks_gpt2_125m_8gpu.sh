#!/bin/bash

TRAINING_STEPS=2000

##
### Pre-training for GPT2 125M parameter.
##

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size 64 \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.0006 \
--min-lr 0.00006 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"


VOCAB_FILE="/megablocks/third_party/Stanford-Megatron-LM/tools/gpt2-vocab.json"
MERGE_FILE="/megablocks/third_party/Stanford-Megatron-LM/tools/gpt2-merges.txt"
DATA_PATH="/megablocks/third_party/Stanford-Megatron-LM/tools/my-gpt2_text_document"
CHECKPOINT_PATH="/megablocks/third_party/Stanford-Megatron-LM/checkpoints"

DATA_ARGUMENTS="\
--data-path ${DATA_PATH} \
--vocab-file ${VOCAB_FILE} \
--merge-file ${MERGE_FILE} \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--bf16 \
--DDP-impl local \
--no-async-tensor-model-parallel-allreduce \
--no-gradient-accumulation-fusion"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ${CHECKPOINT_PATH}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

torchrun ${DISTRIBUTED_ARGUMENTS} \
       /megablocks/third_party/Stanford-Megatron-LM/pretrain_gpt.py \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ${CHECKPOINT_PATH}/train.log