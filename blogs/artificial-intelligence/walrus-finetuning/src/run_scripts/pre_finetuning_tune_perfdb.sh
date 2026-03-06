#!/bin/bash -l

export OMP_NUM_THREADS=32
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0

# Set paths for MIOpen user db and cache
export MIOPEN_USER_DB_PATH=/artifacts/miopen_db/user
export MIOPEN_CUSTOM_CACHE_DIR=/artifacts/miopen_db/cache
export MIOPEN_DEBUG_CONV_DIRECT=0

# Exhaustive algo search
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4
export WANDB_MODE=offline

torchrun --nproc_per_node=$NPROC_PER_NODE \
        ../train.py \
        distribution=local \
        data.module_parameters.max_samples=10 \
        data.module_parameters.batch_size=1 \
        trainer.max_epoch=1 \
        trainer.log_interval=1