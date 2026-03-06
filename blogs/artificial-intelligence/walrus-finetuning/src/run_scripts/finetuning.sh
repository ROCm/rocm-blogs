#!/bin/bash -l

export OMP_NUM_THREADS=32
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export HSA_NO_SCRATCH_RECLAIM=1

# Algorithm selection modes for MIOpen
# If you already have a populated MIOpen user db and cache
# You can point to it here and use mode 1 or 3 to avoid re-tuning from scratch
export MIOPEN_USER_DB_PATH=/artifacts/miopen_db/user
export MIOPEN_CUSTOM_CACHE_DIR=/artifacts/miopen_db/cache
export MIOPEN_DEBUG_CONV_DIRECT=0

MODE="${MODE:-3}"
# if else block to dynamically set MIOpen algo search mode
if [ $MODE -eq 1 ]
then
    # Mode1: Fast startup 
    export MIOPEN_FIND_MODE=2 # Fast mode, use DB entries if they exist otherwise immediate-mode heuristic fallback
    export MIOPEN_FIND_ENFORCE=1 # No forced tuning
elif [ $MODE -eq 2 ]
then
    # Mode 2: Exhaustive algo search (may take hours)
    export MIOPEN_FIND_MODE=1 # Normal mode, exhaustive search
    export MIOPEN_FIND_ENFORCE=4 # SEARCH_DB_UPDATE, perform auto-tune and update DB even if entries exist  
elif [ $MODE -eq 3 ]
then
    # Mode 3: Hybrid
    export MIOPEN_FIND_MODE=3 # If db hit -> use that entry, if db miss -> use the existing find machinery.
    export MIOPEN_FIND_ENFORCE=3 # SEARCH_DB, perform auto-tune only if no entry exists in DB
elif [ $MODE -eq 4 ]
then
    # Mode 4: Default mode
    export MIOPEN_FIND_MODE=6 # Might trigger tuning even if db hit. 
    export MIOPEN_FIND_ENFORCE=1 # No forced tuning
    # OR
    # unset MIOPEN_FIND_MODE
    # unset MIOPEN_FIND_ENFORCE
fi

# Enable MIOpen logging for debugging
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_LOG_LEVEL=6

export WANDB_MODE="${WANDB_MODE:-offline}"
DISTRIBUTION="${DISTRIBUTION:-local}" # local, ddp, fsdp, hsdp
if [ "$DISTRIBUTION" = "local" ]; then
    NPROC_PER_NODE=1
elif [ "$DISTRIBUTION" = "ddp" ] || [ "$DISTRIBUTION" = "fsdp" ] || [ "$DISTRIBUTION" = "hsdp" ]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
else
    echo "Unknown distribution mode: $DISTRIBUTION"
    exit 1
fi

echo "Distribution: $DISTRIBUTION \
      NPROC_PER_NODE: $NPROC_PER_NODE \
      Tuning mode: $MODE"

# See walrus/walrus/configs/config.yaml for config options
torchrun --nproc_per_node=$NPROC_PER_NODE \
        ../train.py \
        distribution=$DISTRIBUTION

echo "Finetuning Done"