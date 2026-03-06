#!/bin/bash -l

export OMP_NUM_THREADS=32
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export HSA_NO_SCRATCH_RECLAIM=1

# If you already have a populated MIOpen user db and cache
# You can point to it here and use mode 1 or 3 to avoid re-tuning from scratch
export MIOPEN_USER_DB_PATH=/artifacts/miopen_db/user
export MIOPEN_CUSTOM_CACHE_DIR=/artifacts/miopen_db/cache
export MIOPEN_DEBUG_CONV_DIRECT=0

MODE="${MODE:-3}"
# if else block below to dynamically set MIOpen algo search mode
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

export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_PATH="${CONFIG_PATH:-/artifacts/finetuning_logs/experiment_dir/sciai_ft_experiment-postn-delta-Isotr[Space-Adapt-]-AdamW-8e-05/finetune/0/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/artifacts/finetuning_logs/experiment_dir/sciai_ft_experiment-postn-delta-Isotr\[Space-Adapt-\]-AdamW-8e-05/finetune/0/checkpoints/best/}"

# Launch the training script
# Folder structures defined by the train script can enter validation just by pointing to a config and weight folder. The rest of the settings are telling the run to validate differently than during training.
python ../train.py \
    --config-path=$CONFIG_PATH \
    --config-name='extended_config.yaml' \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++folder_override=$CHECKPOINT_PATH \
    '++trainer.validation_suite=[{_target_:the_well.benchmark.metrics.NRMSE},{_target_:the_well.benchmark.metrics.VRMSE},{_target_:the_well.benchmark.metrics.PearsonR}]' \
    '++trainer.validation_trajectory_metrics=[]' \
    '++trainer.batch_aggregation_fns=[torch.mean,torch.median,torch.std]' \
    '++data.module_parameters.max_rollout_steps=200' \
    '++data.module_parameters.start_rollout_valid_output_at_t=17' \
    '++trainer.max_rollout_steps=200' \
    '++data.well_base_path=/artifacts/the_well/datasets/' \
    '++logger.wandb_project_name=eval_walrus_example'
