#!/bin/bash

#SBATCH --job-name=FSDP
#SBATCH --gres=gpu:mi300:8
#SBATCH --output=fsdp_training_temp.txt 
#SBATCH --partition=amd-aiss
#SBATCH --exclusive
#SBATCH --time=01-01:20:00

echo "SLURM_NODEID:  $SLURM_NODEID"
echo "SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOB_ID : $SLURM_JOB_ID"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
echo "slurm job node list: $nodes"
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "head_node_ip : $head_node_ip"

## docker container setup and packages installation
srun -l bash ./docker_setup.sh $(pwd)

## test time
current_date_time=`date +%Y%m%d-%H%M%S`
echo "Current date current_date_time : $current_date_time"

## fine-tuning configuration
n_nodes=$SLURM_JOB_NUM_NODES
n_proc_per_node=$1
n_epoch=$2
n_batchsize=$3
model_path=$4
model_size=$5
save_model=$6
context_length="4096"

fine_tune_script="./recipes/quickstart/finetuning/finetuning.py"
fine_tune_dataset="alpaca_dataset"
checkpoints_path="finetune_checkpoints"

work_space_path="/root/"

### test run
echo "Launch: srun -l docker exec -w $work_space_path fsdp \
    torchrun  --nnodes $n_nodes --nproc_per_node $n_proc_per_node \
    --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
    $fine_tune_script  --model_name $model_path \
    --output_dir ./$checkpoints_path/fsdp_fine_tune_results/output_model_${n_nodes}_8_${model_size} \
    --dist_checkpoint_root_folder ./$checkpoints_path/fsdp_fine_tune_results/fsdp_model_finetuned_${n_nodes}_8_${model_size}  \
    --enable_fsdp  \
    --num_epochs $n_epoch --batch_size_training $n_batchsize \
    --dataset $fine_tune_dataset \
    --save_model $save_model \
    --context_length $context_length \
    "

srun -l docker exec -w $work_space_path fsdp torchrun  --nnodes $n_nodes --nproc_per_node $n_proc_per_node \
    --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
    $fine_tune_script  --model_name $model_path \
    --output_dir ./$checkpoints_path/fsdp_fine_tune_results/output_model_${n_nodes}_8_${model_size} \
    --dist_checkpoint_root_folder ./$checkpoints_path/fsdp_fine_tune_results/fsdp_model_finetuned_${n_nodes}_8_${model_size} \
    --enable_fsdp \
    --num_epochs $n_epoch --batch_size_training $n_batchsize \
    --dataset $fine_tune_dataset \
    --save_model $save_model \
    --context_length $context_length \

# save the blog output
mv fsdp_training_temp.txt fsdp_training_${current_date_time}_${model_size}_${n_nodes}_${n_proc_per_node}_${n_epoch}_${n_batchsize}.txt 
