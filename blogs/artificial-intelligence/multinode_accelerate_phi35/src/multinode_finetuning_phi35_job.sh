#!/bin/bash
#SBATCH --job-name=multinode_finetuning_phi35_job # Job name
#SBATCH --nodes=2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks (processes) per node
#SBATCH --cpus-per-task=224 # Number of CPUs per task (process)
#SBATCH --mem=0 # Request all available memory on the node
#SBATCH --partition=<intended_partition> # check available partitions with "sinfo" command
#SBATCH --gres=gpu:8 # Number of GPUs per node
#SBATCH --output=%x-%j.out
#SBATCH --err=%x-%j.err
#SBATCH --exclusive
#SBATCH --time=24:00:00 # Time limit

# Docker credentials
export DOCKER_USERNAME=<your-dockerhub-username>
export DOCKER_TOKEN=<your-personal-access-token>
export DOCKER_REGISTRY=docker.io

# Docker login 
echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin "$DOCKER_REGISTRY"

# Get the list of nodes and the first node (master node)
master_node=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Get the IP address of the master node
master_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$master_node bash -c "ip -f inet addr show rdma0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'")

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_ip
export SLURM_MASTER_PORT=29501
export SLURM_TOTAL_GPUS=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

# Define the Docker image
export DOCKER_IMAGE="<your-dockerhub-username>/multinode-finetuning:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0_accelerate_0.34.2"

# Define the mount points
export HOST_MOUNT="</your/login_node/local_folder>/multinode_finetuning/"
export CONTAINER_MOUNT="/usr/src/app"

# Optional: Print out the values for debugging
echo "Custom parameter values:"
echo "MASTER ADDRESS: $SLURM_MASTER_ADDR"
echo "MASTER_PORT: $SLURM_MASTER_PORT"
echo "NUMBER OF NODES REQUESTED: $SLURM_NNODES"
echo "NUMBER OF NODES ALLOCATED: $SLURM_JOB_NUM_NODES"
echo "NUMBER OF GPUS PER NODE: $SLURM_GPUS_ON_NODE"
echo "TOTAL GPUS: $SLURM_TOTAL_GPUS" 
echo "MACHINE RANK: $SLURM_NODEID"

# Run the Docker container with the script
srun bash -c 'docker run --pull always --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env SLURM_TOTAL_GPUS=$SLURM_TOTAL_GPUS \
 --env SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES \
 --env SLURM_NODEID=$SLURM_NODEID \
 --ipc=host \
 --network=host \
 --device=/dev/kfd \
 --device=/dev/dri \
 --shm-size=13G \
 --security-opt seccomp=unconfined \
 --group-add video \
 --privileged \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c "echo $(date); cd /usr/src/app; \
 accelerate launch \
 --multi_gpu \
 --num_machines=$SLURM_JOB_NUM_NODES \
 --num_processes=$SLURM_TOTAL_GPUS \
 --machine_rank=$SLURM_NODEID \
 --main_process_ip=$SLURM_MASTER_ADDR \
 --main_process_port=$SLURM_MASTER_PORT \
 --mixed_precision=no \
 --dynamo_backend=no \
 $CONTAINER_MOUNT/classification_finetuning_phi35.py; echo $(date)"'