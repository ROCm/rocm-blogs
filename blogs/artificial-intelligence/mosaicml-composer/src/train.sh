export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=7501
#echo "MASTER_ADDR="$MASTER_ADDR
#echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
#8GPUS per node
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:=8}

export DOCKER_USER=<INSERT USERID>
export DOCKER_PASS=<INSERT PASSWORD>

#RCCL info
export NCCL_DEBUG="${NCCL_DEBUG:=WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:=WARN}"
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:=1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
export NCCL_CHECKS_DISABLE=1
export NCCL_ALGO=Ring

docker login packages.xilinx.com -u "$DOCKER_USER" --password "$DOCKER_PASS"
docker pull packages.xilinx.com/instinct/dev-benchmark-300x:mpt-30b_rocm-v6.1.3_llm-foundry-v0.7.0_train

docker  run  --rm   --ipc=host --cap-add=SYS_PTRACE --network=host --device=/dev/kfd --device=/dev/dri \
                --security-opt seccomp=unconfined --group-add video --privileged   \
                        -v /data/data:/data \
                        --env SLURM_NNODES=$SLURM_NNODES \
                        -e MASTER_ADDR=$MASTER_ADDR \
                        -e MASTER_PORT=$MASTER_PORT \
                        -e WORLD_SIZE=$WORLD_SIZE \
                        -e LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE \
                        -e NODE_RANK=$SLURM_NODEID \
                        -e LOCAL_RANK=$SLURM_LOCALID \
                        -e NCCL_DEBUG=$NCCL_DEBUG \
                        -e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS \
                        -e PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER \
                        -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
                        -e NCCL_CHECKS_DISABLE=$NCCL_CHECKS_DISABLE \
                        -e NCCL_ALGO=$NCCL_ALGO \
                        packages.xilinx.com/instinct/dev-benchmark-300x:mpt-30b_rocm-v6.1.3_llm-foundry-v0.7.0_train \
                        /bin/bash -c "bash run_mpt30b.sh"
