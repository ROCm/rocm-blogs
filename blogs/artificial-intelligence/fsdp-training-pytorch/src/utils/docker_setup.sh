#!/bin/bash
docker stop fsdp
docker remove fsdp
docker run --name fsdp -t -d --device=/dev/kfd --device=/dev/dri/ --network=host --group-add=video --shm-size 8G -v $1:/root  rocm/pytorch:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0
docker exec -w /root fsdp bash installation_script.sh
