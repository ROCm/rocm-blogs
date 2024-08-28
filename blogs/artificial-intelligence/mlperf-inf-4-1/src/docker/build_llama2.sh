#!/bin/bash
BASE_IMAGE=rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging
VLLM_REV=799388d722e22ecb14d1011faaba54c4882cc8f5 # MLPerf-4.1
HIPBLASLT_BRANCH=8b71e7a8d26ba95774fdc372883ee0be57af3d28
FA_BRANCH=23a2b1c2f21de2289db83de7d42e125586368e66 # ck_tile - FA 2.5.9
RELEASE_TAG=${RELEASE_TAG:-latest}

git clone https://github.com/ROCm/vllm
pushd vllm
git checkout main
git pull
git checkout ${VLLM_REV}
git cherry-pick b9013696b23dde372cccecdbaf69f0c852008844 # optimizations for process output step, PR #104

docker build --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg HIPBLASLT_BRANCH=${HIPBLASLT_BRANCH} --build-arg FA_BRANCH=${FA_BRANCH} -f Dockerfile.rocm -t vllm_dev:${VLLM_REV} .

popd

docker build --build-arg BASE_IMAGE=vllm_dev:${VLLM_REV} -f Dockerfile.llama2 -t mlperf/llama_inference:${RELEASE_TAG} ..
