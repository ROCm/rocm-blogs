FROM rocm/dev-ubuntu-22.04:7.2-complete

WORKDIR /workspace

ENV ROCM_PATH=/opt/rocm
ENV AER_THRUST_BACKEND=ROCM
ENV QISKIT_AER_PACKAGE_NAME=qiskit-aer-gpu-rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    liblapack-dev \
    libomp-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Might need this if we face import errors
# ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/liblapack.so.3

RUN pip install --no-cache-dir "qiskit>=1.0,<2.0"

RUN git clone https://github.com/coketaste/qiskit-aer.git && \
    cd qiskit-aer && \
    git switch coketaste/amd-rocm-mi300

WORKDIR /workspace/qiskit-aer

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir pybind11 jupyter notebook "qiskit>=1.0,<2.0" "qiskit_nature<0.8" qiskit-algorithms pyscf mlflow

RUN python3 setup.py bdist_wheel -- \
    -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
    -DAER_THRUST_BACKEND=ROCM \
    -DAER_ROCM_ARCH=gfx942 \
    -DCMAKE_BUILD_TYPE=Release

RUN pip install --no-cache-dir dist/qiskit_aer_gpu_rocm-*.whl

WORKDIR /workspace

CMD ["/bin/bash"]