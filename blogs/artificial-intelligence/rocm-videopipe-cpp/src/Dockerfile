FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
ARG GPU_TARGETS="gfx942"
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake

WORKDIR /workspace

# MIGraphX dependencies
RUN apt-get install -y --no-install-recommends \
    libmsgpack-cxx-dev pybind11-dev

# MIGraphX build
RUN git clone https://github.com/ROCm/AMDMIGraphX.git migraphx \
    && cd migraphx \
    && pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz \
    && rbuild build -d depend -B build -DGPU_TARGETS=$GPU_TARGETS --cxx=/opt/rocm/bin/amdclang++ --cc=/opt/rocm/bin/amdclang -DMIGRAPHX_STRIP_SYMBOLS=ON \
    && cd build \
    && make install

# rocDecode build
RUN git clone https://github.com/ROCm/rocDecode.git \
    && cd rocDecode \
    && python3 rocDecode-setup.py --developer=ON \
    && mkdir build \
    && cd build \
    && cmake ../ \
    && make -j8 \
    && make install

# Rebuild MIGraphX to workaround issue with LLVM
RUN cd migraphx \
    && rbuild build -d depend -B build -DGPU_TARGETS=$GPU_TARGETS --cxx=/opt/rocm/bin/amdclang++ --cc=/opt/rocm/bin/amdclang -DMIGRAPHX_STRIP_SYMBOLS=ON \
    && cd build \
    && make install

CMD ["/bin/bash"]
