---
blogpost: true
blog_title: "GROMACS on AMD Instinct GPUs: A Complete Build Guide"
date: 24 Mar 2026
author: 'David Björelind, Sebastian Remander, Gerardo del Muro Gonzalez'
thumbnail: 'gromacs_build_guide_thumbnail.png'
tags: Scientific Computing, AI/ML, HPC, Performance
category: Applications & models
target_audience: HPC engineers, computational scientists, and system administrators who need to deploy GPU-accelerated molecular dynamics workloads on AMD Instinct hardware.
key_value_propositions: Provides a complete, validated build procedure for the full GROMACS stack (UCX, OpenMPI, CMake, GROMACS) with GPU-aware MPI on AMD MI300X/MI355X — something not available as a single end-to-end guide elsewhere.
language: English
myst:
    html_meta:
        "author": "David Björelind, Sebastian Remander, Gerardo del Muro Gonzalez"
        "description lang=en": "Build GROMACS with HIP, UCX, and OpenMPI on AMD MI300X/MI355X — covering bare metal, Apptainer, and Docker deployments."
        "keywords": "GROMACS, AMD Instinct, MI300X, MI355X, HIP, ROCm, OpenMPI, UCX, molecular dynamics, multi-GPU, HPC, Apptainer, Docker"
        "vertical": "HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "amd_blog_authors": "David Björelind, Sebastian Remander, Gerardo del Muro Gonzalez"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# GROMACS on AMD Instinct GPUs: A Complete Build Guide

Molecular dynamics simulations power breakthroughs in drug discovery, materials science, and computational biology. [GROMACS](https://www.gromacs.org/) stands as one of the most widely used molecular dynamics engines, and pairing it with AMD's latest GPU accelerators unlocks exceptional simulation throughput. This guide walks you through installing a complete GROMACS stack with OpenMPI support on AMD MI300X and MI355X systems &mdash; whether you're deploying on bare metal or in containers.

By following this guide, you will:

- Set up AMD's HIP-enabled GROMACS for full GPU acceleration on AMD Instinct hardware
- Configure OpenMPI for efficient multi-GPU communication
- Choose between bare metal and container-based deployment approaches

Cloud providers such as Vultr and TensorWave offer MI300X infrastructure where you can deploy this stack immediately. The instructions apply equally to on-premises installations. This guide is inspired by the [InfinityHub GROMACS page](https://github.com/amd/InfinityHub-CI/tree/main/gromacs) but provides a complete installation guide with control over the full set of dependencies.

## Setup

### Hardware Requirements

This guide targets systems equipped with AMD Instinct accelerators:

- **GPU:** AMD Instinct MI300X or MI355X
- **CPU:** AMD EPYC processors (validated reference platform with NUMA topology optimized for Instinct accelerators)
- **Memory:** Sufficient system RAM for your molecular systems (workload-dependent)
- **Network:** High-bandwidth interconnect recommended for multi-node deployments

For detailed system configuration and BIOS settings, see the [AMD Instinct MI300X system optimization guide](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html).

This installation procedure has been validated on MI300X systems from both Vultr and TensorWave cloud providers.

### Software Prerequisites

Before starting the installation, ensure your system has:

- A supported Linux distribution with ROCm compatibility
- ROCm installed and configured
    - ROCm &ge; 6.0.0 for MI300X and &ge; 7.0.0 for MI355X
- Standard build tools (GCC, make, etc.)
    - This guide uses GCC/G++ 11
- Git for source code retrieval

## Bare metal installation

The bare metal installation builds each component from source, giving you full control over optimization flags and configuration options. This approach suits production deployments where performance tuning matters.

### Installation Overview

The installation proceeds through four phases:

1. **Communication libraries** &mdash; Build UCX with ROCm support for high-performance GPU-aware communication
2. **MPI implementation** &mdash; Build OpenMPI with UCX integration for multi-GPU coordination. For MPICH-based systems, see the [LUMI Supercomputer guide](https://rocm.blogs.amd.com/artificial-intelligence/gromacs-lumi-guide/README.html) as an alternative
3. **Build tools** &mdash; Install a compatible CMake version for the GROMACS build system
4. **GROMACS** &mdash; Build [GROMACS with HIP support](https://gitlab.com/gromacs/gromacs/-/tree/4947-hip-feature-enablement?ref_type=heads) targeting your GPU architecture

### Environment Setup

First, set up the installation directories and version variables. All components install under a single base directory for clean environment management:

```bash
export BASE_INSTALL="$HOME/gromacs"

UCX_VERSION="1.19.1"
OPENMPI_VERSION="5.0.8"
CMAKE_VERSION="3.28.0"
GROMACS_BRANCH="4947-hip-feature-enablement"

GCC_VERSION="gcc-11"
GXX_VERSION="g++-11"

GPU_TARGET="gfx942"
ROCM_PATH="/opt/rocm"
```

The `GPU_TARGET` variable specifies the GPU architecture. Use `gfx942` for MI300X systems and `gfx950` for MI355X systems. For other AMD Instinct systems, consult the [AMD GPU architecture documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for the appropriate target. This installation uses `gfx942`. Make sure to change the value of `GPU_TARGET` when following this guide if your system is different.

### Communication Libraries (UCX)

UCX (Unified Communication X) provides a high-performance communication framework that enables GPU-aware data transfers. Building UCX with ROCm support allows direct GPU memory access for MPI operations, reducing latency and improving multi-GPU scaling.

```bash
echo "Installing UCX ${UCX_VERSION}..."

UCX_TARBALL="ucx-${UCX_VERSION}.tar.gz"
UCX_URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${UCX_TARBALL}"
UCX_BUILD_DIR="/var/tmp/ucx-${UCX_VERSION}"

mkdir -p /var/tmp
wget -q -nc -P /var/tmp "${UCX_URL}"
tar -xzf "/var/tmp/${UCX_TARBALL}" -C /var/tmp

cd "${UCX_BUILD_DIR}"
CC=gcc CFLAGS=-Wno-error CXX=g++ ./configure \
    --prefix="${BASE_INSTALL}/ucx" \
    --with-rocm="${ROCM_PATH}" \
    --without-cuda

make -j$(nproc)
make -j$(nproc) install

rm -rf "${UCX_BUILD_DIR}" "/var/tmp/${UCX_TARBALL}"
```

After installation, update the environment to include UCX:

```bash
export CPATH="${BASE_INSTALL}/ucx/include:${CPATH}"
export LD_LIBRARY_PATH="${BASE_INSTALL}/ucx/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${BASE_INSTALL}/ucx/lib:${LIBRARY_PATH}"
export PATH="${BASE_INSTALL}/ucx/bin:${PATH}"
```

### MPI Implementation (OpenMPI)

OpenMPI provides the Message Passing Interface implementation for multi-GPU systems. Configuring OpenMPI with UCX integration enables GPU-aware collective operations.

```bash
echo "Installing OpenMPI ${OPENMPI_VERSION}..."

OPENMPI_TARBALL="openmpi-${OPENMPI_VERSION}.tar.bz2"
OPENMPI_URL="https://www.open-mpi.org/software/ompi/v$(echo ${OPENMPI_VERSION} | cut -d. -f1-2)/downloads/${OPENMPI_TARBALL}"
OPENMPI_BUILD_DIR="/var/tmp/openmpi-${OPENMPI_VERSION}"

mkdir -p /var/tmp
wget -q -nc -P /var/tmp "${OPENMPI_URL}"
tar -xjf "/var/tmp/${OPENMPI_TARBALL}" -C /var/tmp

cd "${OPENMPI_BUILD_DIR}"
CC=gcc CXX=g++ ./configure \
    --prefix="${BASE_INSTALL}/openmpi" \
    --with-rocm="${ROCM_PATH}" \
    --with-ucx="${BASE_INSTALL}/ucx" \
    --without-cuda

make -j$(nproc)
make -j$(nproc) install

rm -rf "${OPENMPI_BUILD_DIR}" "/var/tmp/${OPENMPI_TARBALL}"
```

Update the environment to include OpenMPI:

```bash
export LD_LIBRARY_PATH="${BASE_INSTALL}/openmpi/lib:${LD_LIBRARY_PATH}"
export PATH="${BASE_INSTALL}/openmpi/bin:${PATH}"
```

### Build Tools (CMake)

GROMACS requires CMake 3.18 or later for HIP support. Installing a known-compatible version ensures the build configuration works correctly:

```bash
echo "Installing CMake ${CMAKE_VERSION}..."

CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-x86_64.sh"
CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}"
CMAKE_INSTALL_DIR="${BASE_INSTALL}/cmake-${CMAKE_VERSION}"

mkdir -p /var/tmp
wget -q -nc -P /var/tmp "${CMAKE_URL}"

mkdir -p "${CMAKE_INSTALL_DIR}"
/bin/sh "/var/tmp/${CMAKE_INSTALLER}" \
    --prefix="${CMAKE_INSTALL_DIR}" \
    --skip-license

rm -rf "/var/tmp/${CMAKE_INSTALLER}"

export PATH="${CMAKE_INSTALL_DIR}/bin:${PATH}"
```

### GROMACS

With the communication stack in place, build GROMACS with HIP GPU acceleration and MPI support:

```bash
echo "Installing GROMACS (branch: ${GROMACS_BRANCH})..."

GROMACS_BUILD_DIR="/var/tmp/gromacs"

rm -rf "${GROMACS_BUILD_DIR}"

mkdir -p /var/tmp
cd /var/tmp
git clone \
    --depth=1 \
    --branch "${GROMACS_BRANCH}" \
    --recursive \
    https://gitlab.com/gromacs/gromacs.git \
    gromacs

cd "${GROMACS_BUILD_DIR}"
mkdir -p build
cd build

cmake \
    -DCMAKE_INSTALL_PREFIX="${BASE_INSTALL}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGMX_GPU=HIP \
    -DGMX_GPU_UPDATE=ON \
    -DGMX_MPI=ON \
    -DGMX_OPENMP=ON \
    -DGMX_GPU_FFT=ON \
    -DGMX_MULTI_GPU_FFT=ON \
    -DGMX_HIP_TARGET_ARCH="${GPU_TARGET}" \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DCMAKE_C_COMPILER="${GCC_VERSION}" \
    -DCMAKE_CXX_COMPILER="${GXX_VERSION}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
    "${GROMACS_BUILD_DIR}"

cmake --build . --target all -- -j$(nproc)
cmake --build . --target install -- -j$(nproc)

rm -rf "${GROMACS_BUILD_DIR}"
```

Key CMake options explained:

- `GMX_GPU=HIP` &mdash; Enables HIP GPU acceleration for AMD GPUs
- `GMX_GPU_UPDATE=ON` &mdash; Enables GPU-resident update and constraints
- `GMX_MPI=ON` &mdash; Builds with MPI support for multi-GPU scaling
- `GMX_OPENMP=ON` &mdash; Enables OpenMP for multi-threaded parallelism within each MPI rank
- `GMX_GPU_FFT=ON` &mdash; Uses GPU-accelerated FFT operations
- `GMX_MULTI_GPU_FFT=ON` &mdash; Enables multi-GPU FFT decomposition
- `GMX_HIP_TARGET_ARCH` &mdash; Specifies the target GPU architecture
- `GMX_BUILD_OWN_FFTW=ON` &mdash; Builds FFTW for optimal CPU FFT performance

### Verifying the Installation

After installation completes, verify your GROMACS configuration:

```bash
source ${BASE_INSTALL}/bin/GMXRC
gmx_mpi --version
```

The output should show HIP support enabled and list your GPU architecture.

## Container Installation

When properly configured, containerized GROMACS delivers performance comparable to bare metal installations while providing additional deployment flexibility. Consider container deployment when:

- You need reproducible environments across multiple systems
- Building from source is impractical for your use case
- You want rapid deployment without compilation time
- Your cluster uses container orchestration (Kubernetes, Slurm with container support)

### Container Options

You can deploy GROMACS using either Apptainer (formerly Singularity) or Docker, depending on your infrastructure:

- **Apptainer** &mdash; Preferred for HPC environments; runs without root privileges
- **Docker** &mdash; Suitable for cloud deployments and development environments

For container builds, the scripts install GROMACS in
```bash
/opt/gromacs
```
instead of installing in your home directory.

The sections below provide complete install scripts for Docker and Apptainer. These scripts are not discussed in detail because the containerized installation procedure is largely the same as the bare metal approach, with some initial differences. Both container options use the base image `rocm/dev-ubuntu-24.04:latest`.

### Apptainer install script

<details>
<summary>Apptainer build & install instructions</summary>

GROMACS Apptainer .def-file:
```bash
Bootstrap: docker
From: rocm/dev-ubuntu-24.04:latest

%labels
    Description "UCX + OpenMPI + GROMACS build"

%environment
    export PATH=/opt/gromacs/bin:/opt/gromacs/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/gromacs/lib:/opt/gromacs/openmpi/lib:/opt/gromacs/ucx/lib:$LD_LIBRARY_PATH
    export GMX_ENABLE_DIRECT_GPU_COMM=1
    export GMX_FORCE_GPU_AWARE_MPI=1
    export ROC_ACTIVE_WAIT_TIMEOUT=0
    export AMD_DIRECT_DISPATCH=1
    export OMPI_MCA_plm=isolated
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    export UCX_TLS=sm,rocm,self
    export UCX_MEMTYPE_CACHE=n
    export UCX_RNDV_THRESH=16384
    export ROCM_PATH=/opt/rocm

%post
    set -ex

    # Version variables
    export BASE_INSTALL="/opt/gromacs"

    export UCX_VERSION="1.19.1"
    export OPENMPI_VERSION="5.0.8"
    export CMAKE_VERSION="3.28.0"
    export GROMACS_BRANCH="4947-hip-feature-enablement"

    export GCC_VERSION="gcc-11"
    export GXX_VERSION="g++-11"

    export GPU_TARGET="gfx942"
    export ROCM_PATH="/opt/rocm"

    # -------- Install build and runtime dependencies ----------
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ${GCC_VERSION} ${GXX_VERSION} \
        git \
        wget \
        bzip2 \
        pkg-config \
        libevent-dev \
        ca-certificates \
        libfftw3-dev \
        libhwloc-dev \
        libnuma-dev \
        libssl-dev \
        rocm-hip-sdk && \
    update-ca-certificates

    # -------- Build UCX ----------
    UCX_TARBALL="ucx-${UCX_VERSION}.tar.gz"
    UCX_URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${UCX_TARBALL}"
    UCX_BUILD_DIR="/var/tmp/ucx-${UCX_VERSION}"

    wget -q -nc -P /var/tmp "${UCX_URL}"
    tar -xzf "/var/tmp/${UCX_TARBALL}" -C /var/tmp

    cd "${UCX_BUILD_DIR}"
    CC=gcc CFLAGS=-Wno-error CXX=g++ ./configure \
        --prefix="${BASE_INSTALL}/ucx" \
        --with-rocm="${ROCM_PATH}" \
        --without-cuda

    make -j$(nproc)
    make -j$(nproc) install

    export CPATH="${BASE_INSTALL}/ucx/include:${CPATH}"
    export LD_LIBRARY_PATH="${BASE_INSTALL}/ucx/lib:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${BASE_INSTALL}/ucx/lib:${LIBRARY_PATH}"
    export PATH="${BASE_INSTALL}/ucx/bin:${PATH}"

    # -------- Build OpenMPI ----------
    OPENMPI_TARBALL="openmpi-${OPENMPI_VERSION}.tar.bz2"
    OPENMPI_URL="https://www.open-mpi.org/software/ompi/v$(echo ${OPENMPI_VERSION} | cut -d. -f1-2)/downloads/${OPENMPI_TARBALL}"
    OPENMPI_BUILD_DIR="/var/tmp/openmpi-${OPENMPI_VERSION}"

    wget -q -nc -P /var/tmp "${OPENMPI_URL}"
    tar -xjf "/var/tmp/${OPENMPI_TARBALL}" -C /var/tmp

    cd "${OPENMPI_BUILD_DIR}"
    CC=gcc CXX=g++ ./configure \
        --prefix="${BASE_INSTALL}/openmpi" \
        --with-rocm="${ROCM_PATH}" \
        --with-ucx="${BASE_INSTALL}/ucx" \
        --without-cuda

    make -j$(nproc)
    make -j$(nproc) install

    export LD_LIBRARY_PATH="${BASE_INSTALL}/openmpi/lib:${LD_LIBRARY_PATH}"
    export PATH="${BASE_INSTALL}/openmpi/bin:${PATH}"

    # -------- Install CMake ----------
    CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-x86_64.sh"
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}"
    CMAKE_INSTALL_DIR="${BASE_INSTALL}/cmake-${CMAKE_VERSION}"

    wget -q -nc -P /var/tmp "${CMAKE_URL}"
    mkdir -p "${CMAKE_INSTALL_DIR}"

    /bin/sh "/var/tmp/${CMAKE_INSTALLER}" \
        --prefix="${CMAKE_INSTALL_DIR}" \
        --skip-license

    export PATH="${CMAKE_INSTALL_DIR}/bin:${PATH}"

    # -------- Build GROMACS ----------
    GROMACS_BUILD_DIR="/var/tmp/gromacs"

    rm -rf "${GROMACS_BUILD_DIR}"
    cd /var/tmp
    git clone \
        --depth=1 \
        --branch "${GROMACS_BRANCH}" \
        --recursive \
        https://gitlab.com/gromacs/gromacs.git \
        gromacs

    cd "${GROMACS_BUILD_DIR}"
    mkdir -p build
    cd build

    cmake \
        -DCMAKE_INSTALL_PREFIX="${BASE_INSTALL}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGMX_GPU=HIP \
        -DGMX_GPU_UPDATE=ON \
        -DGMX_MPI=ON \
        -DGMX_OPENMP=ON \
        -DGMX_GPU_FFT=ON \
        -DGMX_MULTI_GPU_FFT=ON \
        -DGMX_HIP_TARGET_ARCH="${GPU_TARGET}" \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DCMAKE_C_COMPILER="${GCC_VERSION}" \
        -DCMAKE_CXX_COMPILER="${GXX_VERSION}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
        "${GROMACS_BUILD_DIR}"

    cmake --build . --target all -- -j$(nproc)
    cmake --build . --target install -- -j$(nproc)

    # -------- Cleanup ----------
    rm -rf /var/tmp/ucx-* /var/tmp/openmpi-* /var/tmp/gromacs /var/tmp/cmake-*
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%runscript
    exec /bin/bash
```
Build Apptainer image:
```bash
apptainer build --fakeroot gromacs_openmpi.sif gromacs_openmpi.def
```
You need the `--fakeroot` flag because the `%post` section runs `apt-get install`, which needs root-level permissions to install packages. This flag allows unprivileged users to build containers without actual root access.

</details>

### Docker install script

<details>
<summary>Docker build & install instructions</summary>

GROMACS Dockerfile:
```Dockerfile
# -------- STAGE 1: Build UCX + OpenMPI ----------
FROM rocm/dev-ubuntu-24.04:latest AS mpi-builder

ARG UCX_VERSION="1.19.1"
ARG OPENMPI_VERSION="5.0.8"
ARG ROCM_PATH="/opt/rocm"
ARG BASE_INSTALL="/opt/gromacs"

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    bzip2 \
    pkg-config \
    libevent-dev \
    ca-certificates && \
    update-ca-certificates

# Install UCX
RUN UCX_TARBALL="ucx-${UCX_VERSION}.tar.gz" && \
    UCX_URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${UCX_TARBALL}" && \
    UCX_BUILD_DIR="/var/tmp/ucx-${UCX_VERSION}" && \
    wget -q -nc -P /var/tmp "${UCX_URL}" && \
    tar -xzf "/var/tmp/${UCX_TARBALL}" -C /var/tmp && \
    cd "${UCX_BUILD_DIR}" && \
    CC=gcc CFLAGS=-Wno-error CXX=g++ ./configure \
        --prefix="${BASE_INSTALL}/ucx" \
        --with-rocm="${ROCM_PATH}" \
        --without-cuda && \
    make -j$(nproc) && \
    make -j$(nproc) install

# Install OpenMPI
RUN OPENMPI_TARBALL="openmpi-${OPENMPI_VERSION}.tar.bz2" && \
    OPENMPI_URL="https://www.open-mpi.org/software/ompi/v$(echo ${OPENMPI_VERSION} | cut -d. -f1-2)/downloads/${OPENMPI_TARBALL}" && \
    OPENMPI_BUILD_DIR="/var/tmp/openmpi-${OPENMPI_VERSION}" && \
    wget -q -nc -P /var/tmp "${OPENMPI_URL}" && \
    tar -xjf "/var/tmp/${OPENMPI_TARBALL}" -C /var/tmp && \
    cd "${OPENMPI_BUILD_DIR}" && \
    CC=gcc CXX=g++ ./configure \
        --prefix="${BASE_INSTALL}/openmpi" \
        --with-rocm="${ROCM_PATH}" \
        --with-ucx="${BASE_INSTALL}/ucx" \
        --without-cuda && \
    make -j$(nproc) && \
    make -j$(nproc) install


# -------- STAGE 2: Build GROMACS ----------
FROM rocm/dev-ubuntu-24.04:latest AS gromacs-builder

ARG CMAKE_VERSION="3.28.0"
ARG GROMACS_BRANCH="4947-hip-feature-enablement"
ARG GCC_VERSION="gcc-11"
ARG GXX_VERSION="g++-11"
ARG GPU_TARGET="gfx942"
ARG ROCM_PATH="/opt/rocm"
ARG BASE_INSTALL="/opt/gromacs"

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    ${GCC_VERSION} ${GXX_VERSION} \
    git \
    libfftw3-dev \
    libhwloc-dev \
    libnuma-dev \
    pkg-config \
    ca-certificates \
    libssl-dev \
    libevent-dev \
    rocm-hip-sdk && \
    update-ca-certificates

# Install CMake via pre-built installer
RUN CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-x86_64.sh" && \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}" && \
    CMAKE_INSTALL_DIR="${BASE_INSTALL}/cmake-${CMAKE_VERSION}" && \
    wget -q -nc -P /var/tmp "${CMAKE_URL}" && \
    mkdir -p "${CMAKE_INSTALL_DIR}" && \
    /bin/sh "/var/tmp/${CMAKE_INSTALLER}" \
        --prefix="${CMAKE_INSTALL_DIR}" \
        --skip-license

ENV PATH="${BASE_INSTALL}/cmake-${CMAKE_VERSION}/bin:${PATH}"

COPY --from=mpi-builder ${BASE_INSTALL}/openmpi ${BASE_INSTALL}/openmpi
COPY --from=mpi-builder ${BASE_INSTALL}/ucx     ${BASE_INSTALL}/ucx

# Set environment variables for UCX and MPI paths
ENV CPATH="${BASE_INSTALL}/ucx/include:${CPATH}" \
    LD_LIBRARY_PATH="${BASE_INSTALL}/openmpi/lib:${BASE_INSTALL}/ucx/lib:${LD_LIBRARY_PATH}" \
    LIBRARY_PATH="${BASE_INSTALL}/ucx/lib:${LIBRARY_PATH}" \
    PATH="${BASE_INSTALL}/openmpi/bin:${BASE_INSTALL}/ucx/bin:${PATH}"

# Clone and build GROMACS
RUN GROMACS_BUILD_DIR="/var/tmp/gromacs" && \
    rm -rf "${GROMACS_BUILD_DIR}" && \
    cd /var/tmp && \
    git clone \
        --depth=1 \
        --branch "${GROMACS_BRANCH}" \
        --recursive \
        https://gitlab.com/gromacs/gromacs.git \
        gromacs && \
    cd "${GROMACS_BUILD_DIR}" && \
    mkdir -p build && \
    cd build && \
    cmake \
        -DCMAKE_INSTALL_PREFIX="${BASE_INSTALL}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGMX_GPU=HIP \
        -DGMX_GPU_UPDATE=ON \
        -DGMX_MPI=ON \
        -DGMX_OPENMP=ON \
        -DGMX_GPU_FFT=ON \
        -DGMX_MULTI_GPU_FFT=ON \
        -DGMX_HIP_TARGET_ARCH="${GPU_TARGET}" \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DCMAKE_C_COMPILER="${GCC_VERSION}" \
        -DCMAKE_CXX_COMPILER="${GXX_VERSION}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
        "${GROMACS_BUILD_DIR}" && \
    cmake --build . --target all -- -j$(nproc) && \
    cmake --build . --target install -- -j$(nproc)


# -------- STAGE 3: Final Runtime Image ----------
FROM rocm/dev-ubuntu-24.04:latest

ARG BASE_INSTALL="/opt/gromacs"

RUN apt-get update && apt-get install -y \
    libnuma1 \
    libhwloc15 \
    libfftw3-single3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=gromacs-builder ${BASE_INSTALL}/bin   ${BASE_INSTALL}/bin
COPY --from=gromacs-builder ${BASE_INSTALL}/lib   ${BASE_INSTALL}/lib
COPY --from=gromacs-builder ${BASE_INSTALL}/share ${BASE_INSTALL}/share
COPY --from=mpi-builder     ${BASE_INSTALL}/openmpi ${BASE_INSTALL}/openmpi
COPY --from=mpi-builder     ${BASE_INSTALL}/ucx     ${BASE_INSTALL}/ucx

# Copy required shared libraries
COPY --from=gromacs-builder \
        /usr/lib/x86_64-linux-gnu/libevent_core-2.1.so.7 \
        /usr/lib/x86_64-linux-gnu/
COPY --from=gromacs-builder \
        /usr/lib/x86_64-linux-gnu/libevent_pthreads-2.1.so.7 \
        /usr/lib/x86_64-linux-gnu/

# Set all environment variables for performance and container compatibility
ENV PATH="${BASE_INSTALL}/bin:${BASE_INSTALL}/openmpi/bin:${PATH}" \
    LD_LIBRARY_PATH="${BASE_INSTALL}/lib:${BASE_INSTALL}/openmpi/lib:${BASE_INSTALL}/ucx/lib:${LD_LIBRARY_PATH}" \
    GMX_ENABLE_DIRECT_GPU_COMM=1 \
    GMX_FORCE_GPU_AWARE_MPI=1 \
    ROC_ACTIVE_WAIT_TIMEOUT=0 \
    AMD_DIRECT_DISPATCH=1 \
    # MPI container fixes
    OMPI_MCA_plm=isolated \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    # UCX performance tuning
    UCX_TLS=sm,rocm,self \
    UCX_MEMTYPE_CACHE=n \
    UCX_RNDV_THRESH=16384

CMD ["/bin/bash"]
```

Build Docker image:
```bash
docker build -f Dockerfile -t gromacs-openmpi .
```

</details>


## Summary

You now have a working GROMACS installation with OpenMPI support on AMD MI300X or MI355X GPUs. This stack enables GPU-accelerated molecular dynamics simulations with efficient multi-GPU scaling. Building each dependency from source with ROCm support is essential for optimal performance. Without GPU-aware communication libraries, every inter-GPU data exchange must be staged through host memory before being transferred to the destination GPU. UCX eliminates this overhead by enabling direct GPU memory registration and RDMA transfers, keeping data on the GPU throughout the communication path. OpenMPI then builds on UCX to implement GPU-aware MPI collectives, so operations like halo exchanges and PME decomposition scale efficiently across multiple GPUs without redundant memory copies.

Key points from this guide:

- **Bare metal installation** provides maximum control and performance tuning opportunities
- **Container deployment** offers reproducibility and simplified management
- **UCX and OpenMPI** enable GPU-aware communication for multi-GPU workloads
- **HIP support** in GROMACS leverages AMD GPU compute capabilities

For production workloads, consider profiling your specific simulations to optimize settings for your molecular systems. The [GPU Partitioning for GROMACS](https://rocm.blogs.amd.com/artificial-intelligence/gpu-partitioning-workloads/README.html) blog demonstrates how compute partitioning can further increase GROMACS throughput on AMD Instinct GPUs.

## Additional Resources

- [GPU Partitioning for GROMACS](https://rocm.blogs.amd.com/artificial-intelligence/gpu-partitioning-workloads/README.html) &mdash; Applying compute partitioning for increased GROMACS performance
- [GROMACS Performance on AMD Instinct MI355X](https://rocm.blogs.amd.com/artificial-intelligence/mi355-gromacs-benchmarks/README.html) &mdash; GROMACS MI355X benchmark performance across different configurations
- [GROMACS Documentation](https://manual.gromacs.org/) &mdash; Official manual and installation guides
- [AMD ROCm Documentation](https://rocm.docs.amd.com/) &mdash; ROCm installation and optimization
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/) &mdash; HIP development and optimization
- [OpenMPI Documentation](https://www.open-mpi.org/doc/) &mdash; MPI configuration and tuning
- [UCX Documentation](https://openucx.readthedocs.io/) &mdash; Communication framework reference
- [Apptainer Documentation](https://apptainer.org/docs/user/latest/) &mdash; Container building and deployment for HPC

## Disclaimers

- Performance results depend on system configuration, molecular system size, and simulation parameters
- Some features described may be in active development; check GROMACS release notes for current status
- Cloud provider configurations may vary; consult provider documentation for infrastructure-specific details

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
