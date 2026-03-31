---
blogpost: true
blog_title: "Installing AMD HIP-Enabled GROMACS on HPC Systems: A LUMI Supercomputer Case Study"
date: 12 Jan 2026
author: 'Sebastian Remander, Mittul Singh, Paul Bauer'
thumbnail: 'gromacs-ai-driven-drug-discovery.png'
tags: Scientific Computing, HPC, Installation, Performance
category: Applications & models
target_audience: Computational scientists in drug discovery, HPC system administrators
key_value_propositions: Enabling users to install AMD HIP-enabled GROMACS on LUMI and similar HPC and AI Factory environments, with optimized performance and reproducible build instructions.
language: English
myst:
    html_meta:
        "author": "Sebastian Remander, Mittul Singh, Paul Bauer"
        "description lang=en": "Installing AMD HIP-Enabled GROMACS on HPC Systems: A LUMI Supercomputer Case Study"
        "keywords": "GROMACS, HIP, AMD, MI250X, LUMI, HPC, molecular dynamics, GPU computing"
        "vertical": "HPC, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "EPYC Server Processors, Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Design, Simulation & Modeling"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "amd_blog_authors": "Sebastian Remander, Mittul Singh, Paul Bauer"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

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

# Installing AMD HIP-Enabled GROMACS on HPC Systems: A LUMI Supercomputer Case Study

Running molecular dynamics (MD) simulations efficiently is critical for accelerating scientific discovery in many life science use cases, e.g., drug discovery. [GROMACS](https://www.gromacs.org/) is a widely used, GPU-accelerated molecular dynamics engine powering many life science workflows, but its performance can vary significantly depending on the installation method and hardware configuration. For broader context on GROMACS applications in drug design, see [recent research on GROMACS in cloud environments for alchemical drug design](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006219/).

To fully exploit HPC capabilities and include the latest GROMACS features, in this blog post we walk through installing a custom, [AMD-modified, bare metal version of GROMACS](https://manual.gromacs.org/2025.4/install-guide/index.html#hip) on the [LUMI](https://www.lumi-supercomputer.eu/about-lumi/) supercomputer. By tailoring the build to LUMI's architecture and optimizing the installation through benchmarking, you can maximize simulation throughput. Even though we work through steps on LUMI, these steps serve as a playbook for replicating the installation on other similar HPC systems and AI factories. Throughout this guide, we highlight how to adapt LUMI-specific configurations for your system. For further context on large-scale GROMACS deployments in cloud HPC environments, see the [AWS blog on cost-effective GROMACS simulations](https://aws.amazon.com/blogs/hpc/large-scale-cost-effective-gromacs-simulations-using-the-cyclone-solution-from-aws/).

## MI250X Base Recipe

The installation recipe presented in this blog post is based on the [AMD InfinityHub GROMACS baremetal recipe](https://github.com/amd/InfinityHub-CI/tree/main/gromacs/baremetal), adapted for the LUMI supercomputer environment. The InfinityHub provides validated build instructions for AMD GPUs that we extend here with LUMI-specific optimizations.

### Setup

[LUMI](https://en.wikipedia.org/wiki/LUMI) is a state-of-the-art supercomputer featuring AMD EPYC CPUs and AMD Instinct MI250X GPUs. Each LUMI-G compute node contains 4 MI250X devices, and each MI250X device is partitioned into 2 Graphics Compute Dies (GCDs), providing 8 GCDs per node. Note that SLURM scripts on LUMI use `--gpus-per-node` to specify the number of GCDs, not physical MI250X devices. For example, `--gpus-per-node=8` allocates all 8 GCDs (4 MI250X devices) on a node.

To achieve optimal performance, you will install the HIP-enabled GROMACS from source, tailoring the build to LUMI's hardware and software environment. This process involves selecting appropriate compiler toolchains and enabling hardware-specific optimizations. In addition, we incorporated the runtime environment flags and GROMACS-specific optimizations from the [CSC GROMACS LUMI documentation](https://docs.csc.fi/apps/gromacs/#full-gpu-node-batch-script) into the benchmark scripts presented in this blog post.

### Step-by-Step Walkthrough: Installing Custom GROMACS on LUMI

#### Prerequisites

This installation guide applies to any HPC system or workstation equipped with AMD Instinct GPUs. We demonstrate the installation on LUMI, but the build process adapts to other Linux-based HPC environments with minimal modifications. The steps below assume you have SSH access to your system and the necessary permissions to install software in your project or home directory.

#### 0. Set Project Number as an Environment Variable

Before proceeding, set your HPC allocation identifier as an environment variable (on LUMI this is the project number; on other systems it may be a project ID, account number, or similar). It will be used in the commands throughout the installation process:

```bash
export PROJECT_NUMBER=<your_project_number>
```

#### 1. Create the Installation Directory

Before building, create a directory for the GROMACS installation in your project space:

```bash
mkdir -p /project/$PROJECT_NUMBER/software/gromacs/4947-hip-feature-enablement_v1
```

```{note}
The directory name `4947-hip-feature-enablement` corresponds to the AMD-specific GROMACS branch name to distinguish it from the upstream GROMACS repository. The `_v1` suffix indicates the version of your installation, which you can increment if you create multiple installations.
```

To share the installation with your team, you can optionally set group ownership and permissions:

```bash
# (Optional) Set group ownership to your project group
chgrp -R $PROJECT_NUMBER /project/$PROJECT_NUMBER/software

# (Optional) Allow group members to read and execute
chmod -R g+rX /project/$PROJECT_NUMBER/software

# (Optional) Allow group members also to write i.e. install new software
# chmod -R g+rwx /project/$PROJECT_NUMBER/software
```

#### 2. Allocate a Compute Node for Building

Request a GPU node interactively. This ensures you have the correct hardware and environment available during the build:

```bash
salloc --partition=small-g --account=$PROJECT_NUMBER --time=02:00:00 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=7
```

```{tip}
At this point, you can either continue the step by step installation process interactively or jump to the [Full Installation Script](#full-installation-script) section where a single install script is provided.
```

Once the `salloc` allocation is granted, shell into the compute node to carry on with the interactive installation:

```bash
srun --pty $SHELL
```

#### 3. Prepare the Optimal Build Environment on LUMI

The goal is to use the [Cray Programming Environment](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/) (PrgEnv-cray), which provides compiler wrappers (cc and CC) that automatically link against the correct MPI, GTL, and other system libraries. In addition, we load all other necessary library modules for building software on LUMI's AMD hardware.

The key required modules are:

* LUMI & partition/G: Standard setup for the LUMI-G partition.
* PrgEnv-cray: Switches the environment to use the Cray Compiling Environment (CCE).
* rocm/x.y.z: Provides the HIP compiler and headers needed for GPU support. You should use a recent, stable version available on the system.
* craype-accel-amd-gfx90a: Critical module for GPU-aware MPI.
* cray-fftw: Provides libraries which are required for building optimal CPU FFT support in GROMACS.
* buildtools: Exposes `cmake`, the build system generator used by GROMACS.

```bash
module load LUMI/25.03 partition/G buildtools/25.03
module load PrgEnv-cray
module load rocm/6.3.4
module load craype-accel-amd-gfx90a
module load cray-fftw/3.3.10.11
```

```{tip}
**On other Cray systems:** Module names may differ (e.g., `PrgEnv-amd` instead of `PrgEnv-cray`). Use `module avail rocm` and `module avail fftw` to find equivalent modules on your system.
```

#### 4. Download the AMD HIP GROMACS Source Code

Clone the [HIP-enabled GROMACS branch](https://gitlab.com/gromacs/gromacs/-/tree/4947-hip-feature-enablement) and set up the build directory:

```{important}
This branch contains AMD-specific optimizations and will continue to be maintained. It currently tracks GROMACS 2025.4 and will be updated to track newer releases (e.g., 2026) as they become available.
```

```{tip}
AMD also maintains branches specific to individual patch releases if you require a particular GROMACS version for compatibility or reproducibility purposes. For example, use the [2025 HIP-enabled GROMACS branch](https://github.com/ROCm/Gromacs/tree/4947-hip-feature-enablement-2025.4) if you want to reproduce the benchmark numbers presented in this blogpost.
```

```bash
git clone -b 4947-hip-feature-enablement https://gitlab.com/gromacs/gromacs.git gromacs
cd gromacs && mkdir build && cd build
```

#### 5. Configure the Build with CMake

Configure GROMACS with CMake, enabling MPI and HIP support together with the GTL library linking, and set the installation prefix to your chosen directory.

Specifically, the `MPI_CFLAGS` and `MPI_LDFLAGS` flags force the linking of the Cray MPI and GTL libraries, which are optimized for LUMI's hardware and required for multi GPU support.

```{tip}
Make sure to adjust the paths, i.e., the version numbers in `MPI_CFLAGS` and `MPI_LDFLAGS` if your Cray MPI installation is located elsewhere. The paths provided here are based on the current Cray PE version available on LUMI. In addition, on systems where `ROCM_PATH` is not properly set by the ROCm module, you may need to export it manually (e.g., `export ROCM_PATH=/opt/rocm`).
```

```bash
export MPI_CFLAGS="-I/opt/cray/pe/mpich/8.1.33/ofi/cray/20.0/include/"
export MPI_LDFLAGS="-L/opt/cray/pe/mpich/8.1.33/ofi/cray/20.0/lib -lmpi -L/opt/cray/pe/mpich/8.1.33/gtl/lib -lmpi_gtl_hsa"

cmake .. \
    -DCMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang \
    -DCMAKE_C_FLAGS="${MPI_CFLAGS}" \
    -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++  \
    -DCMAKE_EXE_LINKER_FLAGS="${MPI_LDFLAGS}" \
    -DCMAKE_HIP_COMPILER=${ROCM_PATH}/bin/amdclang++ \
    -DCMAKE_INSTALL_PREFIX=/project/$PROJECT_NUMBER/software/gromacs/4947-hip-feature-enablement_v1 \
    -DCMAKE_SHARED_LINKER_FLAGS="${MPI_LDFLAGS}" \
    -DGMX_GPU=HIP \
    -DGMX_HIP_TARGET_ARCH=gfx90a \
    -DGMX_MPI=ON \
    -DGMX_THREAD_MPI=OFF
```

```{tip}
The GPU architecture target is `gfx90a` for MI250X/MI210 and `gfx942` for MI300X/MI300A.
```

#### 6. Compile and Install

Build and install GROMACS using all available CPU cores:

```bash
make -j$(nproc)
make install
```

#### 7. Verify the Installation

After the installation completes, verify that GROMACS is correctly installed and accessible:

```bash
source /project/$PROJECT_NUMBER/software/gromacs/4947-hip-feature-enablement_v1/bin/GMXRC
gmx_mpi --version
```

You should see output indicating the GROMACS version and that it was built with HIP support. Specifically, ensure that the following lines appear in the output:

```text
MPI library:         MPI
MPI library version: MPI VERSION    : CRAY MPICH version 8.1.33 (ANL base 3.4a2)
OpenMP support:      enabled (GMX_OPENMP_MAX_THREADS = 128)
GPU support:         HIP
```

#### Full Installation Script

For convenience, to automate steps 3–7, create a file named `install_gromacs_lumi.sh` with the contents below and run a single command after allocating the compute node:

```bash
srun install_gromacs_lumi.sh
```

```{note}
If you already completed some of the manual installation steps, running the full installation script may crash with an error about already existing installation directory. Please either remove it or modify the `INSTALL_VERSION` variable in the script to avoid overwriting your previous installation.
```

The script below reconciles all the installation steps together. I.e., it will clone the custom GROMACS branch, configure it with the optimal flags for LUMI-G, and install GROMACS to a persistent location in your project directory:

```bash
#!/bin/bash

set -eo pipefail

GROMACS_BRANCH="4947-hip-feature-enablement"
INSTALL_VERSION="v1"
INSTALL_PREFIX="/project/${PROJECT_NUMBER}/software/gromacs/${GROMACS_BRANCH}_${INSTALL_VERSION}"
BUILD_DIR="/scratch/${PROJECT_NUMBER}/$USER"

if [ -d "${INSTALL_PREFIX}" ]; then
    echo "Error: Installation directory ${INSTALL_PREFIX} already exists. Please choose a different version."
    exit 1
fi

module load LUMI/25.03 partition/G buildtools/25.03
module load PrgEnv-cray
module load rocm/6.3.4
module load craype-accel-amd-gfx90a
module load cray-fftw/3.3.10.11

echo "ROCM path:      ${ROCM_PATH}"
echo "MPICH path:     ${MPICH_DIR}"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
rm -rf gromacs_${INSTALL_VERSION}  # Clean up previous attempts
git clone -b ${GROMACS_BRANCH} https://gitlab.com/gromacs/gromacs.git gromacs_${INSTALL_VERSION}
cd gromacs_${INSTALL_VERSION}
mkdir build && cd build

export MPI_CFLAGS="-I/opt/cray/pe/mpich/8.1.33/ofi/cray/20.0/include/"
export MPI_LDFLAGS="-L/opt/cray/pe/mpich/8.1.33/ofi/cray/20.0/lib -lmpi -L/opt/cray/pe/mpich/8.1.33/gtl/lib -lmpi_gtl_hsa"

cmake .. \
    -DCMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang \
    -DCMAKE_C_FLAGS="${MPI_CFLAGS}" \
    -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++  \
    -DCMAKE_EXE_LINKER_FLAGS="${MPI_LDFLAGS}" \
    -DCMAKE_HIP_COMPILER=${ROCM_PATH}/bin/amdclang++ \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_SHARED_LINKER_FLAGS="${MPI_LDFLAGS}" \
    -DGMX_GPU=HIP \
    -DGMX_HIP_TARGET_ARCH=gfx90a \
    -DGMX_MPI=ON \
    -DGMX_THREAD_MPI=OFF

make -j$(nproc)
make install

# Verify installation
source ${INSTALL_PREFIX}/bin/GMXRC
gmx_mpi --version

echo "Build complete! GROMACS is now installed in: ${INSTALL_PREFIX}"
```

```{note}
When adapting the installation script to another system, update the module loads, installation path, MPI paths, and GPU target architecture to match your environment.
```

### Ablation Study

To validate our setup and configuration choices, we performed an ablation study as a sanity check. By systematically disabling individual optimizations, we confirm that each component of our build recipe contributes positively to performance.

We benchmarked two datasets, ADH and STMV, across single-GCD and 8-GCD runs to assess how specific build and runtime choices impact throughput (nanoseconds per day, ns/day).

In the tables below, Baseline refers to the GROMACS installation presented in this blog post, i.e., the HIP-enabled GROMACS built with the recommended flags and Cray Programming Environment (CrayPE). We tested variants that alter the GPU FFT library (using rocFFT instead of vkfft), omit the CrayPE AMD acceleration module (No Craype) at compile time, use OpenMP-only threading without MPI-GPU offload (OMP Only), and remove all optimization flags (No Flags) at runtime. We averaged each configuration over 6 benchmark runs and ran all experiments with 7 OpenMP threads per MPI rank.

Table 1: ADH ablation results
| Configuration | 8 GCD (ns/day) | Single GCD (ns/day) |
|---|---:|---:|
| Baseline | 438 | 290 |
| rocFFT   | 413.6 | 283.6 |
| No Craype | 428.3 | 286.6 |
| OMP Only | 278.0 | 288.5 |
| No Flags | 27.9 | 211.7 |

Table 2: STMV ablation results
| Configuration | 8 GCD (ns/day) | Single GCD (ns/day) |
|---|---:|---:|
| Baseline | 99.2 | 17.6 |
| rocFFT   | 97.5 | 17.4 |
| No Craype | 98.6 | 17.0 |
| OMP Only | 38.0 | 17.0 |
| No Flags | 6.2 | 15.7 |

The ablation results highlight the relative importance of each optimization:

* **Runtime flags (No Flags)**: The largest performance contributor. Disabling GPU-aware MPI flags causes up to 16x slowdown for multi-GCD runs, as data must stage through the CPU instead of transferring directly between GPUs.
* **MPI-GPU offload (OMP Only)**: Critical for multi-GPU scaling (36-62% slower at 8 GCDs), but minimal impact on single-GCD runs where no inter-GPU communication is needed.
* **VkFFT vs rocFFT** and **CrayPE acceleration (No Craype)**: The observed differences (1-6%) fall within typical benchmark run-to-run variation and are not statistically significant. Either FFT library and build configuration will deliver comparable performance.

#### InfinityHub Recipe on LUMI

We also evaluated the InfinityHub recipes running out-of-the-box on LUMI to understand what LUMI-specific adaptations are necessary. While the InfinityHub recipes are designed to be generalizable across AMD GPU systems, they required adjustments to achieve optimal performance on LUMI due to LUMI-specific hardware and software characteristics.

The "vs. Baseline" column in the tables below shows performance relative to the corresponding Baseline values from Tables 1 and 2. For ADH, single-GCD results are compared against 290 ns/day and 8-GCD results against 438 ns/day. For STMV, single-GCD results are compared against 17.6 ns/day and 8-GCD results against 99.2 ns/day.

Table 3: ADH InfinityHub recipe results on LUMI compared to the appropriate baseline presented in Table 1
| GPUs | GCDs | MPI Ranks | Threads (ntomp) | Performance (ns/day) | vs. Baseline |
|---|---|---|---|---:|---:|
| 1 | 1 | 1 | 64 | 298 | 1.03x |
| 4 | 8 | 8 | 8 | 239 | 0.55x |

Table 4: STMV InfinityHub recipe results on LUMI compared to the appropriate baseline presented in Table 2.
| GPUs | GCDs | MPI Ranks | Threads (ntomp) | Performance (ns/day) | vs. Baseline |
|---|---|---:|---:|---:|---:|
| 1 | 1 | 1 | 64 | 17.5 | 0.99x |
| 4 | 8 | 8 | 8 | 37.6 | 0.38x |

#### LUMI-Specific Optimizations

The InfinityHub results in Tables 3 & 4 reveal important LUMI-specific considerations:

1. **CPU Core Availability**: LUMI reserves some CPU cores for system operations, leaving 56 cores available per node rather than the full 64. The InfinityHub recipes use 64 OpenMP threads for single-GCD runs, which slightly improves the single GCD results for ADH due to the baseline being run only with 7 OpenMP threads. This illustrates the importance of selecting thread counts that match the system's actual available core count. However, exceeding this limit is not optimal either, as oversubscription causes contention for CPU resources and increased context switching overhead.

```{tip}
Our optimized recipe (presented later in the [ADH Benchmarks section](#maximizing-gpu-utilization-with-small-systems-adh-benchmarks-with-multidir)) uses appropriate thread counts per MPI rank based on LUMI's actual core availability.
```

2. **MPICH Environment Flags**: LUMI uses MPICH for MPI communication, requiring specific environment variables for optimal multi-GPU performance. The InfinityHub recipes do not document these MPICH-specific flags (`MPICH_GPU_SUPPORT_ENABLED`, `GMX_ENABLE_DIRECT_GPU_COMM`, `GMX_FORCE_GPU_AWARE_MPI`, etc.), which are essential for efficient GPU-aware MPI on LUMI, as per the [CSC LUMI documentation](https://docs.csc.fi/apps/gromacs/#lumi).

3. **Thread Pinning and Affinity**: Proper CPU affinity masks and thread pinning are crucial on LUMI's architecture. The optimized recipe includes detailed CPU binding configurations tailored to LUMI's topology.

These adaptations demonstrate that while the InfinityHub recipes provide an excellent foundation, system-specific tuning is essential for achieving peak performance on HPC systems like LUMI.

#### Results Discussion

The ablation study reveals that the Baseline installation with LUMI-specific optimizations is consistently the best across all datasets, although margins are small in some of the benchmark runs (e.g., in STMV single GCD results presented in Table 2). Nevertheless, the Baseline installation delivers the highest performance in all cases, validating our custom bare metal build and LUMI-specific configuration choices.

Our results show that multi-GPU performance depends heavily on runtime environment variables. 8-GCD runs are highly sensitive to threading, MPI, and GPU runtime choices: simulation throughput (nanoseconds per day) drops significantly when you disable these settings.

Scaling varies by system size and simulation type:

* ADH (Table 1): Weak scaling to 8 GCDs (290 → 438 ns/day, ~1.5x). The system likely does not saturate multiple GPUs efficiently at this size. For such smaller systems with underutilized GPU resources, the [ADH Benchmarks section](#maximizing-gpu-utilization-with-small-systems-adh-benchmarks-with-multidir) demonstrates how ensemble approaches using `-multidir` can dramatically improve aggregate throughput by running multiple independent trajectories concurrently.
* STMV (Table 2): Strong scaling (17.6 → 99.2 ns/day, ~5.6x). Large system benefits from multi-GPU parallelism. The [STMV Benchmarks section](#stmv-benchmarks) further characterizes the strong scaling behavior across different hardware configurations, demonstrating the parallel efficiency and computational acceleration achievable for production-scale molecular dynamics simulations.

In the upcoming benchmarking sections, we present complete benchmarking recipes and more detailed results for both ADH and STMV systems using the optimized HIP-enabled GROMACS installation on LUMI, further improving the baseline results presented in the Tables 1 and 2 of the [Ablation study](#ablation-study).

### STMV Benchmarks

The STMV (Satellite Tobacco Mosaic Virus) benchmark is a larger molecular system that benefits from strong scaling across multiple GPUs. Download and prepare the STMV benchmark dataset from the [AMD InfinityHub GROMACS repository](https://github.com/amd/InfinityHub-CI/blob/main/gromacs/README.md#stmv):

```bash
# Download the STMV benchmark dataset
wget https://raw.githubusercontent.com/amd/InfinityHub-CI/main/gromacs/docker/benchmark/stmv/stmv.tar.gz
tar -xzf stmv.tar.gz
mv topol.tpr stmv_topol.tpr
```

#### Running the STMV Benchmark with HIP-enabled GROMACS

The following sbatch script runs the STMV benchmark across 2 nodes (16 GCDs) using the baremetal HIP-enabled GROMACS build. The script is adapted from the [CSC GROMACS documentation for LUMI](https://docs.csc.fi/apps/gromacs/#lumi).

```{tip}
The hexadecimal CPU masks in the benchmark scripts below are LUMI-specific. On other systems, refer to your system documentation for CPU topology and binding configuration.
```

```bash
#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# Set up the RUNTIME environment
module load LUMI/25.03 partition/G
module load PrgEnv-cray
module load rocm/6.3.4
module load craype-accel-amd-gfx90a

# The path to the GROMACS bare metal libraries
GMX_INSTALL_DIR="/project/$PROJECT_NUMBER/software/gromacs/4947-hip-feature-enablement_v1"

export OMP_NUM_THREADS=7

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1
export GMX_FORCE_UPDATE_DEFAULT_GPU=1
export ROC_ACTIVE_WAIT_TIMEOUT=0
export AMD_DIRECT_DISPATCH=1

# From docs: Setting these variables will help circumvent most known issues
export MPICH_SMP_SINGLE_COPY_MODE=CMA
export MPICH_MALLOC_FALLBACK=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID%SLURM_GPUS_PER_NODE))
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

# Source GROMACS installation to set the executable PATH
source ${GMX_INSTALL_DIR}/bin/GMXRC

srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    -nsteps 100000 -resetstep 90000 \
    -noconfout \
    -nb gpu \
    -bonded gpu \
    -pme gpu \
    -npme 1 \
    -nstlist 100 \
    -g stmv_baremetal \
    -s stmv_topol.tpr

rm -rf ./select_gpu
```

Save the script as `run_stmv_baremetal.sh` and submit it with:

```bash
sbatch --account=$PROJECT_NUMBER run_stmv_baremetal.sh
```

#### Running the STMV Benchmark with LUMI GROMACS Module

For comparison, you can run the same benchmark using GROMACS 2025.4-gpu. This script follows the CSC documentation but is again adjusted to run on two nodes:

```bash
#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

module use /appl/local/csc/modulefiles
module load gromacs/2025.4-gpu

export OMP_NUM_THREADS=7

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID%SLURM_GPUS_PER_NODE))
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    -nsteps 100000 -resetstep 90000 \
    -noconfout \
    -nb gpu \
    -bonded gpu \
    -pme gpu \
    -npme 1 \
    -nstlist 100 \
    -g stmv_gmx \
    -s stmv_topol.tpr

rm -rf ./select_gpu
```

Save the script as `run_stmv_lumi.sh` and submit it with:

```bash
sbatch --account=$PROJECT_NUMBER run_stmv_lumi.sh
```

#### STMV Results Comparison

We compare the HIP-enabled GROMACS build against GROMACS 2025.4-gpu across different hardware scales and nstlist values. Each value represents an average of 6 benchmark runs.

Table 5: STMV results with HIP-enabled GROMACS
| GPUs | GCDs | MPI Ranks | Threads (ntomp) | nstlist | Performance (ns/day) |
|---|---|---|---|---|---:|
| 1 | 1 | 1 | 7 | 100 | 17.6 |
| 1 | 1 | 1 | 56 | 100 | 18.1 |
| 1 | 1 | 1 | 56 | 400 | 16.1 |
| 4 | 8 | 8 | 7 | 100 | 99.3 |
| 4 | 8 | 8 | 7 | 400 | 111.2 |
| 8 | 16 | 16 | 7 | 100 | 108.1 |
| 8 | 16 | 16 | 7 | 400 | 119.5 |

Table 6: STMV results with GROMACS 2025.4-gpu
| GPUs | GCDs | MPI Ranks | Threads (ntomp) | nstlist | Performance (ns/day) |
|---|---|---|---|---|---:|
| 1 | 1 | 1 | 7 | 100 | 17.9 |
| 1 | 1 | 1 | 56 | 100 | 18.6 |
| 1 | 1 | 1 | 56 | 400 | 16.8 |
| 4 | 8 | 8 | 7 | 100 | 94.5 |
| 4 | 8 | 8 | 7 | 400 | 109.2 |
| 8 | 16 | 16 | 7 | 100 | 101.6 |
| 8 | 16 | 16 | 7 | 400 | 110.2 |

Our STMV benchmarks demonstrate strong scaling characteristics of HIP-enabled GROMACS, particularly evident at larger hardware scales. While single-GCD performance is comparable between the two builds (17.6-18.1 ns/day for HIP-enabled vs 17.9-18.6 ns/day for GROMACS 2025.4-gpu), the HIP-enabled build shows clear advantages with increased domain decomposition. At 8 GPUs (16 GCDs), the HIP-enabled build achieves 119.5 ns/day with optimized neighbor list settings (nstlist=400), representing an 8.5% improvement over GROMACS 2025.4-gpu (110.2 ns/day) and a 6.8x speedup from single-GCD performance.

These results highlight the importance of tuning the neighbor list update frequency (nstlist) for production-scale simulations. For STMV, nstlist=400 consistently outperforms nstlist=100 in multi-GPU configurations, as the reduced frequency of neighbor list construction amortizes communication overhead across more MD steps.

The tables also demonstrate the thread configuration considerations we discussed in the [LUMI-Specific Optimizations](#lumi-specific-optimizations) section. Table 5 shows that using 56 threads (matching LUMI's available cores) yields 18.1 ns/day compared to 17.6 ns/day with 7 threads and 17.5 ns/day with 64 threads for single-GCD runs. While the InfinityHub recipes default to 64 OpenMP threads, we use thread counts aligned with LUMI's actual core availability to avoid the oversubscription penalties we described earlier. For multi-GPU runs, we use fewer threads per rank (7 in our benchmarks) to ensure proper distribution across MPI ranks while maintaining optimal CPU utilization.

The HIP-enabled GROMACS build's optimized GPU-aware MPI implementation and GTL library integration enable more efficient inter-GPU communication. This makes it the preferred choice for large-scale molecular dynamics production runs where strong scaling efficiency directly impacts time-to-solution for long-timescale simulations.

### Maximizing GPU utilization with small systems: ADH Benchmarks with Multidir

In general, MD studies require repeating simulations using a number of replicates with different starting conditions (velocity seed, initial coordinates) to ensure that any observations are significant and reproducible. Other studies directly rely on combining several simulations to obtain the final results, and thus need a number of replicates at the same time.
GROMACS enables running several simulations together using the `-multidir` flag, for either independent or coupled simulations, depending on the kind of study being performed.

For smaller molecular systems like ADH (Alcohol Dehydrogenase) that don't fully saturate GPU resources, it can be possible to run several of those simulations on the same device to achieve higher overall simulation throughput. By running multiple replicas in parallel, you can achieve better overall throughput when individual simulations cannot fully utilize all available resources. Furthermore, GPU partitioning significantly improves multidir performance by exposing each partition as an independent device; see [Applying Compute Partitioning for Workloads on MI300X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/gpu-partitioning-workloads/README.html) for details.

#### Setting Up the Multidir Folder Structure

First, download and prepare the ADH benchmark dataset. The ADH dodecahedron benchmark is available from the [AMD InfinityHub GROMACS repository](https://github.com/amd/InfinityHub-CI/tree/main/gromacs#adh-dodec):

```bash
# Download the ADH benchmark dataset
wget https://raw.githubusercontent.com/amd/InfinityHub-CI/main/gromacs/docker/benchmark/adh_dodec/adh_dodec.tar.gz
tar -xzf adh_dodec.tar.gz
mv topol.tpr adh_dodec_topol.tpr
```

Before running multidir simulations, create a directory for each independent replica. Place the `adh_dodec_topol.tpr` topology file in your working directory alongside the run directories created with the following command:

```bash
# Create directories for 96 independent simulations (for 8 GPU / 2 node run)
for i in $(seq 1 96); do mkdir -p run_$i; done
```

#### Running the ADH Benchmark with HIP-enabled GROMACS

The following sbatch script runs 96 independent ADH simulations across 2 nodes (8 GPUs / 16 GCDs) using the baremetal HIP-enabled GROMACS build from this blog post. The script is adapted from the [CSC GROMACS documentation for LUMI](https://docs.csc.fi/apps/gromacs/#lumi) and the [LUMI GROMACS throughput tutorial](https://docs.csc.fi/support/tutorials/gromacs-throughput/#example-batch-script-for-lumi), and configures the runtime environment, sets GPU-aware MPI environment variables, and uses CPU affinity masks to bind processes to appropriate cores.

The runtime parameters in the script (such as GPU-aware MPI settings and GPU offloading flags) are configured based on the findings from our ablation study, which demonstrated that these optimizations are essential for achieving peak performance on LUMI's MI250X GPUs.

```bash
#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=48

# Set up the RUNTIME environment
module load LUMI/25.03 partition/G
module load PrgEnv-cray
module load rocm/6.3.4
module load craype-accel-amd-gfx90a

# The path to the GROMACS bare metal libraries: adjust for custom location
GMX_INSTALL_DIR="/project/$PROJECT_NUMBER/software/gromacs/4947-hip-feature-enablement_v1"

# Adjust accordingly with GPU and multidir values
export OMP_NUM_THREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1
export GMX_FORCE_UPDATE_DEFAULT_GPU=1
export ROC_ACTIVE_WAIT_TIMEOUT=0
export AMD_DIRECT_DISPATCH=1

# From docs: Setting these variables will help circumvent most known issues
export MPICH_SMP_SINGLE_COPY_MODE=CMA
export MPICH_MALLOC_FALLBACK=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID%SLURM_GPUS_PER_NODE))
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

# Source GROMACS installation to set the executable PATH
source ${GMX_INSTALL_DIR}/bin/GMXRC

srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    -nsteps 100000 -resetstep 90000 \
    -noconfout \
    -nb gpu \
    -bonded gpu \
    -pme gpu \
    -nstlist 250 \
    -g adh_dodec_baremetal \
    -multidir run_{1..96} \
    -s ../adh_dodec_topol.tpr

rm -rf ./select_gpu
```

Save the script as `run_adh_multidir_baremetal.sh` and submit it with:

```bash
sbatch --account=$PROJECT_NUMBER run_adh_multidir_baremetal.sh
```

#### Running the ADH Benchmark with LUMI GROMACS Module

For comparison, the following script (adapted from the CSC documentation) runs the same benchmark using GROMACS 2025.4-gpu.

```bash
#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=48

module use /appl/local/csc/modulefiles
module load gromacs/2025.4-gpu

export OMP_NUM_THREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_ENABLE_DIRECT_GPU_COMM=1
export GMX_FORCE_GPU_AWARE_MPI=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID%SLURM_GPUS_PER_NODE))
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    -nsteps 100000 -resetstep 90000 \
    -noconfout \
    -nb gpu \
    -bonded gpu \
    -pme gpu \
    -nstlist 250 \
    -g adh_dodec_gmx2025 \
    -multidir run_{1..96} \
    -s ../adh_dodec_topol.tpr

rm -rf ./select_gpu
```

Save the script as `run_adh_multidir_lumi.sh` and submit it with:

```bash
sbatch --account=$PROJECT_NUMBER run_adh_multidir_lumi.sh
```

#### Adjusting for Different Configurations

For example, to run on a single node with 4 GPUs and 48 multidir simulations, adjust the SBATCH parameters and multidir range in either script:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=48

# ... rest of the script remains the same ...

# Adjust multidir range to match 48 simulations (12 per GPU x 4 GPUs)
srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    ...
    -multidir run_{1..48} \
```

For a single MI250X GPU (which contains 2 GCDs) run with 12 multidir simulations, use:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=12

# Adjust OMP_NUM_THREADS for single GPU
export OMP_NUM_THREADS=4

# Adjust multidir range
srun --cpu-bind=${CPU_BIND} ./select_gpu gmx_mpi mdrun \
    ...
    -multidir run_{1..12} \
```

```{tip}
When adjusting the number of OpenMP threads, ensure that the total number of threads (i.e. `ntasks-per-node * OMP_NUM_THREADS`) matches or is smaller than the available CPU cores per node.
```

#### ADH Results Comparison

We compare the HIP-enabled GROMACS build presented in this blog post against GROMACS 2025.4-gpu, which is the latest officially available GROMACS version on LUMI at the time of writing. The HIP-enabled build includes AMD-specific optimizations that are not yet part of the official GROMACS releases. Each value in the tables below represents an average of 6 benchmark runs.

Table 7: ADH multidir results with HIP-enabled GROMACS
| GPU | Dataset | nstlist | GPUs | Multidir | Threads (ntomp) | Total cores | Performance (ns/day) | ns/day per GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MI250 | adh | 250 | 1 | 12 | 4 | 48 | 778 | 778 |
| MI250 | adh | 250 | 4 | 48 | 1 | 48 | 3169 | 792.25 |
| MI250 | adh | 250 | 8 | 96 | 1 | 96 | 6590 | 823.75 |

Table 8: ADH multidir results with GROMACS 2025.4-gpu
| GPU | Dataset | nstlist | GPUs | Multidir | Threads (ntomp) | Total cores | Performance (ns/day) | ns/day per GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MI250 | adh | 250 | 1 | 12 | 4 | 48 | 698 | 698 |
| MI250 | adh | 250 | 4 | 48 | 1 | 48 | 2863 | 715.75 |
| MI250 | adh | 250 | 8 | 96 | 1 | 96 | 5612 | 701.5 |

The HIP-enabled GROMACS build consistently outperforms GROMACS 2025.4-gpu by approximately 15% across all configurations. At the 8 GPU scale, the HIP-enabled build achieves 6590 ns/day compared to 5612 ns/day with the official GROMACS release.

## Summary

In this blog post, we demonstrated how to build and optimize HIP-enabled GROMACS for AMD MI250X GPUs on the LUMI supercomputer, based on the [AMD InfinityHub baremetal recipe](https://github.com/amd/InfinityHub-CI/tree/main/gromacs/baremetal). Our benchmarks show that the HIP-enabled build delivers approximately 15% better performance compared to the officially available GROMACS on LUMI for certain high-throughput simulation setups, as illustrated with the ADH multidir benchmark.

We designed this installation recipe for ease of use and reproducibility. By following the step-by-step instructions or using the provided automated installation script, you can deploy a fully optimized GROMACS build with minimal effort. The recipe handles all the complex aspects of the build process &mdash; from selecting the correct compiler toolchain and module environment to configuring CMake with optimizations for GPU-aware MPI and GTL library linking. Similarly, the benchmarking scripts are ready-to-use templates that you can easily adapt to different system sizes and hardware configurations, requiring only minor adjustments to SBATCH parameters and multidir counts.

While this guide uses LUMI-specific configurations, you can transfer this installation approach to other Cray-based HPC systems and AMD GPU clusters. The key principles &mdash; leveraging Cray Programming Environment, configuring explicit MPI and GTL library paths, and optimizing for AMD GPU architectures &mdash; provide a robust foundation that you can adapt to similar environments. This portability makes the recipe valuable not only for LUMI users but also for the broader HPC community working with AMD GPUs.

### Adapting to Other HPC Systems

When adapting this recipe for other environments or future systems, consider:

* **Use HIP GROMACS:** The HIP-enabled build provides optimal performance on AMD GPUs and is the recommended approach for MI250X and similar architectures.
* **Use multidir for small datasets:** For molecular systems that don't fully saturate GPU resources, running multiple simulations in parallel via the GROMACS `-multidir` feature significantly improves throughput.
* **System-specific tuning:** Always validate thread counts, MPI environment variables, and CPU affinity settings against the specific hardware configuration of your target system.

For other Cray-based HPC systems, the MPI and GTL library linking approach remains the same &mdash; adjust module versions and library paths to match your environment.

**Quick adaptation checklist:**

1. Set correct GPU architecture (`gfx90a` for MI250X, `gfx942` for MI300X)
2. Find equivalent modules: `module avail rocm`, `module avail fftw`
3. Locate MPI paths using `${MPICH_DIR}` or `mpicc --showme`
4. Update GROMACS installation base path to your system's convention
5. Adjust SLURM partition names and allocation identifiers
6. Refer to your system documentation for CPU topology and binding configuration

## Related Resources

You can apply the benchmarking principles and build strategies from this guide to newer AMD Instinct GPU architectures:

**Performance on Next-Generation Hardware:**

Newer AMD Instinct GPUs like MI300X and MI355X offer increased memory bandwidth and compute capabilities that can further accelerate your GROMACS simulations. Explore these resources to learn optimization techniques and performance characteristics for HPC workloads:

* [Optimizing LLM Workloads: AMD Instinct MI355X GPUs Drive Competitive Performance](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html) - Discover MI355X and ROCm 7 performance and scalability improvements that benefit compute-intensive scientific applications like GROMACS
* [AMD Instinct™ MI325X Produces Strong Performance in MLPerf Inference v5.0](https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html) - Learn about MI325X architecture and advanced optimization techniques for maximizing GPU utilization in the context of LLMs.

**GPU Partitioning for Multi-Workload Environments:**

We leveraged MI250X GCD partitioning (2 GCDs per physical GPU) throughout this guide. Newer generation devices like MI300X and MI355X extend this capability, allowing you to partition a single device into up to 8 XCDs (Compute Dies). For production environments running multiple simulations or compute-heavy workloads, you can use GPU partitioning to efficiently share compute resources:

* [GPU Partitioning Made Easy: Pack More AI Workloads Using AMD GPU Operator](https://rocm.blogs.amd.com/software-tools-optimization/gpu-operator-partitioning/README.html) - Learn how to partition AMD GPUs for multi-tenant HPC environments, enabling concurrent GROMACS simulations or mixed HPC/AI workloads on shared infrastructure.
* [Applying Compute Partitioning for Workloads on MI300X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/gpu-partitioning-workloads/README.html) - Explore how GPU compute partitioning can boost GROMACS and other parallel workloads on MI300X. The ADH benchmarks in that blog achieve up to 2x higher throughput on a single MI300X compared to [our MI250X results](#adh-results-comparison) on LUMI.

## Additional Readings

For readers looking to build on the techniques presented in this blog post, the AMD InfinityHub GROMACS repository provides the foundational baremetal recipes and benchmark datasets used throughout this work. Both the LUMI GROMACS Documentation and Tutorial by CSC offer additional context on optimizing throughput-oriented GROMACS workloads on LUMI's GPU partition. Additional resources for GROMACS installation, LUMI system documentation, and AMD GPU development are also listed below.

* [AMD InfinityHub GROMACS](https://github.com/amd/InfinityHub-CI/tree/main/gromacs)
* [CSC GROMACS Documentation for LUMI](https://docs.csc.fi/apps/gromacs/#lumi)
* [CSC GROMACS Throughput Tutorial](https://docs.csc.fi/support/tutorials/gromacs-throughput/)
* [GROMACS Official Website](https://www.gromacs.org/)
* [GROMACS Official Installation Guide](https://manual.gromacs.org/documentation/current/install-guide/index.html)
* [LUMI User Documentation](https://docs.lumi-supercomputer.eu/)
* [Cray Programming Environment Documentation](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)
* [AMD Developer Resources](https://developer.amd.com/resources/)

## Disclaimers

Benchmark results may vary depending on simulation system size, input parameters, and hardware allocation. Some features used in this guide may be in beta or subject to change. Always validate performance improvements with your own workloads.

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
