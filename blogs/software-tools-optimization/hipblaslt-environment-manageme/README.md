---
blogpost: true
blog_title: "Building and Deploying Custom hipBLASLt Libraries on AMD Instinct GPUs"
date: 18 Jun 2026
author: 'Yuchen Lin, Bill Ku, Chunhung Wang'
thumbnail: 'build-deploy-custom-hipblaslt-thumbnail.png'
tags: AI/ML, Performance, Optimization
category: Software tools & optimizations
target_audience: AI developers, AI practitioners
key_value_propositions: Build, package, and deploy custom hipBLASLt libraries on AMD Instinct GPUs with custom source builds, RPM/DEB packaging, and runtime version switching.
language: English
myst:
    html_meta:
        "author": "Yuchen Lin, Bill Ku, Chunhung Wang"
        "description lang=en": "Learn how to manage hipBLASLt environments with custom source builds, RPM/DEB packaging, and version switching on AMD Instinct GPUs."
        "keywords": "Kernels, Inference, hipBLASLt, Source Build, DevOps, MLOps, AMD Instinct GPUs"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yuchen Lin, Bill Ku, Chunhung Wang"
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

# Building and Deploying Custom hipBLASLt Libraries on AMD Instinct GPUs

General Matrix Multiply (GEMM) operations are a core component of many generative AI workloads. Whether you are running attention mechanisms in the prefill phase of a Large Language Model (LLM) or generating tokens sequentially during the decode phase, matrix multiplication performance has a direct impact on end-to-end latency and throughput.

For developers and engineers working within the AMD ecosystem, `hipBLASLt` provides optimized GEMM implementations for AMD Instinct™ GPUs. Frameworks such as PyTorch, vLLM, and SGLang can use these low-level kernels to execute matrix operations efficiently on supported hardware.

The pre-installed ROCm binaries, typically located in `/opt/rocm`, are sufficient for many standard use cases. However, some development and deployment workflows require more control over the `hipBLASLt` version in use. Common examples include:

* **Testing a recent bug fix:** You may need a fix that has been merged upstream but is not yet included in an official ROCm release.
* **Validating hardware-specific optimizations:** You may have completed a TensileLite tuning process and generated custom kernels for the matrix shapes used by your model.
* **Evaluating new architectures or algorithms:** You may need to test experimental changes without modifying the stable environment used by production workloads.

Relying only on the system-wide installation can limit operational flexibility and increase the risk of dependency conflicts. In shared or multi-tenant environments, using `sudo make install` to overwrite the default library can also affect other applications that depend on the installed ROCm stack.

Instead, developers often need a workflow to compile `hipBLASLt` from source, package the build into portable artifacts, and select library versions on a per-application basis. This guide walks through that workflow: compiling from source, managing custom builds for specific architectures such as the AMD Instinct™ MI300X (`gfx942`), deploying builds without requiring root privileges, and using Linux environment variables to control runtime library selection.

In this blog, you will learn how to compile `hipBLASLt` from source, build it for a specific GPU architecture such as the AMD Instinct™ MI300X (`gfx942`), package the result into portable `.deb` or `.rpm` artifacts, and switch between library versions at runtime without root privileges. By the end, you will be able to deploy and validate a custom `hipBLASLt` build on both development workstations and shared clusters, with confidence that the intended library is the one actually loaded at runtime.

## Test Environment

The procedures in this guide use the following reference hardware and software configuration:

| Component | Specification |
| --- | --- |
| **GPU** | AMD Instinct™ **MI300X** (`gfx942`) |
| **ROCm Package Repository** | **7.2** |
| **hipBLASLt Source Revision** | `1784d40186` from [`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries) |
| **OS Targets** | Ubuntu 22.04 and RHEL/RPM-based Linux distributions |

---

## Step 1: Preparing the Build Environment

Building a C++ project like `hipBLASLt` from source requires a correctly configured environment. You need development tools, compilers such as ROCm LLVM, and Python packages used to parse configurations and generate code.

Depending on your operating system, follow the instructions below to prepare your build environment. Do not skip the repository configuration steps, because the ROCm mathematical libraries depend on AMD compiler toolchains and related development packages. Install additional packages only when the build reports that they are missing. If a required ROCm or AMDGPU package is not available from your configured package manager, check [repo.radeon.com](https://repo.radeon.com/) for the repository or package that matches your operating system and ROCm version.

### Option A: Ubuntu / Debian Systems

For Ubuntu-based environments, we utilize the Advanced Package Tool (`apt`) to pull in the core build essentials and configure the official AMD repository.

```bash
# ==============================================================================
# 1. System Level Compilers and Tools
# ==============================================================================
# 'build-essential' provides the gcc/g++ compilers, make, and standard libc
# headers required for compiling the host-side C++ code.
apt install -y build-essential

# ==============================================================================
# 2. AMDGPU Repository Configuration (Example for Ubuntu 22.04 Jammy)
# ==============================================================================
# Download the official AMD package that configures your apt sources list
# to point to the secure ROCm package repositories.
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb

# Install the downloaded deb package to register the repository.
# You may need to run 'apt update' immediately after this step.
apt install ./amdgpu-install_7.2.70200-1_all.deb

```

### Option B: RHEL / RPM-based Systems

For enterprise environments using RPM-based distributions, such as Red Hat Enterprise Linux, Rocky Linux, or AlmaLinux 8.x, use the `dnf` package manager to install the required build dependencies.

```bash
# ==============================================================================
# 1. AMDGPU Repository and Clean Slate Configuration
# ==============================================================================
# If you are upgrading or repairing an environment, it is best practice to
# remove the internal or outdated installer first to avoid repository conflicts.
dnf remove amdgpu-install-internal

# Fetch and install the repository configuration package directly from AMD.
# This ensures dnf knows where to find the specialized rocm-llvm packages.
dnf install https://repo.radeon.com/amdgpu-install/7.2/rhel/8/amdgpu-install-7.2.70200-1.el8.noarch.rpm

# ==============================================================================
# 2. Development Tools and Compilers
# ==============================================================================
# Install the "Development Tools" group (equivalent to build-essential).
dnf groupinstall -y "Development Tools"

# Install clang, DRM headers, and the specialized ROCm LLVM compiler toolchain.
# The 'rocm-llvm-devel' package is the heart of the compilation process,
# containing the hipcc compiler used to build device-side GPU kernels.
dnf install -y clang libdrm-devel
dnf install rocm-llvm-devel
dnf install boost-devel
```

### CMake and Python Dependencies for Tensile and Build Scripts

The ROCm build system is sensitive to the CMake version, and the build process relies on Python scripts to generate Tensile configurations. Install these packages once after completing the operating-system-specific setup above:

```bash
python -m pip install cmake==3.26.4 pyyaml joblib packaging msgpack numpy invoke
```

---

## Step 2: Fetching the Source Code

Historically, AMD's mathematical libraries were maintained across multiple repositories. Recently, `hipBLASLt` and other core ROCm mathematical libraries were consolidated into a single repository to simplify development and dependency management.

You will need to clone the `rocm-libraries` repository and navigate to the `hipblaslt` project folder.

For reproducibility, check out a specific commit hash rather than building directly from a moving `develop` or `main` branch. In this example, we use commit `1784d40186`, so future rebuilds can use the same source revision.

```bash
# Clone the unified ROCm libraries repository. This might take a moment
# due to the extensive history and submodules involved.
git clone https://github.com/ROCm/rocm-libraries.git

# Enter the root directory of the repository
cd rocm-libraries

# Hard-reset the working tree to a specific, tested commit hash.
# This ensures our build is isolated from unexpected upstream changes.
git checkout 1784d40186

# Navigate into the specific project directory for hipBLASLt
cd projects/hipblaslt/
```

Before starting a build, record the source revision and keep the build log with the generated package. This makes it easier to connect a deployed binary back to the source tree that produced it. A simple convention is to include the commit hash and target architecture in the build directory name, then preserve the corresponding `tee` log alongside the package artifact.

It is also useful to keep the source checkout separate from production runtime environments. The source tree is needed for compiling, packaging, and debugging build failures, but the runtime environment only needs the installed package or the selected shared library path. Keeping these concerns separate reduces the chance that a temporary build artifact or local source modification changes the behavior of a production workload.

### Understanding the Build Artifact Layout

A successful `hipBLASLt` build creates several types of artifacts. The exact paths can vary by build option, but the same categories are generally present:

* **Build system files:** CMake and Ninja or Make files used to drive incremental builds.
* **Compiled libraries:** Shared libraries such as `libhipblaslt.so`, built for the selected architecture.
* **Client binaries:** Tools such as `hipblaslt-bench`, which are useful for functional and performance validation.
* **Installation staging directory:** A `rocm_install` layout that mirrors the final installation structure under `/opt/rocm`.
* **Package output:** `.deb` or `.rpm` files generated by CPack for system-level installation.

The staging directory is especially important when debugging deployment issues. If a package installation does not behave as expected, compare the files under the staging directory with the files installed on the target system. This helps determine whether the issue is in the build, the packaging step, or the runtime library search path.

For reproducibility, treat the package file, build log, source commit, architecture target, and validation command output as one bundle. This bundle provides enough context to answer common operational questions later, such as which build was deployed, which GPU architecture it targeted, and which `hipblaslt-bench` result was observed before the package was promoted.

---

## Step 3: Building hipBLASLt for Specific Architectures

Building a library like `hipBLASLt` can be computationally expensive. By default, compiling a ROCm library might build matrix multiplication kernels for every supported GPU architecture (the "fat binary" approach). This can take hours and consume significant disk space.

To reduce build time and produce a smaller binary, use the `-a` (architecture) flag to target only the hardware you intend to use. The `--build_dir` flag creates an isolated build directory with an explicit name.

The following commands build `hipBLASLt` with a clean workspace, dependency installation, and logging:

**For AMD Instinct™ MI300X (`gfx942`):**

```bash
# 1. Clean up any previous failed or stale builds to ensure a fresh compilation
reset; rm -rf build_1784d40186_942/

# 2. Execute the official installation script with precise targeting
#    -a gfx942: Instructs the compiler to only build kernels for MI300X.
#    -d: Automatically installs required ROCm dependencies.
#    -c: Builds the client applications and testing binaries.
#    2>&1 | tee <file>: Captures both stdout and stderr for later debugging.
./install.sh --build_dir build_1784d40186_942 -a gfx942 -d -c 2>&1 | tee build_1784d40186_942.log
```

**For Next-Gen Architecture (`gfx950`):**

```bash
# The exact same process, but targeting a different silicon architecture.
reset; rm -rf build_1784d40186_950/
./install.sh --build_dir build_1784d40186_950 -a gfx950 -d -c 2>&1 | tee build_1784d40186_950.log
```

### Build Flags

* `--build_dir <dir>`: Specifies a custom output directory. Naming it explicitly with the commit hash and architecture (e.g., `build_1784d40186_942`) makes version tracking across your workspace effortless and prevents accidental overwrites.
* `-a <arch>`: Limits compilation to the specified GPU architecture. This avoids building kernels for hardware that is not part of the target deployment.
* `-c` (Clients): Builds the client applications, including `hipblaslt-bench`. This benchmark is useful for validating numerical correctness and measuring performance of the compiled library.

### Choosing an Architecture-Specific Build Strategy

For a development workstation, a narrow architecture build is usually the right default. If you are testing on MI300X, building only `gfx942` reduces compile time, disk usage, and the amount of generated kernel logic that you need to inspect when debugging. The resulting package is easier to reason about because it is tied to one hardware target.

For a shared cluster, the decision depends on the hardware mix. If every node in the target pool uses the same GPU architecture, a single-architecture package is still preferred. If the cluster contains multiple GPU generations, build and publish separate packages for each architecture, or maintain a clearly labeled package matrix. Avoid deploying a package whose target architecture is ambiguous, because that makes later performance regressions harder to diagnose.

The architecture target should also appear in your build directory, package metadata if available, and validation notes. For example, `build_1784d40186_942` tells future readers that the package came from commit `1784d40186` and targeted `gfx942`. This naming convention is simple, but it prevents a common operational failure mode: installing a package that was built successfully but not built for the GPU architecture being tested.

---

## Step 4: Version Control and Portable Deployment

Once the compilation completes successfully, you will find the compiled shared object files (`.so`) nested deep within your build directory.

At this point, choose a deployment method for the custom library. As mentioned earlier, using `sudo make install` to overwrite `/opt/rocm/hipblaslt` is discouraged in multi-user environments, because it can affect applications that depend on the installed library version.

The following sections describe two approaches: packaging for cluster-wide deployments, and environment variable configuration for isolated testing.

### Choosing a Deployment Method

Use system packages when you need repeatable installation across multiple machines or containers. A `.deb` or `.rpm` package gives your deployment pipeline a single artifact to promote, scan, archive, and roll back. It also lets the operating system package manager track ownership of installed files. This is the preferred approach for cluster images, CI-produced containers, and environments where multiple users need the same `hipBLASLt` build.

Use `LD_LIBRARY_PATH` when you need fast local iteration. This approach is useful while testing a candidate library, comparing two builds, or validating a fix before publishing a package. It does not require root privileges and it keeps the system installation unchanged. The tradeoff is that runtime behavior now depends on the launch environment. If the environment variable is missing, ordered incorrectly, or overwritten by a job launcher, the application can silently fall back to the system library.

The table below summarizes the operational tradeoffs:

| Deployment method | Best use case | Main advantage | Main risk |
| --- | --- | --- | --- |
| `.deb` / `.rpm` package | Cluster, container, or production deployment | Package manager owns installed files and supports repeatable rollout | Requires package installation privileges |
| `LD_LIBRARY_PATH` | Local validation or per-job experimentation | No system installation change required | Runtime behavior depends on environment variable ordering |

For production workflows, a common pattern is to validate with `LD_LIBRARY_PATH` first, then promote the same source revision through the packaging path. This keeps early testing flexible while ensuring the final deployment is traceable and reproducible.

### Method A: Generating an RPM/DEB Package for the Cluster

If you have validated your custom build and need to distribute it across multiple bare-metal nodes, package the compiled binaries into a standard installer file (`.rpm` for RHEL-based systems or `.deb` for Ubuntu/Debian).

The CMake build system integrated into `hipBLASLt` uses CPack. After a successful build from Step 3, navigate to the `release` directory inside your custom build folder and invoke the CMake packaging target. CPack gathers the compiled binaries, headers, and library logic files, then creates a system-native package.

```bash
# 1. Navigate to the release directory of your specific architecture build
cd build_1784d40186_942/release/

# 2. Generate the installation package using the CMake build system
# The '-- package' target automatically detects your OS and builds the correct format.
cmake --build . -- package
```

Depending on your underlying operating system, the system will output a file named something like `hipblaslt-1.1.0-Linux.deb` or `.rpm` within that directory.

You can then copy this artifact into MLOps pipelines or Dockerfiles for containerized deployment. This helps keep the installed `hipBLASLt` version consistent across development, staging, and production environments:

```dockerfile
# ==============================================================================
# Example snippet inside a Dockerfile for Ubuntu deployments
# ==============================================================================
# Copy the custom-built Debian package from the host machine into the container
COPY hipblaslt-*.deb /tmp/

# Install the package. The package manager handles placing the files into
# the correct /opt/rocm/... directories safely.
RUN apt-get update && apt-get install -y /tmp/hipblaslt-*.deb

# Verify that the hipBLASLt package is installed.
RUN dpkg -l | grep hipblaslt

# ==============================================================================
# OR for RHEL/CentOS deployments
# ==============================================================================
COPY hipblaslt-*.rpm /tmp/
RUN dnf install -y /tmp/hipblaslt-*.rpm

# Verify that the hipBLASLt package is installed.
RUN rpm -qa | grep hipblaslt
```

After confirming that the package manager reports the installed package, run the same `hipblaslt-bench` configuration before and after installation to verify that the expected `hipBLASLt` build is loaded at runtime:

```bash
hipblaslt-bench -m 64 -n 1268 -k 320 -r f16_r --transA T --transB N --print_kernel_info
```

A baseline run with the previous installation reports the earlier `hipBLASLt` git version and selected kernel:

```text
hipBLASLt git version: 6cf84a89a7
Query device success: there are 8 devices. (Target device ID is 0)
Device ID 0 : AMD Radeon Graphics gfx942:sramecc+:xnack-
with 206.1 GB memory, max. SCLK 1420 MHz, max. MCLK 1300 MHz, compute capability 9.4

Is supported 1 / Total solutions: 1
hipblaslt-Gflops: 1324.93
hipblaslt-GB/s: 24.1095
us: 39.2
Solution index: 139588
Solution name: Cijk_Alik_Bljk_HHS_BH_UserArgs_MT112x128x64_MI16x16x1_...
kernel name: Cijk_Alik_Bljk_HHS_BH_UserArgs_MT112x128x64_MI16x16x1_...
```

After installing the custom package built from commit `1784d40186`, the same command should report the new git version and the kernel selected from the updated library:

```text
hipBLASLt git version: 1784d40186
Query device success: there are 8 devices. (Target device ID is 0)
Device ID 0 : AMD Radeon Graphics gfx942:sramecc+:xnack-
with 206.1 GB memory, max. SCLK 1420 MHz, max. MCLK 1300 MHz, compute capability 9.4

Is supported 1 / Total solutions: 1
hipblaslt-Gflops: 4679.03
hipblaslt-GB/s: 85.1434
us: 11.1
Solution index: 141156
Solution name: Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_...
kernel name: Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_...
```

This check verifies both the loaded `hipBLASLt` revision and the runtime behavior for a fixed GEMM shape. The exact device count can vary by system, but the reported git version should match the package you installed.

### Method B: Runtime Library Selection with `LD_LIBRARY_PATH`

For rapid iteration, validating specific TensileLite tuning parameters, or bundling the library directly with an application such as vLLM, SGLang, or a custom PyTorch serving script without requiring root privileges, you can copy the compiled shared object file (`.so`) and select it at runtime.

This leverages the Linux dynamic linker's search hierarchy.

1. **Locate the Library:** After building, your custom library is located inside the release installation directory:
   `./build_1784d40186_942/release/rocm_install/lib/libhipblaslt.so`

2. **Isolate the Binary:** Create a custom, safe directory in your user workspace and copy the library files there.

   ```bash
   # Create a personal directory for custom libraries
   mkdir -p /home/user/custom_libs/hipblaslt_gfx942

   # Copy the shared object and its associated symlinks
   cp -a ./build_1784d40186_942/release/rocm_install/lib/libhipblaslt.so* /home/user/custom_libs/hipblaslt_gfx942/
   ```

3. **Select the library at runtime:** When you launch your AI application, prefix the command with the `LD_LIBRARY_PATH` environment variable.

   When a Linux program starts, the dynamic linker (`ld.so`) searches for required libraries in a specific order. By defining `LD_LIBRARY_PATH`, you force the linker to look in your custom folder *first*, before it falls back to checking the system default in `/opt/rocm/lib`.

   ```bash
   # Prepend your custom path to the existing environment variable
   export LD_LIBRARY_PATH=/home/user/custom_libs/hipblaslt_gfx942:$LD_LIBRARY_PATH

   # Now run your workload. The OS will seamlessly load your custom-built hipBLASLt!
   python run_llm_inference.py
   ```

**Verification using `ldd`:**
To definitively prove that your Python script or binary is loading the correct, custom version of the library, you can use the Linux `ldd` (List Dynamic Dependencies) utility on your executable.

```bash
# Example showing that the system is resolving to your custom path
$ ldd ./build_1784d40186_942/release/clients/staging/hipblaslt-bench | grep libhipblaslt

# Output should point to your home directory, NOT /opt/rocm/
libhipblaslt.so => /home/user/custom_libs/hipblaslt_gfx942/libhipblaslt.so (0x00007f9c2...)
```

Using `LD_LIBRARY_PATH` allows you to run the same Python script in two different terminal windows with different `hipBLASLt` libraries. One terminal can use the system default library, while another uses the custom build, without modifying the system installation.

## Operational Checklist

Before handing a custom `hipBLASLt` build to another user or deploying it to a shared environment, run through a short checklist. The goal is to verify not only that the package installs, but also that the runtime path and benchmark behavior match the build you intended to deploy.

1. **Confirm the source revision:** Record the `git rev-parse HEAD` output from the `rocm-libraries` checkout used for the build.
2. **Confirm the target architecture:** Verify that the build command includes the expected `-a` target, such as `gfx942` for MI300X.
3. **Preserve the build log:** Keep the `tee` log from the `install.sh` invocation. It captures dependency installation, compiler output, and packaging messages.
4. **Verify package ownership:** Use `dpkg -l | grep hipblaslt` or `rpm -qa | grep hipblaslt` after installation.
5. **Verify runtime loading:** Use `ldd` for binaries, or a controlled `LD_LIBRARY_PATH` test for applications launched from Python or a job scheduler.
6. **Run a fixed benchmark shape:** Use the same `hipblaslt-bench` command before and after installation so performance and kernel selection are comparable.
7. **Document rollback:** Keep the previous package or container tag available until the new build has passed functional and performance checks.

Rollback should be explicit. For package-based deployment, rollback usually means reinstalling the previous `.deb` or `.rpm`, then rerunning the package manager query and benchmark check. For `LD_LIBRARY_PATH` deployment, rollback can be as simple as removing the custom path from the environment, but it is still important to verify the loaded library with `ldd` or equivalent runtime logging.

In multi-user systems, avoid changing `/opt/rocm` manually outside the package manager. Manual file replacement can leave the system in a state that is hard to audit, especially if the package database still reports the older version. If you need a temporary override, prefer a per-user or per-job runtime path and document the command used to launch the workload.

---

## Troubleshooting Common Build Issues

Even with a correctly configured build script, compiling low-level GPU libraries can fail due to hardware or environment constraints. Here are two common issues:

* **Out of Memory (OOM) Kills During Compilation:** Building massive C++ template libraries consumes an enormous amount of RAM. If your build abruptly fails with a "Killed" message or a segmentation fault from the compiler, your machine likely ran out of memory. To avoid this, limit the number of parallel build jobs. You can pass the `-j` flag to the underlying Make/Ninja system (e.g., `-j 8` to restrict compilation to 8 CPU cores), sacrificing build speed to ensure the system doesn't exhaust its RAM.
* **CMake Version Mismatch:** If the configuration step fails immediately, complaining about syntax or unrecognized commands, ensure that your system's default CMake is not overriding the `3.26.4` version you installed via `pip`. You can verify this by typing `cmake --version` in your terminal.

---

## Summary

In this blog, you explored a complete workflow for managing custom `hipBLASLt` environments: preparing the build environment on Ubuntu and RHEL-based systems, fetching a fixed source revision, building for a specific GPU architecture such as the MI300X (`gfx942`), packaging the result into `.deb`/`.rpm` artifacts, and selecting library versions at runtime with `LD_LIBRARY_PATH`. You also verified the active library using package queries, `ldd`, and a fixed `hipblaslt-bench` shape.

With these techniques, you can test upstream fixes, validate TensileLite-tuned kernels, and experiment with new architectures without disturbing the system-wide ROCm installation.

In upcoming posts, our team will go deeper into TensileLite tuning workflows and performance analysis for production inference on AMD Instinct GPUs. Stay tuned for more hands-on guides.

## Acknowledgement

We would like to express our thanks to our colleagues [Brian Chang](../../authors/brian-chang.md), [Eveline Chen](../../authors/eveline-chen.md), [Bobo Fang](../../authors/bobo-fang.md), [Clement Lin](../../authors/clement-lin.md), [Kaiping Lu](../../authors/kaiping-lu.md), and [Menghsuan Yang](../../authors/menghsuan-yang.md) for their insightful feedback and technical assistance.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
