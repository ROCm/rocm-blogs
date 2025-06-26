---
blogpost: true
date: 26 June 2025
blog_title: "Performance Profiling on AMD GPUs – Part 1: Foundations"
author: 'Gina Sitaraman, Thomas Gibson, Luka Stanisic, Giacomo Capodaglio, Alessandro Fanfarillo, Asitav Mishra'
thumbnail: '2025-06-25-profiling-guide.png'
tags: HPC, Performance, Optimization, Profiling
category: "Software tools & optimizations"
key_value_propositions: ""
target_audience: ""
language: English
myst:
    html_meta:
        "description lang=en": "Part 1 of our GPU profiling series introduces ROCm tools, setup steps, and key concepts to prepare you for deeper dives in the posts to follow."
        "keywords": ""
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Data Science, Design, Simulation & Modeling"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "property=og:locale": "en_US"
---

# Performance Profiling on AMD GPUs – Part 1: Foundations

Profiling tools are vital for optimizing application performance, especially on heterogeneous platforms including advanced supercomputing systems like [El Capitan](https://hpc.llnl.gov/hardware/compute-platforms/el-capitan) and [Frontier](https://www.olcf.ornl.gov/frontier/).
There is a variety of  powerful profiling tools that can help you analyze your application’s performance on AMD Instinct™ hardware.
This three-part blog series introduces the iterative process of profiling and optimizing applications on AMD GPUs. In this first post, we focus on foundational topics: defining the intended audience, outlining prerequisites, and providing installation guidance.
Whether you are new to GPU profiling or an experienced performance engineer, this series is designed to help you unlock the full potential of AMD hardware by understanding and applying the right profiling strategies and tools.

This blog series, starting with this introduction and continuing with two further posts tailored for both novice and advanced users,
will guide you through the iterative process of performance profiling and optimization,
detailing appropriate tools for each stage.

This blog series is designed for readers with all levels of experience in GPU performance analysis.

In this first post, we begin by defining the target audience and outlining prerequisite knowledge. We then walk through the installation and setup process, ensuring your environment is ready for profiling. Finally, we provide a high-level overview of the profiling tools available in the ROCm™ ecosystem, establishing the foundation for deeper exploration in the posts that follow.

## Targeted audience and prerequisites

This blog series is designed for two audiences based on their experience with GPU programming and profiling: novice and advanced users.

Below, we outline the prerequisites for each group.

### Novice

At the novice level, we make the following assumptions about the reader’s prior knowledge:

- The reader is aware that some of the applications being profiled execute on a GPU.
- The reader has no prior exposure to performance assessment methodologies such as roofline modeling,
compute- vs. memory-bound limitations, or latency-bound kernels.
However, they have a basic understanding of the cost associated with data transfers between CPU and GPU memory.
- The application being profiled is capable of running on a single GPU.
- The reader has a fundamental understanding of the algorithmic purpose of each GPU kernel within the application.
- The application’s performance has been evaluated across different hardware platforms,
with results indicating superior performance on non-AMD hardware of comparable specifications.

This foundational understanding will guide our approach as we explore GPU performance profiling in a structured and accessible manner.

### Advanced

At the advanced level, we assume that readers have a deeper understanding of GPU performance analysis and system architecture.
Specifically, the advanced level content is tailored for those who meet the following criteria:

- The reader is familiar with roofline models, key performance bottlenecks in GPU kernels,
and fundamental GPU architectural concepts, including local data share (LDS) memory, caches, and coalesced memory accesses.
- The reader understands the architectural differences between AMD GPUs and competing hardware,
including variations in memory hierarchy, compute units, and execution models.
- The application being analyzed is designed to run across multiple GPUs and potentially across multiple nodes,
requiring considerations for inter-GPU and inter-node communication overheads.

With these assumptions, this blog will focus on advanced performance profiling techniques, optimization strategies, and hardware-aware tuning for multi-GPU workloads.

We will now outline the tools we plan to feature in upcoming novice and advanced blog posts.

## `rocprofiler-sdk` library

Profiling and tracing capabilities for AMD GPUs are built into the newly
designed [`rocprofiler-sdk`](https://github.com/ROCm/rocprofiler-sdk)
library shipped with ROCm&trade;.
The `rocprofiler-sdk` infrastructure can be used to develop tools for device
activity tracing and hardware counter collection. This library replaces
the functionalities provided by the legacy `rocprofiler` and `roctracer` libraries.

## `rocprofv3` for tracing and raw counter collection

`rocprofv3` is a command-line tool for tracing device activity and collecting
raw GPU counters for kernel performance analysis. It is a tool that is built
on the functionality provided by the `rocprofiler-sdk` library. Some of the most
useful features of `rocprofv3` are:

- Collect a variety of traces for ROCm based applications (HIP API, HSA API,
offloaded kernels, memory copies, scratch memory, marker API, etc.)
- Collect device hardware counters for GPU kernel performance analysis
- Find GPU hotspots quickly
- Profile Python workloads efficiently on AMD GPUs
- Visualize the outputs in Perfetto trace format (`.pftrace`) or OTF2 format
- Use the JSON format for building new and useful post-processing capabilities

`rocprofv3` replaces `rocprof` and `rocprofv2` legacy tools with better,
robust functionality. We strongly urge you to use `rocprofv3` as the legacy
tools will be deprecated in future ROCm releases.

## `rocprof-sys` for holistic application tracing

The ROCm systems profiler, also known as `rocprof-sys`, is best suited to collect
host, device, and communication (MPI) activity in one comprehensive,
unified trace of your application's run. `rocprof-sys` extends the
capabilities of `rocprofv3` by using tools such
as [AMDuProf](https://www.amd.com/en/developer/uprof.html),
[`amd-smi`](https://rocm.docs.amd.com/projects/amdsmi/en/latest/),
[`perf`](https://perfwiki.github.io/main/) and others to provide a comprehensive
view of the system where your application ran.

With configurable runtime options, you can use it for call-stack sampling,
binary instrumentation, causal profiling, and hardware counter collection.
Simply put, this is the tool you would use
if you want to understand what is happening in your application on the host *and*
on the device. The output trace is in protobuf format (`.proto`) for easy
visualization using [Perfetto UI](https://ui.perfetto.dev) in the Chrome browser.

`rocprof-sys` has evolved from the former `Omnitrace` research tool from AMD.
Support for the `rocprofiler-sdk` library is being added to bring newer, more
advanced capabilities such as OMPT support for tracing Fortran codes with
OpenMP&reg; offload, network performance profiling, etc.

## `rocprof-compute` for kernel performance analysis

The ROCm compute profiler, also known as `rocprof-compute`, is useful to analyze the
performance characteristics of a given GPU kernel. The tool performs automated
collection of hardware counters via application replay using the rocprofiler
tool (uses legacy `rocprof` tool by default, but can be configured to use
`rocprofv3`) in the back end. `rocprof-compute` can be used to:

- Produce a roofline plot showing the performance limiters of your kernels at a glance
- Perform baseline comparisons of workloads to easily visualize the impact of your kernel optimizations
- Visualize system speed-of-light, memory throughput analysis, compute throughput analysis, etc.
- Analyze the collected profile using the command-line or a standalone graphical user interface

`rocprof-compute` has evolved from the former `Omniperf` research tool from AMD.
Support for using `rocprofv3` has been added as a beta feature in ROCm 6.4 and we
strongly encourage you to use it and report any issues via the GitHub repo.

## Installation and testing

In this section, we provide instructions on how to install the AMD tools,
and how to verify that the installation has completed correctly. The scripts that we use can install on bare metal,
but we **STRONGLY** suggest you first test them in a container, as will be explained in the section
[testing the installation](./README.md#testing-the-installation).

We use scripts from two AMD open-source GitHub repositories:

- Repo with installation scripts: `https://github.com/amd/HPCTrainingDock`: this repository is intended to provide a model installation for AMD (and ROCm supporting) software that is relevant to HPC and AI/ML applications running on AMD data center GPUs. If you decided to browse it on your own, you will find many more scripts than just those that we discuss here, which focus only on AMD tools. See the [`README`](https://github.com/amd/HPCTrainingDock/blob/main/README.md) of the repo for detailed information on how to browse its content.

- Repo with testing scripts `https://github.com/amd/HPCTrainingExamples`: this repository contains a multitude of examples aimed at showcasing the features of the ROCm software stack in the context of HPC and AI/ML applications for data center GPUs. See the [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/README.md) of the repo for a detailed walk through of the repo structure. The directory relevant to this document is the [`tests`](https://github.com/amd/HPCTrainingExamples/tree/main/tests) directory, where we provide a test suite to check that the software installed with the scripts in the first repo mentioned has actually been successful. The test scripts in the suite mostly represent sanity checks to verify that the main functionalities of the software produce an output, and it is the responsibility of the user to make sure that the output is indeed correct for their application.

Please be aware that the above repos are in continuous evolution, therefore we encourage readers to periodically review the `README` files on those repos for the latest changes.

### Assumptions

We assume an Ubuntu 22.04  operating system, however, most of the scripts will also work for other distributions (such as Red Hat and SUSE). We are currently working on adding support for Ubuntu 24.04 as well. We also assume that the user's system has some basic operating system packages already installed, as well as `lmod` to enable the use of modules. If that is not the case, these can be installed by doing, from a terminal:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/rocm/scripts
./baseospackages_setup.sh
./lmod_setup.sh
```

Note that `sudo` privileges are needed for the above scripts.

### Installation of AMD tools

Let us begin by discussing how to proceed with the installation of the tools. The scripts provided here also create module files so that packages can be loaded just by doing `module load <package>`. We would like to stress that setting up module files correctly is an important part of the installation process, as it allows users to maintain a clean software environment that can be easily controlled and modified by loading and unloading modules.

#### `rocm` installation

Let us begin with the installation of ROCm, which can be carried out with this script: [`ROCm install script`](https://github.com/amd/HPCTrainingDock/blob/main/rocm/scripts/rocm_setup.sh). To run and install ROCm, open a terminal window and do:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/rocm/scripts
./rocm_setup.sh --rocm-version ${ROCM_VERSION} --amdgpu-gfxmodel ${AMDGPU_GFXMODEL}
```

NOTE: for the above script to complete, you need `sudo` privileges. As options to the script, you can supply the desired ROCm version and the desired architecture for the AMD GPU (for example, MI300 has gfx942 and MI200 has gfx90a). The default installation directory is: `/opt/rocm-${ROCM_VERSION}`. You can replace an existing installation of ROCm by supplying the `--replace` input flag to the above run command. The ROCm installation script will take care of installing the software in the ROCm stack, such as for instance the compilers, hipify tools, rocprofv3.

#### `rocprofiler-sdk` installation from source

The `rocprofiler-sdk` library is part of the ROCm stack and therefore will be automatically installed with the script in the previous section. However, it is sometimes helpful to install from source, in case one is interested in testing the very latest features that may not yet have been included in an official release. To do so, open a terminal window and do:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/tools/scripts
./rocprofiler-sdk_setup.sh --rocm-version ${ROCM_VERSION} --install-path ${INSTALL_PATH}  --module-path ${MODULE_PATH} --github-branch ${GITHUB_BRANCH}
```

NOTE: for the above script, `sudo` privileges are not needed as long as the `${INSTALL_PATH}` and the `${MODULE_PATH}` have write access for the user. `sudo`, however, might still be required to update the repositories for `apt-get` in case the sources are not enabled.
The `${GITHUB_BRANCH}` string decides what branch to clone for the installation from source, default is `amd-staging`.

#### `rocprof-sys` installation

The installation of `rocprof-sys` can be done in two ways:

- With `sudo apt-get install`: this can be achieved with this dedicated [`script`](https://github.com/amd/HPCTrainingDock/blob/main/rocm/scripts/rocm_rocprof-sys_setup.sh), which will also create a module file. To run the script, open a terminal window and run:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/rocm/scripts
./rocm_rocprof-sys_setup.sh --rocm-version ${ROCM_VERSION}
```

The `${ROCM_VERSION}` supplied to the script is used to check that the ROCm version is greater than 6.1.2: since ROCm 6.2.0 `omnitrace` (and then `rocprof-sys` since ROCm 6.3.0 ) are packaged with ROCm, so the script will install in `/opt/rocm-${ROCM_VERSION}`. The script will produce an error message and exit if the ROCm version is lower than 6.2.0. Note that since ROCm 6.4.0, `rocprof-sys` is already included in the ROCm stack, hence it is no longer needed to install it with `sudo apt-get install`. The above script checks whether the `rocprof-sys` binaries are already present in the ROCm directory, and if they are (which is the case for ROCm 6.4.0), it does not run `sudo apt-get install`. If the `rocprof-sys` binaries are present (whether installed with ROCm or with `apt-get`) the script creates a module file.

- From source with a dedicated [`script`](https://github.com/amd/HPCTrainingDock/blob/main/tools/scripts/rocprof-sys_setup.sh): to run the script, open a terminal window and do:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/tools/scripts
./rocprof-sys_setup.sh --rocm-version ${ROCM_VERSION} --install-path ${INSTALL_PATH} --amdgpu-gfxmodel ${AMDGPU_GFXMODEL} --install-rocprof-sys-from-source 1 --module-path ${MODULE_PATH} --github-branch ${GITHUB_BRANCH} --mpi-module ${MPI_MODULE} --python-version ${PYTHON_VERSION} 
```

The above script will install either `rocprof-sys` (if the ROCm version is 6.3.0 or higher) or `omnitrace` (if the ROCm version is 6.2.4 or lower).  The flag `--install-rocprof-sys-from-source` needs to be set to 1, otherwise the script will not execute. Once again, if the `${INSTALL_PATH}` and `${MODULE_PATH}` have write access for the user, `sudo` privileges will not be needed. The default `${GITHUB_REPO}` is again `amd-staging`. The `${PYTHON_VERSION}` is the minor version for Python3, default is 10. The default for the `${MPI_MODULE}` is openmpi.

#### `rocprof-compute` installation

Similar to `rocprof-sys`, the installation of `rocprof-compute` can be done in two ways:

- With `sudo apt-get install`: this is achieved with this dedicated [`script`](https://github.com/amd/HPCTrainingDock/blob/main/rocm/scripts/rocm_rocprof-compute_setup.sh). To run the script, open a terminal window and run:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/rocm/scripts
./rocm_rocprof-compute_setup.sh --rocm-version ${ROCM_VERSION} --python-version ${PYTHON_VERSION}
```

The `${ROCM_VERSION}` supplied to the script is used to check that the ROCm version is greater than 6.1.2: since ROCm 6.2.0 `omniperf` (and then `rocprof-compute` since ROCm 6.3.0 ) are packaged with ROCm, so the script will install in `/opt/rocm-${ROCM_VERSION}`. The script will produce an error message and exit if the ROCm version is lower than 6.2.0. As for `rocprof-sys`, since ROCm 6.4.0 the binary for `rocprof-compute` is already included when installing ROCm, and the script above will not run `sudo apt-get install` if the binary is already present. For `rocprof-compute` however, it is still necessary to install the required Python dependencies, which is something that the above script takes care of, regardless of the version of ROCm.

- From source with a dedicated [`script`](https://github.com/amd/HPCTrainingDock/blob/main/tools/scripts/rocprof-compute_setup.sh): to run the script, open a terminal window and do:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock/tools/scripts
./rocprof-compute_setup.sh --rocm-version ${ROCM_VERSION} --install-path ${INSTALL_PATH} --install-rocprof-compute-from-source 1 --module-path ${MODULE_PATH} --github-branch ${GITHUB_BRANCH} --python-version ${PYTHON_VERSION}
```

The default `${GITHUB_BRANCH}` is `develop`. The Python requirements need to be installed also when installing `rocprof-compute` from source, and the above script takes care of that.

### Testing the installation

The scripts can be tested on a container before they are deployed on your system as bare metal installs. Docker or Podman are needed in your system for this task. Singularity can also be used but the process is different than what will be shown here, for details on how to use Singularity to test our installation scripts, see [`here`](https://github.com/amd/HPCTrainingDock?tab=readme-ov-file#13-singularity). Assuming Docker or Podman are available, open a terminal window and run:

```bash
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock
./bare_system/test_install.sh --rocm-version ${ROCM_VERSION} --use-makefile 1
```

Details on how the `test_install.sh` script  works are available [`here`](https://github.com/amd/HPCTrainingDock#22-training-enviroment-install-on-bare-system). In short, the above command will automatically get you in a container with Ubuntu 22.04, from where you can install the software described in the previous sections and test it in a clean and safe environment. Your username will be `sysadmin` and you will have `sudo` privileges. From the container, you could either just clone the repo with the scripts and install them manually as explained above, or do:

- `make rocm`: to install ROCm.
- `make rocprof-compute_rocm`: to install the rocprof-compute version that comes packaged with ROCm.
- `make rocprof-compute_source`: to install rocprof-compute from source.
- `make rocprof-sys_rocm`: to install rocprof-sys version that comes packaged with ROCm.
- `make rocprof-sys_source`: to install rocprof-sys from source.
- `make rocprofiler-sdk`: to install rocprofiler-sdk from source.

The above commands will rely on the `Makefile` that is copied over to the container, and use the default values for the input flags, hence if you want to specify user defined values, you should either modify the `Makefile`, or just install the software manually as explained above. If you want to use the installation of `rocprofiler-sdk` from source for other libraries, make sure to append its path to `LD_LIBRARY_PATH`.

For what concerns the verification of the correctness of the installation, we recommend you clone the HPCTrainingExamples repo `https://github.com/amd/HPCTrainingExamples` and run the test suite located in the `tests` directory as follows:

```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/tests
./runTests.sh
```

The `runTests.sh` script will use `ctest` to run over 200 sanity check scripts aimed at verifying that the main functionalities of the software installed are present. The scripts in the suite use modules and rely on `lmod`: tests for which a `module load <package>` fails will be marked as skipped.
You can also run a subset of the whole suite, selecting a specific package of interest, such as:

```bash
./runTest.sh --pytorch
```

To see all the options available for the subset, run the script with the `--help` input flag.
The tests in the suite are continuously enhanced and improved, and we strongly suggest to periodically check what is new on the two repositories mentioned above, as the content of the repos is growing and improving at record speed.

## Useful resources

The following are links to the GitHub repos and ROCm docs for the tools described
above for your quick reference.

- `rocprofiler-sdk`:
  - Open source at [rocprofiler-sdk github repo](https://github.com/ROCm/rocprofiler-sdk)
  - [`rocprofiler-sdk` library Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)
- `rocprofv3`:
  - Open source at [rocprofiler-sdk github repo](https://github.com/ROCm/rocprofiler-sdk)
  - [`rocprofv3` tool documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#using-rocprofv3)
- `rocprof-sys`:
  - Open source at [rocprofiler-systems github repo](https://github.com/ROCm/rocprofiler-systems)
  - [ROCM Systems Profiler documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html#rocm-systems-profiler-documentation)
- `rocprof-compute`:
  - Open source at [rocprofiler-compute github repo](https://github.com/ROCm/rocprofiler-compute)
  - [ROCm Compute Profiler documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/#rocm-compute-profiler-documentation)

## Summary

In this introductory post, we have outlined the importance of profiling tools in optimizing application performance on AMD GPUs, especially for advanced HPC platforms like El Capitan and Frontier.
We discussed the intended audience, differentiating between novice users, who require foundational guidance, and advanced users, who seek deeper insights into GPU architecture and multi-GPU profiling techniques.
Additionally, we introduced key profiling tools available within the ROCm ecosystem—rocprofiler-sdk, rocprofv3, rocprof-sys, and rocprof-compute—each tailored for different profiling scenarios, ranging from kernel-level performance analysis to comprehensive system-wide tracing.
Finally, we provided clear installation instructions, recommended testing methodologies, and pointed readers toward useful resources to support effective GPU profiling.
In the upcoming posts in the series, we will demonstrate how to apply these tools to real-world profiling scenarios, progressively advancing from basic analysis to advanced multi-GPU profiling.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD.
ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND.
USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT.
YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
