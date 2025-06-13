---
blogpost: true
blog_title: 'Boosting Computational Fluid Dynamics Performance with AMD Instinct™ MI300X'
key_value_propositions: ""
target_audience: ""
date: 14 Jan 2025
thumbnail: 'Ansys_Fluent_benchmarks_Blog.jpg'
author: Martin Huarte
tags: HPC
category: Ecosystems and Partners
language: English
myst:
    html_meta:
        "description lang=en": "The blog introduces CFD Ansys Fluent benchmarks and provides hands-on guide on installing and running four different Fluent models on AMD GPUs using ROCm."
        "keywords": "HPC, CFD"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models, Ecosystem and Partners, Benchmarks and Testing"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Design, Simulation & Modeling"
        "amd_blog_topic_categories": "HPC & Scientific Computing, Industry Applications & Use Cases, Software & Ecosystem"
---

# Boosting Computational Fluid Dynamics Performance with AMD Instinct™ MI300X

This blog will guide you, step-by-step, through the process of installing and running benchmarks with [Ansys Fluent](https://www.ansys.com/products/fluids/ansys-fluent) and AMD MI300X. We start with an overview of the Ansys Fluent CFD application and then show you how to set up an AMD MI300X system to run benchmarks. The blog benchmarks results demonstrate the dramatic impact the MI300X has on speeding up simulations, improving design efficiency, and reducing costs in the automotive, aerospace, and environmental engineering industries.

## Ansys Fluent Overview

Ansys Fluent® is well-known in the commercial computational fluid dynamics (CFD) space and is praised for its versatility as a general-purpose solver. Scientists and engineers alike use Fluent across numerous industry segments globally, particularly in the automotive and aerospace sectors, as well as in energy, materials and chemical processing, and high-tech industries. [Ansys recently integrated support for AMD Instinct™ MI200 and MI300 accelerators into Fluent fluid simulation software](https://community.amd.com/t5/instinct-accelerators/ansys-fluent-adds-amd-instinct-mi200-and-mi300-acceleration-to/ba-p/711883), significantly enhancing simulation efficiency and power. Fluent runs in production on AMD Instinct™ GPUs out of the box today. In order to run Ansys Fluent on AMD Instinct™ GPUs simply check a box in the Fluent launcher or add the `-gpu` flag to your command line, since the running environments are similar for the CPU and GPU solvers in Fluent fluid simulation software.

## Performance Testing Methodology

The benchmarks described in this blog evaluate the performance of the AMD Instinct™ MI300X platform and the NVIDIA H100 platform across key CFD use cases using four Ansys Fluent Benchmark models. Ansys designed these tests to measure time to solution speed and efficiency under configurations representative of real-world applications such sedan cars, airplane wings, and racing cars. We tailored the benchmarks workloads to reflect varying model sizes in terms of millions of cells, and efficiency in terms of cell types relevant to these use cases, as detailed in the table below. As will become apparent, the large 192 GB HBM3 memory capacity and high memory bandwidth, along with the new AMD Infinity Cache™ of the AMD Instinct MI300X makes the MI300X an excellent choice for these applications and their steady-state analysis requirements.

### Summary and Descriptions of Use Cases

| Use Case            | Size [millions of cells] | Cell Type   | Description                                                                                                                                                                                                                                                                                                                                 |
|---------------------|--------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sedan_4m            |            4             | Mixed       | Standard k-epsilon turbulence model that simulates the external flow over a passenger sedan. Allows customers to optimize the aerodynamic performance of their designs, automate shape optimizations, especially for complex components like side mirrors, improving the vehicle’s fuel efficiency, stability, and overall performance.          |
| aircraft_wing_14m   |            14            | Hexahedral  | Realizable K-e turbulence model simulates the airflow over an aircraft wing. Allows aviation customers to optimize the aerodynamic performance of their aircraft designs and analyze low-speed air flow corresponding to take-off and landing conditions over a mock-up aircraft wing. This helps improve aircraft's fuel efficiency, and stability.  |
| exhaust_system_33m  |            33            | Mixed       | SST k-omega turbulence simulating the flow and heat transfer in an exhaust manifold, measures flow structure and conservation of mass and energy.  Allows customers to understand and optimize the performance of their exhaust systems.                                                                                                    |
| f1_racecar_140m     |           140            | Hex-core    | Realizable k-epsilon turbulence model simulates the flow around a Formula 1 racing car, measures flow structure and conservation of mass. Allows F1 constructors and teams to have insights into aerodynamics, thermal management, and vehicle performance, thereby contributing to competitive advantage on the track.                                                                       |

### AMD Instinct MI300X GPU Test System Configuration

For the MI300X benchmarks, Ansys Fluent version 2024 R2 was executed using FP64 precision across 8 GPUs on a SuperMicro server. The Fluent Coupled Solver ran using the sedan_4m, aircraft_wing_14m, exhaust_system_33m, and f1_racecar_140m benchmark models.

| Information  |                       Specifications                      |
|------------------|--------------------------------------------------------------------------|
| Server Platform  |                       Supermicro AS -8125GS-TNMR2                        |
| GPUs             | AMD Instinct MI300X Platform (8x MI300X OAM GPUs 192GB HBM3, 750W each)  |
| CPUs             |                          2x AMD EPYC™ 9554 CPUs                          |
| Cores            |                               2x 64 cores                                |
| Numa Config      |                          1 NUMA node per socket                          |
| Memory           |                                   2TB                                    |
| Storage          |                                  877GB                                   |
| Host OS          |                            Ubuntu 22.04.4 LTS                            |
| Host GPU Driver  |                              ROCm™ 6.2.0-66                              |
| BIOS Version     |                      American Megatrends Inc. - 1.1                      |
| ROCm™ Version    |                                 6.1.0-48                                 |

[Learn more about the compute, memory, and networking capabilities of the MI300X platform in the AMD CNDA™ 3 Architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf).

### NVIDIA H100 GPU Test System Configuration

The NVIDIA H100 benchmarks ran on Ansys Fluent version 2024 R2,
executed using FP64 precision across 8 GPUs, using the Fluent Coupled
Solver on the sedan_4m, aircraft_wing_14m, exhaust_system_33m, and
f1_racecar_140m benchmark models.

| Information  |                       Specifications               |
|------------------|:----------------------------------------------:|
| Server Platform  |                 NVIDIA DGXH100                 |
| GPUs             | 8x NVIDIA H100 SXM GPUs 80GB HBM3 (700W each)  |
| CPUs             |           2x Intel Xeon 8480CL CPUs            |
| Cores            |                  2x 56 cores                   |
| Numa Config      |             1 NUMA node per socket             |
| Memory           |                      2TB                       |
| Storage          |                     1.8TB                      |
| Host OS          |               Ubuntu 22.04.4 LTS               |
| Host GPU Driver  |                   CUDA 12.2                    |
| BIOS Version     |                 96.00.61.00.01                 |

### Test Walkthrough

The following is a walkthrough of the code installation and configurations set as part of this benchmarking study. It includes
step-by-step instructions for replicating the MI300X and H100 results.

Single-Node Server Requirements

| CPUs | GPUs | Operating Systems | ROCm™ Driver | Container Runtimes |
|:---:|:---:|:---:|:---:|:---:|
| X86_64 CPU(s) | [AMD Instinct MI300A/X APU/GPU(s)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus) <br>  [AMD Instinct MI200 GPU(s)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus) <br> [AMD Instinct MI100 GPU(s)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus) | [Ubuntu](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems) <br> [RHEL](https://RHEL.com) <br> [SLES](https://SLES.com) <br> [Oracle Linux](https://www.oracle.com/linux/technologies/oracle-linux-downloads.html) | [ROCm Latest](https://rocm.docs.amd.com/en/latest/) | [Docker Engine  Singularity](https://docs.docker.com/engine/install/) |

For ROCm installation procedures and validation checks, see:

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Lab Notes ROCm installation notes](https://github.com/amd/amd-lab-notes/tree/release/rocm-installation)
- [ROCm Examples](https://github.com/amd/rocm-examples)

### Installing Ansys Fluent bare metal option

[*To build or install Fluent on a Docker container instead of bare metal, follow the instructions in the section Installing Ansys Fluent base container option below.*](### Installing Ansys Fluent base container option)

This installation guide
assumes that you have a license for Ansys Fluent and a tar with the Fluent application provided by Ansys. This example is using a tar with the name `fluent.24.2.lnamd64.tgz.`

Build System Requirements

- ROCm
- Mesa 3D Graphics libraries and XCB Utilities Libraries
- MPI Stack (optional)
  - OpenMPI/UCC/UCX
  - Cray MPI

Install Steps\
This example installs Ansys Fluent into `/opt`, but this is not
required.

Extract tar files

```shell
cd /opt
tar -xzvf fluent.24.2.lnamd64.tgz
```

Adding Fluent to PATH and setting environment variables

```shell
export PATH=/opt/ansys_inc/v242/fluent/bin:/opt/ansys_inc/v242/fluent/bench/bin:$PATH
export ANSGPU_OVERRIDE=1
```

Link Fluent Python into Path

```shell
ln -s /opt/ansys_inc/v242/commonfiles/CPython/3_10/linx64/Release/python/bin/python /usr/bin/python
```

or

```shell
sudo update-alternatives -set python /opt/ansys_inc/v242/commonfiles/CPython/3_10/linx64/Release/python/bin/python
```

Setup external MPI (optional)\
This example uses local Open MPI installed at `/opt/ompi`. To use
`CRAY MPI`, point it to the root of the install.

```shell
export OPENMPI_ROOT=/opt/ompi 
```

#### Ansys License

There are two license methods for Ansys Fluent. One of the following is required to run Ansys Fluent on a GPU node.

- License Server\
The `ANSYSLMD_LICENSE_FILE` environment variable should be set to the AnsysLMD server address that you use. `export
ANSYSLMD_LICENSE_FILE=1055@<AnsysLMD server IP address>`

- Temporary License\
This example uses a file `ansyslmd.ini` as the licensing file. This assumes Fluent is installed in `/opt/ansys_inc/`

```shell
cp ./ansyslmd.ini /opt/ansys_inc/shared_files/licensing/ 
ANSYSLI_ELASTIC=1 
 # SSL shenanigans (fixes elastic issue) 
capath=/etc/ssl/certs 
cacert=/etc/ssl/certs/ca-certificates.crt 
echo "capath=/etc/ssl/certs" >> $HOME/.curlrc 
echo "cacert=/etc/ssl/certs/ca-certificates.crt" >> $HOME/.curlrc 
cd /etc/ssl/certs && wget http://curl.haxx.se/ca/cacert.pem && ln -s cacert.pem ca-bundle.crt 
```

### Installing Ansys Fluent base container option

These instructions use Docker to create a container for Ansys Fluent. This container assumes that you have a license for Ansys Fluent and a tar with the Fluent application provided by Ansys ([see Ansys License section above](#### Ansys License)). These files are expected to be in a directory named `sources` in the docker build context.

This example is using a tar with the name `fluent.24.2.lnamd64.tgz.`

System Requirements

- Git
- Docker

Updating the Docker File\
Within the `Dockerfile` the default value for the `FLUENT_TAR` and `FLUENT_VERSION` can be hard coded before building or input at build time.

Ansys License\
There are two license methods for Ansys Fluent. At build time, the `ANSYSLMD_LICENSE_FILE` `build-arg` can be provided or
update the [Temporary License section](https://github.com/amd/InfinityHub-CI/blob/main/ansys-fluent/docker/Dockerfile#L62) by
uncommenting out the section and make sure the `ansyslmd.ini` is alongside the tar in the `sources` directory.

Inputs\
Possible `build-arg` for the Docker build command

&nbsp;&nbsp; IMAGE \
&nbsp;&nbsp; Default: `rocm_gpu:6.2.4` \
&nbsp;&nbsp; This container needs to be build using [Base ROCm GPU](https://github.com/amd/InfinityHub-CI/blob/main/base-gpu-mpi-rocm-docker/Dockerfile).

&nbsp;&nbsp; FLUENT_TAR \
&nbsp;&nbsp; Default: `fluent.24.2.lnamd64.tgz` \
&nbsp;&nbsp; This should reflect the tar file provided by Ansys. This file must be in the folder `sources` and this folder must be referenced at build time.

&nbsp;&nbsp; FLUENT_VERSION \
&nbsp;&nbsp; Default: `242` \
&nbsp;&nbsp; This is the numeric version of the Fluent version number. Eg: The example is 24.2, so use 242, as that is the reference in the Fluent tar.

&nbsp;&nbsp; ANSYSLMD_LICENSE_FILE (mandatory) \
&nbsp;&nbsp; If not using the Temporary License, This must be provided. This is the reference to the Ansys License Server/License required to run Ansys Fluent.

Building the container

Download the [Dockerfile](https://github.com/amd/InfinityHub-CI/blob/main/ansys-fluent/Dockerfile)

To run the default configuration:

```shell
docker build -t ansys/fluent:latest --build-arg ANSYSLMD_LICENSE_FILE=1234 -f /path/to/Dockerfile .
```

Notes:

- `ansys/fluent:latest` is an example container name.
- the `.` at the end of the build line is important. It tells Docker where your build context is located, the Ansys Fluent files should be relative to this path.
- `-f /path/to/Dockerfile` is only required if your docker file is in a different directory than your build context. If you are building in the same directory, it is not required.

To run a custom configuration, include one or more customized `build-arg` parameters.

*DISCLAIMER:* This Docker build has only been validated using the default values. Alterations may lead to failed builds if instructions are not followed.

```shell
docker build \
    -t fluent:latest \
    -f /path/to/Dockerfile \
    --build-arg IMAGE=rocm_gpu:6.2 \
    --build-arg FLUENT_TAR=fluent.24.2.lnamd64.tgz \
    --build-arg FLUENT_VERSION=242 \
    --build-arg ANSYSLMD_LICENSE_FILE=1055@127.0.0.1 \
    .  
```

#### Running an Application Container

This section describes how to launch the containers. It is assumed that up-to-versions of Docker and/or Singularity is installed on your system. If needed, please consult with your system administrator or view official documentation.

#### Docker

To run the container interactively, run the following command:

```shell
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt \
    seccomp=unconfined \
    -v /PATH/TO/FLUENT_TEST_FILES/:/benchmark \
    fluent:latest bash
```

User running container user must have permissions to `/dev/kfd` and `/dev/dri`. This can be achieved by being a member of `video` and/or `render` group.

Additional Parameters

- `-v [system-directory]:[container-directory]` will mount
    a directory into the container at run time.
- `-w [container-directory]` will designate what directory
    within a container to start in.
- This container is build with `OpenMPI`, to use `Cray
    MPICH`, it will need to be mount in over the OpenMPI
    installation. `-v [/absolute/path/to/mpich]/:/opt/ompi/`\
    Include any/all Cray environment variables necessary
    using `-e` for each variable\
    `-e MPICH_GPU_SUPPORT_ENABLED=1`

#### Singularity

Singularity, like Docker, can be used for running HPC containers. To create a Singularity container from your local Docker container, run the following command:

```shell
singularity build fluent.sif  docker-daemon://fluent:latest 
```

Singularity can be used similar to Docker to launch interactive and non-interactive containers, as shown in the following example of launching a interactive run

```shell
singularity shell --writable-tmpfs fluent.sif
```

- `--writable-tmpfs` allows for the file system to be writable, many benchmarks/workloads require this.
- `--no-home` will *not* mount the users home directory into the container at run time.
- `--bind [system-directory]:[container-directory]` will mount a directory into the container at run time.
- `--pwd [container-directory]` will designate what directory within a container to start in.
- This container is build with `OpenMPI`, to use `Cray MPICH`, it will need to be mount in over the OpenMPI installation. `-bind [/absolute/path/to/mpich]/:/opt/ompi/`\
    Include any/all Cray environment variables necessary using `--env` for each variable `--env MPICH_GPU_SUPPORT_ENABLED=1`

*For more details on Singularity please see their [User Guide](https://docs.sylabs.io/guides/3.7/user-guide/)*

### Downloading the workloads

Copy the `.tar` files to Ansys installation under `[path]/ansys_inc/v<version>/fluent`

```shell
tar -xvf <case>.tar
```

Repeat above steps for each individual case packages.

### Starting the Performance Test

- Utilize 8 MI300X GPUs to run the use cases sedan_4m, aircraft_wing_14m, exhaust_system_33m, and f1_racecar_140m:

```shell
ansys_inc/v242/commonfiles/CPython/3_10/linx64/Release/python/bin/python
ansys_inc/v242/fluent/bench/bin/fluent_benchmark_gpu.py -gpu.amd -cores 8 -cases
sedan_4m,aircraft_wing_14m,exhaust_system_33m,f1_racecar_140m -reorder_by_partition 1024 -mpi=openmpi
```

- Utilize 8 H100 GPUs to run the use cases sedan_4m, aircraft_wing_14m, exhaust_system_33m, and f1_racecar_140m:

```shell
ansys_inc/v242/commonfiles/CPython/3_10/linx64/Release/python/bin/python
ansys_inc/v242/fluent/bench/bin/fluent_benchmark_gpu.py -gpu -cores 8 -cases
sedan_4m,aircraft_wing_14m,exhaust_system_33m,f1_racecar_140m -reorder_by_partition 1024 -mpi=openmpi
```

### Troubleshooting

Use the verbose mode to obtain useful information about the execution of the commands above should you experience any issues. Simply add the flag `-verbose 1` at the end of either of the above commands.

Find further details about these installations at [AMD's Infinity Hub](https://github.com/amd/InfinityHub-CI/tree/main/ansys-fluent).

## Performance Highlights

The benchmarks results demonstrate that the AMD Instinct™ MI300X accelerator delivers an immediate, out-of-the-box performance gain over the Nvidia H100 across critical CFD Coupled Solver tasks. These benchmarks show up to 10% uplift in time-to-solution speed for models ranging from 4 million to 140 million cells, demonstrating MI300X’s superior efficiency and responsiveness in handling complex CFD operations.

![Performance Test Image](images/image1.png)

*Results for latency were calculated as the median of 5 test runs for each cells size and type combination.*

## Summary

The benchmarking results in this blog demonstrate that the AMD Instinct MI300X accelerators deliver an immediate, out-of-the-box performance gain over the Nvidia H100 GPUs across critical CFD Coupled Solver tasks. These benchmarks show up to 10% uplift in time-to-solution speed for models ranging from 4 million cells external car aerodynamics to 140 million cells detailed aerodynamics performance of a Formula 1 car, demonstrating the MI300X’s superior efficiency and responsiveness in handling complex CFD operations. The benchmarking results highlight the dramatic impact of the high memory bandwidth and capacity of the AMD Instinct MI300X accelerators and show how they can provide an advantage for applications requiring steady-state analysis.

MI300X is ready for deployment in environments requiring competitive latency, offering developers a platform that minimizes setup complexity while maximizing performance. For organizations seeking dependable, efficient, and scalable solutions for HPC CFD workloads, the AMD Instinct MI300X accelerators represent a compelling option that can deliver measurable value.

## Licensing Information

Your access and use of this application is subject to the terms of the applicable component-level license identified below. To the extent any subcomponent in this container requires an offer for corresponding source code, AMD hereby makes such an offer for corresponding source code form, which will be made available upon request. By accessing and using this application, you are agreeing to fully comply with the terms of this license. If you do not agree to the terms of this license, do
not access or use this application.

The application is provided in a container image format that includes the following separate and independent components:

|    Package    |                      License                      |                       URL                       |
|:-------------:|:-------------------------------------------------:|:-----------------------------------------------:|
| Ubuntu        | Creative Commons CC-BY-SA Version 3.0 UK License  | [Ubuntu Legal](https://ubuntu.com/legal)                                    |
| CMAKE         | OSI-approved BSD-3 clause                         | [CMake License](https://cmake.org/licensing/)                                   |
| OpenMPI       | BSD 3-Clause                                      | [OpenMPI License](https://www-lb.open-mpi.org/community/license.php) <br> [OpenMPI Dependencies Licenses](https://docs.open-mpi.org/en/v5.0.x/license/index.html)  |
| OpenUCX       | BSD 3-Clause                                      | [OpenUCX License](https://openucx.org/license/)                                 |
| OpenUCC       | BSD 3-Clause                                      | [OpenUCC License](https://github.com/openucx/ucc?tab=BSD-3-Clause-1-ov-file#readme)                                 |
| ROCm          | Custom/MIT/Apache V2.0/UIUC OSL                   | [ROCm Licensing Terms](https://rocm.docs.amd.com/en/latest/release/licensing.html)                            |
| Ansys Fluent  | Custom                                            | [Ansys Fluent](https://www.ansys.com/products/fluids/ansys-fluent)                                    |

### About The Data

Testing performed by AMD in May 2024.

Results (Median):

MI300X - sedan_4m, 82.67

H100 - sedan_4, 76.79

MI300X - aircraft_wing_14m, 71.01

H100 - aircraft_wing_14m, 67.67

MI300X -exhaust_system_33m, 128.24

H100 - exhaust_system_33m, 138.46

MI300X - f1_racecar_140m, 115.76

H100 f1_racecar_140m, 105.63

Server manufacturers may vary configurations, yielding different results. Performance may vary based on use of latest drivers and
optimizations.\
MI300-58

DISCLAIMER: The information contained herein is for informational
purposes only and is subject to change without notice. While every
precaution has been taken in the preparation of this document, it may
contain technical inaccuracies, omissions and typographical errors, and
AMD is under no obligation to update or otherwise correct this
information. Advanced Micro Devices, Inc. makes no representations or
warranties with respect to the accuracy or completeness of the contents
of this document, and assumes no liability of any kind, including the
implied warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein. No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document. Terms and limitations applicable to
the purchase or use of AMD products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and Conditions
of Sale. GD-18u.

COPYRIGHT NOTICE: © 2024 Advanced Micro Devices, Inc. All rights
reserved. AMD, the AMD Arrow logo, Instinct, ROCm, CDNA, and
combinations thereof are trademarks of Advanced Micro Devices, Inc.
Other product names used in this publication are for identification
purposes only and may be trademarks of their respective owners. Certain
AMD technologies may require third-party enablement or activation.
Supported features may vary by operating system. Please confirm with the
system manufacturer for specific features. No technology or product can
be completely secure.
