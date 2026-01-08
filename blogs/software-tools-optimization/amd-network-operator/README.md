---
blogpost: true
blog_title: "Introducing the AMD Network Operator : Simplifying High-Performance Networking for AMD Platforms"
date: 8 Jan 2026
author: 'Sundara Murthy Gurunathan, Yuvarani Shankar, Shrey Ajmera'
thumbnail: 'AI-Networking.png'
tags: Kubernetes, AI/ML
category: Software tools & optimizations
target_audience: For General Audience, but mainly targeted towards Kubernetes users
key_value_propositions: The AMD Network Operator provides an automated, Kubernetes-native, end-to-end solution for deploying, configuring, monitoring, and validating AMD AI NIC networking for large-scale AI and HPC workloads—dramatically simplifying cluster operations while improving performance, reliability, and observability.
language: English
myst:
    html_meta:
        "author": "Sundara Murthy Gurunathan, Yuvarani Shankar, Shrey Ajmera"
        "description lang=en": "Introducing the AMD Network Operator for automating high-performance AI NIC networking in Kubernetes for AI and HPC workloads"
        "keywords": "AMD Network Operator, AMD AI NIC, RoCE Networking, Distributed Training, AI/ML Workloads"
        "vertical": "AI, HPC, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Pensando Network Infrastructure"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Sundara Murthy Gurunathan, Yuvarani Shankar, Shrey Ajmera"
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

# Introducing the AMD Network Operator v1.0.0: Simplifying High-Performance Networking for AMD Platforms

In this blog, you will learn how the AMD Network Operator simplifies high-performance networking for AMD GPU clusters, automates NIC discovery and configuration, supports RDMA/RoCE workloads, and provides real-time monitoring to keep your AI/ML and HPC jobs running efficiently.

In modern high-performance computing (HPC) and AI/ML workloads, efficient networking is as critical as the compute itself. Since the advent of using GPUs in combination with Kubernetes orchestration for AI/ML workloads over the last few years, processing large amounts of data has become significantly faster.

However, one of the critical bottlenecks in this process is often the networking infrastructure. AI/ML training and inference workloads require the rapid and reliable transfer of massive datasets. Even the most powerful GPUs can be underutilized without an optimized networking setup, leading to significant delays and inefficiencies.

The [AMD Network Operator](https://github.com/ROCm/network-operator) introduced recently, streamlines the discovery, deployment and management of  network resources such as AMD AI NICs, RDMA and network interfaces within Kubernetes clusters at scale, ensuring consistent and reliable network connectivity for distributed applications, AI/ML jobs and HPC workloads.

## What is the AMD Network Operator?

The AMD Network Operator simplifies the deployment and management of AMD AI NICs within Kubernetes clusters. The AMD Network Operator is based on the kubernetes operator framework  and facilitates dynamic network provisioning, RoCE support and RDMA network monitoring for applications running on AMD AI NIC nodes. The Network operator automates the entire AI NIC lifecycle including deployment, updates, and potential rollbacks, thus simplifying operational tasks for cluster administrators.

Working in coordination with the AMD GPU Operator, the Network Operator delivers high-bandwidth, low-latency networking across scale-out GPU computing environments.

Network Operator defines custom resources (CRs) that allow users to declare the desired state of their network configuration. The operator then watches these CRs and reconciles the actual state of the cluster to match the desired state.

The AMD Network Operator will soon be open-sourced and available on our [GitHub repo](https://github.com/ROCm/network-operator).

### Key Capabilities

- `NIC Discovery & Configuration`
  - Automatically detects AMD NICs and ensures they are configured correctly for RDMA/RoCE workloads.

- `Dynamic Resource Management`
  - The AMD Network Device Plugin allocates network resources effectively for pods running RCCL and MPI-based distributed jobs.

- `Device Metrics Exporter`
  - The exporter provides real-time Prometheus-compatible metrics for various network, RDMA and queue-pair statistics.
  - Preconfigured [Grafana dashboards](https://github.com/rocm/device-metrics-exporter/tree/main/grafana) provide comprehensive visibility across Cluster, Node and Job layers, helping teams monitor performance and identify bottlenecks efficiently.

- `Network Drivers`
  - Automatic AMD NIC driver management which dynamically builds and installs kernel drivers on demand or uses pre-compiled versions.
  - Ability to use inbox drivers (i.e., skip out-of-tree driver installation) if the required drivers are already present.

- `Container Network Interfaces (CNIs)`
  - Sets up secondary CNIs alongside the default Kubernetes CNI to manage specialized network configurations for AI workloads.
  - Open-source CNIs like Host-Device, SR-IOV, and RDMA integrate seamlessly, ensuring smooth operation across diverse networking setups

- `IP address management (IPAM)`
  - Allocates and manages IP addresses for the RDMA network components within the cluster
  - Open-source IPAMs including Whereabouts, DHCP, Host-Local, and Static IPAM plug in effortlessly, making IP management simple and consistent across Kubernetes networking layers

- `Node Feature Discovery (NFD)`
  - NFD automatically detects AMD NICs using PCI vendor and device IDs and  advertises the NIC features to Kubernetes using node labels.

- `Simplified installation`
  - The Network Operator Helm chart simplifies deployment of the operator and its components such as device plugin, network drivers, metrics exporter and CNIs, all  with default settings that can be easily customized via the chart’s values.yaml file during installation.

- `Cluster Validation`
  - Cluster Validation Framework is an optional add-on utility that builds on the functions provided by the Network Operator. It is designed to ensure that AMD GPU nodes, AI NIC networking, and RoCE configurations are set up correctly before running production workloads. This framework can also be leveraged for scheduling and orchestrating distributed AI and HPC workloads, ensuring that performance-verified nodes participate in large-scale compute jobs

For a full description of features see the AMD Network Operator [Release Notes](<https://instinct.docs.amd.com/projects/network-operator/en/release-v1.0.0/>)

## Supported Hardware, OS & Platform Compatibility

The AMD Network Operator is validated on AMD AI NIC hardware and modern Kubernetes platforms, ensuring reliable performance across supported operating systems and cluster versions.

### Hardware Support

| AMD AI NICs               | Status        |
|---------------------------|---------------|
| AMD Pollara AI NIC 400G   | ✅ Supported  |

### OS & Kubernetes Support

| Operating Systems | Kubernetes Versions |
|-------------------|---------------------|
| Ubuntu 22.04 LTS  | 1.29 – 1.34         |
| Ubuntu 24.04 LTS  | 1.29 – 1.34         |

## Quick Start Guide

Getting up and running with the AMD Network Operator on Kubernetes is quick and easy. Below is a short guide on how to get started using the helm installation method on a standard Kubernetes install. Note that more detailed instructions along with other installation methods and configurations can be found on [AMD Network Operator Docs](https://instinct.docs.amd.com/projects/network-operator/en/main/installation/kubernetes-helm.html)

### 1. Add AMD Helm Repository

Once `cert-manager` is installed, you are just a few commands away from installing the Network Operator and having a fully managed AMD AI NIC infrastructure

```bash
# Add the Helm repository
helm repo add rocm https://rocm.github.io/network-operator
helm repo update
```

### 2. Install the Network Operator

Basic installation:

```bash
helm install amd-network-operator rocm/network-operator-charts \
  --namespace kube-amd-network \
  --create-namespace \
  --version=v1.0.0
```

### 3. Install Custom Resource

You should now see the Network Operator component pods starting up in the `kube-amd-network` namespace.

To deploy the Network Device Plugin, Node Labeller, CNI Plugins and Network Metrics exporter to your cluster you need to create a new NetworkConfig custom resource. For a full list of configurable options refer to the [Full Reference Config](https://instinct.docs.amd.com/projects/network-operator/en/latest/fullnetworkconfig.html) documentation. An [example DeviceConfig](https://github.com/ROCm/network-operator/blob/main/example/networkconfig.yaml) is supplied in the ROCm/network-operator repository which can be used as reference.

```bash
kubectl apply -f https://raw.githubusercontent.com/ROCm/network-operator/refs/heads/release-v1.0.0/example/networkconfig.yaml
```

That's it! The Network Operator components should now all be running and will automatically:

- Deploy and configure AI NIC drivers across your nodes
- Install and configure the Network Device Plugin for AI NIC scheduling
- Install the CNI plugins for secondary network attachment
- Configure the Metrics Exporter for monitoring
- Label your nodes with AI NIC capabilities

### 4. RoCE Workload Image

Docker images to be used in workload pod creation are available at [ROCm DockerHub](https://hub.docker.com/r/rocm/roce-workload/tags)

These RoCE workload images bundle everything you need.

- Ubuntu base layers
- ROCm runtime
- MPI support for multi-GPU + multi-AINIC+ multi-node scaling
- RCCL
- AMD Network Plugin (ANP)
- RDMA-Core drivers and ibverbs libraries
- AI NIC Firmware Bundle user-space packages and libraries
- Full RCCL-tests binaries (all_reduce_perf, broadcast_perf, etc.)

These workload images are optimized for real cluster environments, making it ideal for:

- Cluster bring-up
- Pre-production validation
- Regression testing
- Benchmark-driven tuning
- Reproducing network edge cases

#### Understanding the Tag Format

RoCE workload image tags posted in the dockerhub typically follow this general structure:

```bash
<os-release>_<rocm-version>_<rccl-version>_<anp-version>_<ainic-fw-bundle-version>

Example tag: ubuntu24_rocm7_rccl-J13A-1_anp-v1.1.0-4D_ainic-1.117.1-a-63 
```

### 5. Deploying Distributed Workloads Using RoCE Workload Image

The AMD Network Operator simplifies AI NIC and AMD GPU cluster networking and workload deployment, including running RCCL benchmarks using prebuilt docker images from ROCm’s repository. You can use these images to validate multi-GPU, multi-AINIC communication and interconnect performance across your cluster.
For guidance on RoCE networking, and workload deployment, see the [documentation](https://instinct.docs.amd.com/projects/network-operator/en/main/installation/workload.html)

### 6. Run Sample Workloads like RCCL tests Using Cluster Validation Framework

Run MPIJobs using Cluster Validation Framework which deploy benchmark containers like RCCL tests, across your cluster to verify performance and connectivity.  The workloads can be customized and scheduled using [ConfigMap](https://github.com/ROCm/network-operator/blob/main/docs/_static/cluster-validation-config.yaml) and [CronJob](https://github.com/ROCm/network-operator/blob/main/docs/_static/cluster-validation-job.yaml) Kubernetes manifests

```bash
# Deploy Cluster Validation Job

kubectl apply -f cluster-validation-config.yaml
kubectl apply -f cluster-validation-job.yaml
```

See Cluster Validation [README](https://instinct.docs.amd.com/projects/network-operator/en/main/cluster_validation_framework/README.html) for full description

## Standalone Metrics Collection

While the Device Metrics Exporter is included with the Network Operator, it can also be deployed independently — useful for bare-metal setups or testing alternative monitoring configurations. The easiest way to get started is using Docker.

The following docker run command starts the exporter container in privileged mode with host networking and  mounting the nicctl binary from the host's AI NIC installation.

```bash
docker run -d        \
      --privileged   \
      --network=host \
      -v /usr/sbin/nicctl:/usr/sbin/nicctl   \
      --name network-device-metrics-exporter \
      rocm/device-metrics-exporter:nic-v1.0.0 -monitor-nic=true -monitor-gpu=false
```

See the updated list of exported AI NIC metrics [here](https://github.com/ROCm/device-metrics-exporter/blob/main/docs/configuration/network-metricslist.md)

## Summary

This blog outlined how the AMD Network Operator streamlines the management of AMD AI NICs in your Kubernetes environments. Its key features - automatic NIC discovery and configuration, RDMA/RoCE workload support, dynamic resource allocation and real-time monitoring - ensure that high-performance workloads run efficiently without the hassle of manual network setup.

By automating driver deployment and upgrades, AI NIC discovery, and metrics collection, these tools ease the complexity of managing AI NIC infrastructure at scale. This release is just the beginning. Some of our future roadmap features include:

- Automated creation of user-defined VF interfaces and RoCE profiles
- Local topology aware scheduling (co-location of GPUs & NICs)
- Network Operator support for RedHat OpenShift

Please visit our comprehensive documentation sites to learn more:

- AMD Network Operator: [Documentation](https://instinct.docs.amd.com/projects/network-operator/)
- Device Metrics Exporter: [Documentation](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/nic-v1.0.0/)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
