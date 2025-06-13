---
blogpost: true
blog_title: 'Announcing the AMD GPU Operator and Metrics Exporter'
thumbnail: '2025-01-20-gpu-op.jpg'
description: 'Announcing the AMD GPU Operator for Kubernetes and the Device Metrics Exporter'
date: 29 Jan 2025
author: Farshad Ghodsian, Matt Elliott
tags: Kubernetes
category: Software tools & optimizations
language: English
target_audience: Platform Engineers
key_value_propositions: 'Simplifies the deployment and management of AMD Instinct Accelerators on Kubernetes'
myst:
    html_meta:
        "description lang=en": "This post announces the AMD GPU Operator for Kubernetes and and the Device Metrics Exporter, including instructions for getting started with these new releases."
        "keywords": "Kubernetes"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
        "property=og:locale": "en_US"
---

# Announcing the AMD GPU Operator and Metrics Exporter

As AI workloads continue to grow in complexity and scale, we've consistently heard one thing from our customers: "Managing GPU infrastructure shouldn't be the hard part". For many, this is where Kubernetes comes into play. Kubernetes allows customers to easily manage and deploy their AI workloads at scale by providing a robust platform for automating deployment, scaling, and operations of application containers across clusters of hosts. It ensures that your applications run consistently and reliably, regardless of the underlying infrastructure. A pod is the smallest and simplest Kubernetes object. It represents a single instance of a running process in your cluster and can contain one or more containers. Pods are used to host your application workloads and are managed by Kubernetes to ensure they run as expected. Having pods be able to leverage GPUs on your cluster, however, is not something that is trivial.

That is until now! With today's announcement this is what we are hoping to change. Whether you're running a small Kubernetes research cluster or orchestrating thousands of GPUs across multiple data centers, the underlying infrastructure should just work.

We're taking a major step toward that vision with the release of the [AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/) and [AMD Device Metrics Exporter](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/). These tools represent more than just new software - they embody our commitment to making GPU-accelerated computing accessible and manageable at any scale.

Consider this scenario: You're scaling up your existing infrastructure from a few GPUs to hundreds. Suddenly, you're juggling driver installations across nodes, wondering about GPU health, and trying to figure out why some pods aren't getting scheduled correctly. Your team is spending more time managing infrastructure than innovating on their AI models.

In this blog post, you will learn how the AMD GPU Operator transforms the admin experience by automating the entire GPU lifecycle in your Kubernetes cluster. Everything from driver installation to monitoring is managed by the GPU Operator, allowing your team to focus on what matters: your applications.

## Key Features

The **AMD GPU Operator** brings several game-changing capabilities:

- **Zero-Touch GPU Setup**
  - Automatic ROCm driver management - build and install on demand or use pre-compiled versions
  - Integrated device plugin deployment for seamless GPU discovery and scheduling
  - Automatic node labeling for GPU capabilities

- **Enterprise-Ready Features**
  - Support for air-gapped installations and proxy environments
  - Flexible driver management, including support for pre-installed drivers
  - Automatic service provisioning for metrics collection
  - Ability to skip out-of-tree driver installation if drivers are already installed

</br>

Complementing this, the **Device Metrics Exporter** provides deep visibility into your GPU clusters:

- **Comprehensive Monitoring**
  - Real-time, Prometheus-compatible metrics for GPU utilization, memory usage, and power consumption
  - Integration with both Kubernetes and Slurm environments, including Job/Pod name
  - Pre-configured Grafana dashboards for immediate insights

- **Flexible Deployment Options**
  - Seamless integration with the GPU Operator
  - Standalone installation options for Kubernetes, Docker, and bare metal
  - Support for customized metrics fields and labels

For a full description of features see the [AMD GPU Operator Release Notes](https://instinct.docs.amd.com/projects/gpu-operator/en/main/releasenotes.html)

## Supported Hardware

| **GPUs** | |
| --- | --- |
| AMD Instinct™ MI300X | ✅ Supported |
| AMD Instinct™ MI250 | ✅ Supported |
| AMD Instinct™ MI210 | ✅ Supported |

## OS & Platform Support Matrix

Below is a matrix of supported Operating systems and the corresponding Kubernetes version that have been validated to work. We will continue to add more Operating Systems and future versions of Kubernetes with each release of the AMD GPU Operator and Metrics Exporter.

<table style="border-collapse: collapse; margin-left: 0; margin-right: auto;">
  <thead style="background-color: #2c2c2c; color: white;">
    <tr>
      <th style="border: 1px solid grey;">Operating System</th>
      <th style="border: 1px solid grey;">Kubernetes</th>
      <th style="border: 1px solid grey;">Red Hat OpenShift</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: white; color: black;">
      <td style="background-color: #2c2c2c; color: white; border: 1px solid grey;">Ubuntu 22.04 LTS</td>
      <td style="border: 1px solid grey;">1.29—1.31</td>
      <td style="border: 1px solid grey;"></td>
    </tr>
    <tr style="background-color: lightgrey; color: black;">
      <td style="background-color: #2c2c2c; color: white; border: 1px solid grey;">Ubuntu 24.04 LTS</td>
      <td style="border: 1px solid grey;">1.29—1.31</td>
      <td style="border: 1px solid grey;"></td>
    </tr>
    <tr style="background-color: white; color: black;">
      <td style="background-color: #2c2c2c; color: white; border: 1px solid grey;">Red Hat Core OS (RHCOS)</td>
      <td style="border: 1px solid grey;"></td>
      <td style="border: 1px solid grey;">4.16—4.17</td>
    </tr>
  </tbody>
</table>

## Red Hat OpenShift Certification

The AMD GPU Operator v1.1.0 marks a significant milestone, having successfully completed Red Hat OpenShift's rigorous [Operator Certification Process](https://connect.redhat.com/en/partner-with-us/what-are-operators). This certification validates that the GPU Operator meets strict security standards and Operator Framework requirements. It confirms that the operator is fully containerized, undergoes continuous vulnerability scanning, and is optimized for Red Hat OpenShift cluster administrators managing cluster-wide services.

The certified AMD GPU Operator is now available directly through the Red Hat OpenShift Catalog. You can find detailed information on the [AMD GPU Operator Catalog](https://catalog.redhat.com/software/container-stacks/detail/6722781e65e61b6d4caccef8) page.

AMD's continued collaboration and partnership with Red Hat ensures our operator will continue to be supported and evolve with each new version of Red Hat OpenShift.

## Quick Start Guide

Getting up and running with the AMD GPU Operator and Device Metrics Exporter on Kubernets is quick and easy. Below is a short guide on how to get started using the helm installation method on a standard Kubernetes install. Note that more detailed instructions along with other installation methods and configurations can be found on the [AMD GPU Operator Documentation Site](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/).

1. The GPU Operator uses [cert-manager](https://cert-manager.io/) to manage certificates for MTLS communication between services. If you haven't already installed `cert-manager` as a prerequisite on your Kubernetes cluster, you'll need to install it as follows:

    ```bash
    # Add and update the cert-manager repository
    helm repo add jetstack https://charts.jetstack.io --force-update

    # Install cert-manager
    helm install cert-manager jetstack/cert-manager \
      --namespace cert-manager \
      --create-namespace \
      --version v1.15.1 \
      --set crds.enabled=true
    ```

2. Once `cert-manager` is installed, you're just a few commands away from installing the GPU Operating and having a fully managed GPU infrastructure:

    ```bash
    # Add the Helm repository
    helm repo add rocm https://rocm.github.io/gpu-operator
    helm repo update

    # Install the GPU Operator
    helm install amd-gpu-operator rocm/gpu-operator-charts \
      --namespace kube-amd-gpu --create-namespace
    ```

3. You should now see the GPU Operator component pods starting up in the namespace you specified above, `kube-amd-gpu`. You will also notice that the `gpu-operator-charts-controller-manager`, `kmm-controller` and `kmm-webhook-server` pods are in a pending state. This is because you need to label a node in your cluster as the control-plane node for those pods to run on:

    ```bash
    # Label the control-plane node
    kubectl label nodes <node-name> node-role.kubernetes.io/control-plane=
    ```

4. To deploy the Device Plugin, Node Labeller and Metrics exporter to your cluster you need to create a new DeviceConfig custom resource. For a full list of configurable options refer to the [Full Reference Config](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/fulldeviceconfig.html) documenattion. An [example DeviceConfig](https://github.com/ROCm/gpu-operator/blob/release-v1.1.0/example/deviceconfig_example.yaml) is supplied in the ROCm/gpu-operator repository which can be used to get going:

    ```bash
    # Apply the example DeviceConfig to enable the Device Plugin, Node Labeller and Metrics Exporter plugins
    kubectl apply -f https://raw.githubusercontent.com/ROCm/gpu-operator/refs/heads/release-v1.1.0/example/deviceconfig_example.yaml
    ```

</br>
That's it! The GPU Operator components should now all be running and will automatically:

- Deploy and configure ROCm drivers across your nodes
- Install and configure the [Device Plugin](https://github.com/ROCm/k8s-device-plugin) for GPU scheduling
- Configure the Metrics Exporter for monitoring
- Label your nodes with GPU capabilities

<!-- Please see the [Kubernetes Installation Instructions](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/installation/kubernetes-helm.html) for prerequities, customization options, verification steps and an example workload. To get started with the AMD GPU Operator on OpenShift, visit the [OpenShift Installation Instructions](https://dcgpu.docs.amd.com/projects/gpu-operator/en/main/installation/openshift-olm.html). -->

## Standalone Metrics Collection

While the Device Metrics Exporter comes bundled with the GPU Operator, some will prefer to deploy it independently - particularly in bare metal environments without Kubernetes or when testing different monitoring configurations. Luckily the Metrics Exporter can be deployed standalone in a number of ways. Using Docker is the simplest way to get started with the Device Metrics Exporter.

Use this command to run the exporter via docker, which will expose metrics on port 5000:

<div style="margin-left: 20px;">

```bash
# Run the Device Metrics Exporter as a docker container in standalone mode
docker run -d \
  --device=/dev/dri \
  --device=/dev/kfd \
  -p 5000:5000 \
  --name device-metrics-exporter \
  rocm/device-metrics-exporter:v1.1.0
```

</div>

## Summary and Looking Ahead

The AMD GPU Operator and Device Metrics Exporter represent a significant step forward in simplifying the management of AMD Instinct Accelerators in your Kubernetes environments. By automating driver deployment and upgrades, GPU discovery, and metrics collection, these tools ease the complexity of managing GPU infrastructure at scale. This release is just the beginning. Some of our future roadmap features include:

- Advanced GPU Health Monitoring capabilities
- Automated Driver Lifecycle Management
- GPU Partitioning Support on Kubernetes

We're excited to see how you'll use these tools to transform your GPU infrastructure. Please visit our comprehensive documentation sites to learn more:

- **AMD GPU Operator**: [Documentation](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/) | [GitHub](https://github.com/ROCm/gpu-operator)
- **Device Metrics Exporter**: [Documentation](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/) | [GitHub](https://github.com/ROCm/device-metrics-exporter)
