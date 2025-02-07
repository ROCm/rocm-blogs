---
blogpost: true
date: 07 Feb 2025
author: Victor Robles
blog_title: AI Inference Orchestration with Kubernetes on Instinct MI300X, Part 1
tags: Kubernetes, LLM, AI/ML, GenAI 
target_audience: Datacenter Administrators
key_value_propositions: This blog walks through setting up a Kubernetes cluster using MicroK8s, configuring Helm for streamlined deployments, implementing persistent storage for model caching, and installing the AMD GPU Operator. This is the first blog in a series that aims to provide a comprehensive, step-by-step guide for deploying and scaling AI inference workloads with Kubernetes, Grafana, Prometheus, vLLM, and the AMD GPU Operator on the AMD Instinct platform.
category: Software tools & optimizations
language: English
thumbnail: '2025-02-07-k8s-orch-pt1.jpg'
myst:
  html_meta:
    "description lang=en": "This blog is part 1 of a series aimed at providing a comprehensive, step-by-step guide for deploying and scaling AI inference workloads with Kubernetes and the AMD GPU Operator on the AMD Instinct platform"
    "keywords": "Kubernetes, AMD GPU Operator, MLOps, vLLM, Inferencing, ROCm, Mi210, MI250, MI300, AI/ML, Generative AI"
    "property=og:locale": "en_US"
---

# AI Inference Orchestration with Kubernetes on Instinct MI300X, Part 1

As organizations scale their AI inference workloads, they face the challenge of efficiently deploying and managing large language models across GPU infrastructure. This three-part blog series provides a production-ready foundation for orchestrating AI inference workloads on the AMD Instinct platform with Kubernetes.

In this post, we'll establish the essential infrastructure by setting up a Kubernetes cluster using MicroK8s, configuring Helm for streamlined deployments, implementing persistent storage for model caching, and installing the AMD GPU Operator to seamlessly integrate AMD hardware with Kubernetes.

Part 2 will focus on deploying and scaling the vLLM inference engine, implementing [MetalLB](https://github.com/metallb/metallb) for load balancing, and optimizing multi-GPU deployments to maximize the performance of AMD Instinct accelerators.

The series concludes in Part 3 with implementing Prometheus for metrics collection, Grafana for performance visualization, and [Open WebUI](https://github.com/open-webui/open-webui) for interacting with models deployed with vLLM.

Let's begin with Part 1, where we'll build the foundational components needed for a production-ready AI inference platform.

## MI300X Test System Specifications

|          |             |
|----------|-------------|
|**Node Type** | Supermicro AS -8125GS-TNMR2 |
|**CPU**   | 2x AMD EPYC 9654 96-Core Processor |
|**Memory** | 24x 96GB 2R ddr5-4800 dual rank |
|**GPU** | 8x MI300X |
|**OS** | Ubuntu 22.04.4 LTS |
|**Kernel** | 6.5.0-45-generic |
|**ROCm** | 6.2.0 |
|**AMD GPU driver** | 6.7.0-1787201.22.04 |

## Install Kubernetes (microk8s)

[Microk8s](https://microk8s.io/) is a lightweight, but powerful kubernetes instance that can run on as little as a single node and can scale up to mid-level cluster sizes. Here we run the snap install to load microk8s on the host node.

```bash
sudo apt update
sudo apt install snapd
sudo snap install microk8s --classic
```

Add the current user to the microk8s group and create the `.kube` directory in your home directory to store the microk8s config file.

```bash
sudo usermod -a -G microk8s $USER
mkdir -p ~/.kube
chmod 0700 ~/.kube
```

Register the `microk8s` group you just added in your current shell

```bash
newgrp microk8s
```

```{note}
For convenience you can add the following alias to your bashrc to use the native "kubectl" command with microk8s `echo "alias kubectl='microk8s kubectl'" >> ~/.bashrc; source ~/.bashrc`
```

Let's confirm our cluster is up and running.

```bash
kubectl get nodes
```

We should see the `STATUS` as `Ready`

```bash
NAME                 STATUS   ROLES    AGE   VERSION
mi300x-server01      Ready    <none>   5m   v1.31.3
```

Since this is a single-node instance of Kubernetes, we will need to label our node as a control-plane node in order for the AMD GPU Operator to successfully find the node.

```{note}
For vanilla Kubernetes installations this is not required, but if you're running Kubernetes on a single master node you will need to [remove the taint](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/) to be able to schedule jobs on this node
```

We can do this in one line

```bash
kubectl label node $(kubectl get nodes --no-headers | grep "<none>" | awk '{print $1}') node-role.kubernetes.io/control-plane=''
```

Let's confirm the node has been labeled correctly

```bash
kubectl get nodes
```

We should see the `ROLES` as `control-plane`

```bash
NAME                 STATUS   ROLES           AGE   VERSION
mi300x-server01      Ready    control-plane   8m   v1.31.3
```

## Install Helm

The [AMD GPU Operator](https://github.com/ROCm/gpu-operator), Grafana, and Prometheus installations are faciliated by using Helm charts. To install the latest version of Helm we will download the install script from the Helm repository and run it.

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

If you are utilizing microk8s as your Kubernetes instance, you also need to tell Helm where the `KUBECONFIG` environment variable is for microk8s. This is accomplished by exporting the environment variable into the `.bashrc` of the current user.

```bash
echo "export KUBECONFIG=/var/snap/microk8s/current/credentials/client.config" >> ~/.bashrc; source ~/.bashrc
```

```{note}
For vanilla Kubernetes installations this is not required unless you also wish to change the default location `$HOME/.kube/config`
```

## Persistent Storage

Next, we enable persistent storage so that we don't have to keep downloading the model when starting or scaling up an instance of the vLLM inference server. Here are methods for both microk8s and vanilla Kubernetes.

### Microk8s

Enable storage on microk8s with:

```bash
microk8s enable storage
```

Create a persistent volume claim (PVC) using `vllm-pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim  # Defines a PersistentVolumeClaim (PVC) resource
metadata:
  name: llama-3.2-1b  # Name of the PVC, used for reference in deployments
  namespace: default  # Kubernetes namespace where this PVC resides
spec:
  accessModes:
  - ReadWriteOnce  # Access mode indicating the volume can be mounted as read-write by a single node
  resources:
    requests:
      storage: 50Gi  # Specifies the amount of storage requested for this volume
  storageClassName: microk8s-hostpath  # Specifies the storage class to use (e.g., MicroK8s hostPath)
  volumeMode: Filesystem  # Indicates that the volume will be formatted as a filesystem
```

Apply the persistent volume claim in the console with:

```bash
kubectl apply -f vllm-pvc.yaml
```

```{note}
For microk8s, hostpath provisioner will store all volume data in `/var/snap/microk8s/common/default-storage` on the host machine.
```

### Vanilla Kubernetes

In environments without dynamic storage provisioning, define a `PersistentVolume` (PV) that the PVC can bind to with `pv.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: llama-3.2-1b-pv  # Name of the PersistentVolume
  namespace: default
spec:
  capacity:
    storage: 50Gi  # Amount of storage available in this PV
  accessModes:
  - ReadWriteOnce  # Access mode for the PV
  hostPath:
    path: /mnt/data/llama  # Path on the host where the data is stored
  volumeMode: Filesystem  # Indicates that the volume will be formatted as a filesystem
```

Define a `PersistentVolumeClaim` (PVC) with `vllm-pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-3.2-1b
  namespace: default
spec:
  accessModes:
  - ReadWriteOnce  # Access mode indicating the volume can be mounted as read-write by a single node
  resources:
    requests:
      storage: 50Gi  # Amount of storage requested
  volumeMode: Filesystem  # Indicates that the volume will be formatted as a filesystem
  ```

```{note}
The PVC will bind to the specific PV, based on matching `storage` and `accessModes` attributes.
```

Apply the two manifests:

```bash
kubectl apply -f pv.yaml
kubectl apply -f vllm-pvc.yaml
```
<!-- > **_Note:_** If you are using a vanilla installation of Kubernetes you can omit the storageClassName and k8s will use choose the default storage backend (e.g., hostPath or a cloud storage class). If no dynamic storage provisioner exists (e.g., when running Kubernetes on bare metal), you may need to define a [Persistent Volume manually](https://kubernetes.io/docs/concepts/storage/persistent-volumes/). -->

## Install the AMD GPU Operator

We proceed with installing the [AMD GPU Operator](https://github.com/ROCm/gpu-operator).  First we need to install [cert-manager](https://cert-manager.io/) as a prerequisite by adding the Jetstack repo to Helm.

```bash
helm repo add jetstack https://charts.jetstack.io
```

Then run the install for `cert-manager` via Helm.

```bash
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.15.1 \
  --set crds.enabled=true
```

Finally, install the AMD GPU Operator.

```bash
# Add the Helm repository
helm repo add rocm https://rocm.github.io/gpu-operator
helm repo update

# Install the GPU Operator
helm install amd-gpu-operator rocm/gpu-operator-charts \
  --namespace kube-amd-gpu --create-namespace
```

### Configure AMD GPU Operator

For our workload we will use a default ```device-config.yaml``` file to configure the AMD GPU Operator. This sets up the device plugin, node labeler, and metrics exporter to properly assign AMD GPUs to workloads, label the nodes that have AMD GPUs, and export metrics for Grafana and Prometheus. For a full list of configurable options refer to the [Full Reference Config](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/fulldeviceconfig.html) page of the AMD GPU Operator documentation.

```yaml
apiVersion: amd.com/v1alpha1
kind: DeviceConfig
metadata:
  name: gpu-operator
  # use the namespace where AMD GPU Operator is running
  namespace: kube-amd-gpu
spec:
  driver:
    # disable the installation of out-of-tree amdgpu kernel module
    enable: false

  devicePlugin:
    # Specify the device plugin image
    # default value is rocm/k8s-device-plugin:latest
    devicePluginImage: rocm/k8s-device-plugin:latest

    # Specify the node labeller image
    # default value is rocm/k8s-device-plugin:labeller-latest
    nodeLabellerImage: rocm/k8s-device-plugin:labeller-latest

    # Specify to enable/disable the node labeller
    # node labeller is required for adding / removing blacklist config of amdgpu kernel module
    # please set to true if you want to blacklist the inbox driver and use our-of-tree driver
    enableNodeLabeller: true
        
  metricsExporter:
    # To enable/disable the metrics exporter, disabled by default
    enable: true
    # kubernetes service type for metrics exporter, clusterIP(default) or NodePort
    serviceType: "NodePort"
    # internal service port used for in-cluster and node access to pull metrics from the metrics-exporter (default 5000)
    port: 5000
    # Node port for metrics exporter service, metrics endpoint $node-ip:$nodePort
    nodePort: 32500
    # exporter image
    image: "docker.io/rocm/device-metrics-exporter:v1.0.0"

  # Specify the node to be managed by this DeviceConfig Custom Resource
  selector:
    feature.node.kubernetes.io/amd-gpu: "true"
```

We apply the device-config file as:

```bash
kubectl apply -f device-config.yaml
```

To confirm the node labeler is working we run

```bash
kubectl get nodes -L feature.node.kubernetes.io/amd-gpu
```

and we should see `AMD-GPU` set as `true`

```bash
NAME                 STATUS   ROLES           AGE    VERSION   AMD-GPU
mi300x-server01      Ready    control-plane   15m   v1.31.3   true
```

To show available GPUs for workloads we can use:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,"Total GPUs:.status.capacity.amd\.com/gpu","Allocatable GPUs:.status.allocatable.amd\.com/gpu"
```

Now, we can see the total available GPUs

```bash
NAME             Total GPUs   Allocatable GPUs
mi300x-server01  8            8
```

## Summary

In this post, we've established a solid foundation for AI inference workloads by:

- Setting up a Kubernetes cluster with MicroK8s
- Configuring essential components like Helm
- Implementing persistent storage for model management
- Installing and validating the AMD GPU Operator

The next installment in this series will walk through deploying and scaling vLLM for inference, implementing MetalLB for load balancing, and optimizing multi-GPU deployments on AMD Instinct hardware. Part 3 will round out the series by deploying Open WebUI as a front end and configuring monitoring and management with Prometheus and Grafana. Stay tuned!

Disclaimers<br>
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
