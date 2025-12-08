---
blogpost: true
blog_title: "GPU Partitioning Made Easy: Pack More AI Workloads Using AMD GPU Operator"
date: 1 Oct 2025
author: 'Alireza Sariaslani'
thumbnail: 'gpu-partitioning-thumbnail.png'
tags: AI/ML, Kubernetes, Performance, Serving
category: Software tools & optimizations
target_audience: For General Audience, but mainly targeted towards Kubernetes users
key_value_propositions: "What’s New in AMD GPU Operator: Learn About GPU Partitioning and New Kubernetes Features"
language: English
myst:
    html_meta:
        "author": "Alireza Sariaslani"
        "description lang=en": "What’s New in AMD GPU Operator: Learn About GPU Partitioning and New Kubernetes Features"
        "keywords": "AI/ML, Kubernetes, Performance, Serving"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "Software Developer, ML/AI Developer, HPC Developer, Data & Research Scientists"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_product_type": "Software & Applications"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training, Design, Simulation & Modeling, Deploying AI at Scale, Edge Computing, Generative AI"
        "amd_blog_topic_categories": 'Software & Ecosystem'
        "amd_blog_releasedate": Fri Oct 10, 12:00:00 PST 2025
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

# GPU Partitioning Made Easy: Pack More AI Workloads Using AMD GPU Operator

Modern AI workloads often don't utilize the full capacity of advanced GPU hardware, especially when running smaller models or during development phases. The AMD GPU partitioning feature addresses this challenge by allowing you to divide physical GPUs into multiple virtual GPUs, dramatically improving resource utilization and cost efficiency.

In this guide, we'll walk through setting up GPU partitioning on Kubernetes using the [AMD GPU Operator](https://github.com/ROCm/gpu-operator), deploy a sample workload, and demonstrate the benefits of this approach.

## System Requirements

- **AMD Instinct™ MI300 series** or newer GPUs
- **AMD ROCm™ 6.4** or later
- [Supported Linux Distributions](https://instinct.docs.amd.com/projects/gpu-operator/en/release-v1.3.1/#os-platform-support-matrix)

## Environment Setup

### 1. Kubernetes Installation

For this tutorial, we'll use [MicroK8s](https://microk8s.io/), a lightweight Kubernetes distribution ideal for development and testing. If you already have a Kubernetes cluster, skip to the [GPU Operator installation](#2-amd-gpu-operator-installation) or follow the [official helm installation guide](https://instinct.docs.amd.com/projects/gpu-operator/en/release-v1.3.1/installation/kubernetes-helm.html#pre-installation-steps).

```bash
# Clean up any existing installations
sudo snap remove microk8s

# Install MicroK8s
sudo snap install microk8s --classic --channel=1.34/stable

# Set up kubectl alias for convenience
echo "alias kubectl='microk8s kubectl'" >> ~/.bashrc
source ~/.bashrc

# Configure user permissions
sudo usermod -a -G microk8s $USER
mkdir -p ~/.kube
chmod 0700 ~/.kube

# Reload the user groups (reboot or run newgrp)
newgrp microk8s

# Verify cluster is ready
microk8s status --wait-ready
```

### 2. AMD GPU Operator Installation

AMD provides addon support to simplify GPU Operator deployment on MicroK8s.

```bash
# Enable community addons repository
microk8s enable community

# Install AMD GPU Operator
microk8s enable amd
```

This automatically deploys the AMD GPU operator stack in the `kube-amd-gpu` namespace:

```bash
# Verify successful installation
microk8s helm list -n kube-amd-gpu
```

### 3. Enable Device Config Manager

The Device Config Manager is essential for GPU partitioning. Enable it by patching the `deviceconfig` resource:

```bash
kubectl patch deviceconfig default \
  -n kube-amd-gpu \
  --type merge \
  -p '{
    "spec": {
      "configManager": {
        "enable": true
      }
    }
  }'
```

Verify all pods are running:

```bash
kubectl get pods -n kube-amd-gpu
```

## Deploy a Sample Workload

Let's deploy a lightweight LLM application to demonstrate the underutilization problem.

Create a file named `qwen.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qwen-25
  labels:
    app: qwen-25
spec:
  serviceName: qwen-25-svc
  replicas: 1
  selector:
    matchLabels:
      app: qwen-25
  template:
    metadata:
      labels:
        app: qwen-25
    spec:
      volumes:
      - name: cache-volume
        hostPath:
          path: /tmp/qwen-25-cache  # Update this path as needed
          type: DirectoryOrCreate
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "2Gi"
      containers:
      - name: qwen-25
        image: docker.io/rocm/vllm-dev:main
        command: ["/bin/sh", "-c"]
        args:
          - |
            # Extract ordinal from hostname for GPU assignment
            ORDINAL=${HOSTNAME##*-}
            echo "Setting HIP_VISIBLE_DEVICES to $ORDINAL"
            export HIP_VISIBLE_DEVICES=$ORDINAL

            # Start vLLM server
            vllm serve Qwen/Qwen2.5-1.5B-Instruct \
              --trust-remote-code \
              --enable-chunked-prefill \
              --max_num_batched_tokens 1024 \
              --tensor-parallel-size 1
        securityContext:
          privileged: true
        ports:
        - containerPort: 8000
        resources:
          requests:
            amd.com/gpu: "1"
          limits:
            amd.com/gpu: "1"
        volumeMounts:
        - mountPath: /root/.cache/huggingface
          name: cache-volume
        - name: shm
          mountPath: /dev/shm
---
apiVersion: v1
kind: Service
metadata:
  name: qwen-25-svc
spec:
  type: NodePort
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 30080
  selector:
    app: qwen-25
```

Deploy the application:

```bash
kubectl apply -f qwen.yaml 
```

Monitor the startup process:

```bash
# Monitor startup logs
kubectl logs -f qwen-25-0

# Test the API endpoint
curl http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Helsinki is",
    "max_tokens": 26,
    "temperature": 0
  }'
```

### Measure Baseline Performance

Run a benchmark to establish baseline performance:

```bash
kubectl exec -it qwen-25-0 -- python3 vllm/benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --backend openai \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --seed 12345
```

Monitor GPU utilization during the benchmark:

```bash
# In a separate terminal
rocm-smi
```

**Key Observation**: With `--tensor-parallel-size 1`, you'll likely notice the GPU underutilization. Even though the workload has exclusive access to the entire GPU, utilization remains low because this small model doesn't require such compute capacity. Additionally, you can only run up to 8 pods (1 per GPU) on a single node with the current configuration.

```bash
============================================= ROCm System Management Interface =============================================
======================================================= Concise Info =======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                    
============================================================================================================================
0       2     0x74a1,   28851  55.0°C      518.0W    NPS1, SPX, 0        1863Mhz  1300Mhz  0%   auto  750.0W  84%    73%   
1       3     0x74a1,   51499  35.0°C      131.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
2       4     0x74a1,   57603  41.0°C      140.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
3       5     0x74a1,   22683  35.0°C      136.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
4       6     0x74a1,   53458  40.0°C      136.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
5       7     0x74a1,   26954  33.0°C      138.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
6       8     0x74a1,   16738  35.0°C      139.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
7       9     0x74a1,   63738  35.0°C      144.0W    NPS1, SPX, 0        132Mhz   900Mhz   0%   auto  750.0W  0%     0%    
============================================================================================================================
=================================================== End of ROCm SMI Log ====================================================
```

**Tip**: See our [Workload Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) for strategies to fine-tune workloads and further improve performance and efficiency on AMD Instinct™ MI300X GPUs.

## GPU Partitioning

GPU partitioning allows you to divide each physical GPU into up to 8 virtual GPUs (depending on the mode), enabling up to 64 allocatable GPUs per node on supported hardware. This dramatically improves resource utilization for lightweight workloads.

### Understanding Partitioning Modes

AMD GPUs support various partitioning schemes. For a comprehensive overview, refer to our [previous blog post on compute and memory partitioning modes](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html). In this guide, we'll only focus on the Kubernetes implementation.

### Configuring GPU Partitioning

GPU partitioning is managed through Kubernetes ConfigMaps that define `gpu-config-profiles`. Each profile specifies how GPUs should be partitioned on a node. These profiles are applied to nodes using labels in the format `dcm.amd.com/gpu-config-profile=<Profile Name>`. The `device-config-manager` component automatically detects these labels and applies the corresponding GPU configuration to the hardware.

#### 1. Apply Configuration Profiles

Create a file named `gpu-config-profiles.yaml` or make your own following [this guide](https://instinct.docs.amd.com/projects/gpu-operator/en/release-v1.3.1/dcm/device-config-manager-configmap.html):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: config-manager-config
  namespace: kube-amd-gpu
data:
  config.json: |
    {
      "gpu-config-profiles":
      {
          "default":
          {
              "skippedGPUs": {
                  "ids": []
              },
              "profiles": [
                  {
                      "computePartition": "SPX", 
                      "memoryPartition": "NPS1",
                      "numGPUsAssigned": 8
                  }
              ]
          },
          "cpx-nps1":
          { 
              "skippedGPUs": {
                  "ids": []  
              },
              "profiles": [
                  {
                      "computePartition": "CPX",
                      "memoryPartition": "NPS1",
                      "numGPUsAssigned": 8
                  }          
              ]
          }
      }
    }
```

Apply the configuration:

```bash
kubectl apply -f gpu-config-profiles.yaml 

# Update device configuration to use this ConfigMap
kubectl patch deviceconfig default \
  -n kube-amd-gpu \
  --type merge \
  -p '{
    "spec": {
      "configManager": {
        "config": {
          "name": "config-manager-config"
        }
      }
    }
  }'
```

#### 2. Prepare the Node for Partitioning

Before applying partitioning, ensure the node is free of GPU workloads by adding a taint. The Config Manager already includes a toleration for the `amd-dcm`.

```bash
# Replace <NODE_NAME> with your actual node name
kubectl taint nodes <NODE_NAME> amd-dcm=up:NoExecute

# Wait for pods to be terminated. Verify that all GPU workloads have been evicted
kubectl describe node <NODE_NAME>
```

#### 3. Apply Partitioning Profile

We start with the default profile ([SPX + NPS1](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/index.html)), then switch to partitioned configuration:

```bash
# Replace <NODE_NAME> with your actual node name
kubectl label nodes <NODE_NAME> dcm.amd.com/gpu-config-profile=default --overwrite

# Monitor logs to confirm the new profile is applied
kubectl logs -n kube-amd-gpu -l app.kubernetes.io/name=device-config-manager -f

# Switch to a partitioned profile (CPX + NPS1)
kubectl label nodes <NODE_NAME> dcm.amd.com/gpu-config-profile=cpx-nps1 --overwrite

# Monitor logs to confirm the new profile is applied
kubectl logs -n kube-amd-gpu -l app.kubernetes.io/name=device-config-manager -f
```

**Troubleshooting**: If you encounter `AMDSMI_STATUS_BUSY - Device busy` errors:

```bash
# Check for active GPU processes
sudo systemctl list-units --type=service | grep -E "(gpu|amd|metrics)"

# Stop any GPU services if needed
# sudo systemctl stop <service-name>

# Check for processes using GPUs
amd-smi process

# Verify no pods are stuck terminating
kubectl get pods -A | grep Terminating
```

#### 4. Verify and Re-enable Workloads

```bash
# Remove the taint to allow workloads
kubectl taint nodes <NODE_NAME> amd-dcm=up:NoExecute-
  
# Verify GPU partitions are recognized
rocm-smi
kubectl describe nodes <NODE_NAME> | grep dcm.amd.com

# Check allocatable GPU resources
kubectl describe nodes <NODE_NAME> | grep -A 10 "Allocatable:"
```

**Expected Result**: You should now see up to 63 allocatable GPUs instead of the original 8 physical GPUs.

**Note**: In CPX mode, you may see 63 allocatable GPUs instead of 64. This is a known limitation in the current implementation.

## Scaling with Partitioned GPUs

With partitioning enabled, you can now scale your workloads much more efficiently. The deployment manifests remain unchanged — Kubernetes automatically allocates virtual GPU slices.

### Scaling to Multiple Replicas

You can increase the number of replicas to utilize more GPU partitions:

```bash
kubectl scale statefulset qwen-25 --replicas=10
```

You should see something like:

```bash
# kubectl get pods
NAME        READY   STATUS    RESTARTS   AGE
qwen-25-0   1/1     Running   0          2m58s
qwen-25-1   1/1     Running   0          17s
qwen-25-2   1/1     Running   0          16s
qwen-25-3   1/1     Running   0          15s
qwen-25-4   1/1     Running   0          14s
qwen-25-5   1/1     Running   0          13s
qwen-25-6   1/1     Running   0          12s
qwen-25-7   1/1     Running   0          11s
qwen-25-8   1/1     Running   0          10s
qwen-25-9   1/1     Running   0          8s
```

Test one of the new instances:

```bash
# Check the last pod to ensure it's running
kubectl logs -f qwen-25-9

# Run benchmark on the new instance
kubectl exec -it qwen-25-9 -- python3 vllm/benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --backend openai \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --seed 12345

# Monitor GPU utilization during the benchmark:
rocm-smi
```

**Key observations on the first two GPUs (index 0-15)**: 10 concurrent workloads running on just 2 physical GPUs versus the previous limitation of 1 workload per GPU, with each partition consuming approximately 1/8th of the memory and compute capacity. Notice that on GPU-2 (index 7-15) only one benchmark workload is active which uses ~11% GPU utilization with 2 models loaded in memory each taking 11% (22% total VRAM usage).

```bash
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                   
==========================================================================================================================
0       2     0x74a1,   4403   40.0°C      180.0W    NPS1, CPX, 0        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
1       3     0x74a1,   60722  40.0°C      180.0W    NPS1, CPX, 1        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
2       4     0x74a1,   43314  40.0°C      180.0W    NPS1, CPX, 2        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
3       5     0x74a1,   21811  40.0°C      180.0W    NPS1, CPX, 3        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
4       6     0x74a1,   8498   40.0°C      181.0W    NPS1, CPX, 4        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
5       7     0x74a1,   56627  40.0°C      181.0W    NPS1, CPX, 5        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
6       8     0x74a1,   39219  40.0°C      181.0W    NPS1, CPX, 6        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
7       9     0x74a1,   25906  40.0°C      181.0W    NPS1, CPX, 7        2106Mhz  900Mhz   0%   auto  750.0W  85%    0%    
8       10    0x74a1,   43179  65.0°C      331.0W    NPS1, CPX, 0        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
9       11    0x74a1,   21674  65.0°C      331.0W    NPS1, CPX, 1        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
10      12    0x74a1,   4266   65.0°C      325.0W    NPS1, CPX, 2        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
11      13    0x74a1,   60587  65.0°C      327.0W    NPS1, CPX, 3        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
12      14    0x74a1,   39082  65.0°C      327.0W    NPS1, CPX, 4        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
13      15    0x74a1,   25771  65.0°C      331.0W    NPS1, CPX, 5        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
14      16    0x74a1,   8363   65.0°C      331.0W    NPS1, CPX, 6        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
15      17    0x74a1,   56490  65.0°C      335.0W    NPS1, CPX, 7        2112Mhz  1300Mhz  0%   auto  750.0W  22%    11%   
============================================================================================================================
=================================================== End of ROCm SMI Log ====================================================
```

### Multi-Partition Workloads

For larger models that require more compute resources, you can allocate multiple GPU partitions to a single pod. This enables fitting bigger models while still leveraging the benefits of partitioning:

```yaml
resources:
  requests:
    amd.com/gpu: "8" # Request 8 GPU partitions
  limits:
    amd.com/gpu: "8"
```

Update your vLLM configuration to match the `tensor-parallel-size`.

### Performance Considerations

While partitioning enables higher density, consider these trade-offs:

1. Inter-partition communication overhead
2. Total GPU memory is shared across partitions
3. Best suited for lightweight workloads

## Summary

In this guide, we covered the essentials of GPU partitioning on Kubernetes using the AMD GPU Operator. We explored the benefits of partitioning and performance considerations. In our upcoming blogs, we'll dive deeper into these advanced topics:

- **Partitioning Strategies**: Comparison of SPX+NPS1 vs CPX+NPS4/NPS1 configurations with real-world workloads
- **Heterogeneous Configurations**: Running mixed partition styles on a single node
- **Advanced GPU Management**: Using `HIP_VISIBLE_DEVICES` for precise GPU allocation
- **Monitoring**: Integrating AMD metrics exporter with Prometheus and Grafana

## Additional Resources

- [AMD GPU Operator Documentation](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/)
- [AMD GPU Driver Documentation](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/index.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
