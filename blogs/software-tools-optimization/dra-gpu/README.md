---
blogpost: true
blog_title: "Reimagining GPU Allocation in Kubernetes: Introducing the AMD GPU DRA Driver"
date: 13 Jan 2026
author: 'Nitish Bhat, Yan Sun, Shrey Ajmera'
thumbnail: 'dra_blog.png'
tags: Kubernetes, AI/ML
category: Software tools & optimizations
target_audience: For General Audience, but mainly targeted towards Kubernetes users
key_value_propositions: This blog highlights the launch of the AMD GPU DRA Driver and its integration with Kubernetes’ new Dynamic Resource Allocation framework. It demonstrates how AMD GPUs can now be requested, allocated, and managed natively through Kubernetes APIs — simplifying cluster operations and enabling next-generation scheduling capabilities.
language: English
myst:
    html_meta:
        "author": "Nitish Bhat"
        "description lang=en": "Explore how the AMD GPU DRA Driver brings declarative, attribute-aware GPU scheduling to Kubernetes — learn how to request and manage GPUs natively"
        "keywords": "GPU, ROCm, Kubernetes, Dynamic Resource Allocation, DRA, GPU Operator, GPU partitioning, topology-aware scheduling"
        "vertical": "Systems, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Nitish Bhat"
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

# Reimagining GPU Allocation in Kubernetes: Introducing the AMD GPU DRA Driver

In this blog, you’ll learn how Kubernetes’ new Dynamic Resource Allocation (DRA) framework and the [AMD GPU DRA Driver](https://github.com/ROCm/k8s-gpu-dra-driver) turn GPUs into first-class, attribute-aware resources. We’ll walk through how to publish AMD Instinct GPUs via ResourceSlices, request specific models and partition profiles with declarative ResourceClaims, and observe allocations through Kubernetes-native lifecycle objects, so you can simplify cluster operations compared to traditional Device Plugin–based setups.

## From Device Plugin to DRA: Why Kubernetes Needed a New Model

Kubernetes managed GPUs for years through the Device Plugin framework - a node-local system that handled simple "count-based" scheduling but couldn't express device details or relationships. Every GPU looked identical, and there was no clean way to request *two MI300X GPUs on the same PCIe root* or *two partitions from one parent device*, for example. As clusters grew more diverse, operators depended on node labels and vendor-specific logic just to approximate the placement they wanted.

The Dynamic Resource Allocation framework solves this by making devices first-class Kubernetes resources. DRA drivers publish **ResourceSlices** that expose structured attributes of devices - model, PCIe root, partition profile, memory, and more --- across the cluster. Workloads then create **ResourceClaims** that match those attributes through declarative CEL expressions, allowing the scheduler to select GPUs based on actual characteristics instead of blind counts.

For readers interested in the DRA design, see the official [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) documentation. The table below summarizes how the AMD GPU DRA Driver improves on the legacy Device Plugin model in terms of visibility, expressiveness, and lifecycle management.

| **Aspect** | **Device Plugin** | **AMD GPU DRA Driver** |
| --- | --- | --- |
| Resource visibility | Node-local only; opaque to scheduler | Cluster-visible via ResourceSlices |
| Request model | Count-based (amd.com/gpu: 2) | Attribute-based (e.g., productName == "MI300X") |
| Topology awareness | Node-local hints via GetPreferredAllocation | Topology modeled explicitly (pciRoot, partitionProfile, deviceID) |
| Complex constraints | Impossible to express declaratively | Supported via constraints (matchAttribute, distinctAttribute) |
| Cross-device allocation (GPU + NIC) | Not supported | Supported with multi-request claims |
| Lifecycle management | Ephemeral; tied to Pod | Persistent ResourceClaim objects |

## The AMD GPU DRA Driver

The **AMD GPU DRA Driver** is our DRA-compliant implementation that manages ROCm-based accelerators using this new model. It integrates with the Kubernetes control plane through the *gpu.amd.com* ResourceClass and exposes each GPU through a ResourceSlice containing structured attributes and capacities.

To illustrate, we have set up a Kubernetes cluster with two worker nodes - *gpu-worker-1* and *gpu-worker-2* - each hosting several AMD *Instinct MI300X* GPUs. The DRA driver running on each node advertises its GPUs by creating ResourceSlice objects in the cluster. The example below shows how one GPU on node *gpu-worker-2* appears in such a slice:

### Example: A GPU advertised via ResourceSlice

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceSlice
metadata:
  name: gpu-worker-2-gpu.amd.com-5fddf
spec:
  driver: gpu.amd.com
  nodeName: gpu-worker-2
  devices:
  - name: gpu-56-184
    attributes:
      cardIndex:
        int: 56
      deviceID:
        string: "8462659767828489944"
      driverVersion:
        version: 6.12.12
      family:
        string: AI
      partitionProfile:
        string: spx_nps1
      pciAddr:
        string: "0003:00:04.0"
      productName:
        string: AMD_Instinct_MI300X_OAM
      resource.kubernetes.io/pcieRoot:
        string: pci0003:00
      type:
        string: amdgpu
    capacity:
      computeUnits:
        value: "304"
      simdUnits:
        value: "1216"
      memory:
        value: 196592Mi
```

### Example: A Partitioned GPU advertised via ResourceSlice

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceSlice
metadata:
  name: gpu-worker-2-gpu.amd.com-5fddf
spec:
  driver: gpu.amd.com
  nodeName: gpu-worker-2
  devices:
  - name: gpu-4-132
    attributes:
      cardIndex:
        int: 4
      driverSrcVersion:
        string: CFBB20DF2696094F98A209B
      driverVersion:
        version: 6.14.14
      family:
        string: AI
      parentDeviceID:
        string: "16912319329091163297"
      parentPciAddr:
        string: "0002:00:01.0"
      partitionProfile:
        string: cpx_nps1
      productName:
        string: AMD_Instinct_MI300X_OAM
      renderIndex:
        int: 132
      resource.kubernetes.io/pcieRoot:
        string: pci0002:00
      type:
        string: amdgpu-partition
    capacity:
      computeUnits:
        value: "38"
      memory:
        value: 24574Mi
      simdUnits:
        value: "152"
```

Each device is fully described - PCI address, partition profile, driver version, and capacity details (memory, compute units, SIMD units) - giving the scheduler and user precise visibility into the GPU landscape.

### Requesting GPUs Declaratively

With the AMD GPU DRA Driver, users can finally express placement preferences that the scheduler can reason directly.

1. **Select MI300X GPUs with a specific partition profile**

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  namespace: gpu-test
  name: partitioned-gpu
spec:
  devices:
    requests:
    - exactly:
        deviceClassName: gpu.amd.com
        count: 1
        capacity:
          requests:
            gpu.amd.com/memory: 192Gi
        selectors:
        - cel:
          expression:
            device.attributes["gpu.amd.com"].productName == "AMD_Instinct_MI300X_OAM" &&
            device.attributes["gpu.amd.com"].partitionProfile == "spx_nps1"
```

No node selectors, no labels - Kubernetes matches against real device attributes.

2. **Allocate two GPU partitions from the same parent GPU**

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  namespace: gpu-test
  name: two-partitions-same-parent
spec:
  devices:
    constraints:
    - matchAttribute: "gpu.amd.com/deviceID"
      requests: ["gpu1", "gpu2"]
    requests:
    - name: gpu1
      exactly:
        deviceClassName: gpu.amd.com
        selectors:
        - cel:
            expression: device.attributes["gpu.amd.com"].type == "amdgpu-partition"
    - name: gpu2
      exactly:
        deviceClassName: gpu.amd.com
        selectors:
        - cel:
            expression: device.attributes["gpu.amd.com"].type == "amdgpu-partition"
```

The scheduler ensures both partitions come from the same parent GPU - something that, while implicitly handled by the Device Plugin, the user could not control or declare. With DRA, this intent is explicit: users can specify the relationship directly, and the allocation will fail if the constraint isn't met.

### GPU Allocation Lifecycle

Each GPU allocation in DRA is represented by a real Kubernetes object. When a Pod references a ResourceClaim, the driver allocates a matching device and records the result in the claim's status:

```console
$ kubectl get resourceclaim -n gpu-test pod1-gpu-7j2k4 -o yaml

apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  annotations:
    resource.kubernetes.io/pod-claim-name: gpu
  creationTimestamp: "2025-10-23T17:48:23Z"
  finalizers:
  - resource.kubernetes.io/delete-protection
  generateName: pod1-gpu-
  name: pod1-gpu-7j2k4
  namespace: gpu-test
  ownerReferences:
  - apiVersion: v1
    blockOwnerDeletion: true
    controller: true
    kind: Pod
    name: pod1
    uid: 3806c016-c81c-47ee-8650-ae71bfbefb89
  resourceVersion: "571"
  uid: 8e51bc7d-9cd5-452e-afcd-a63f3206a1e5
spec:
  devices:
    requests:
    - exactly:
        allocationMode: ExactCount
        count: 1
        deviceClassName: gpu.amd.com
      name: gpu
status:
  allocation:
    devices:
      results:
      - device: gpu-56-184
        driver: gpu.amd.com
        pool: k8s-gpu-dra-driver-cluster-worker
        request: gpu
    nodeSelector:
      nodeSelectorTerms:
      - matchFields:
        - key: metadata.name
          operator: In
          values:
          - k8s-gpu-dra-driver-cluster-worker
  reservedFor:
  - name: pod1
    resource: pods
    uid: 3806c016-c81c-47ee-8650-ae71bfbefb89
```

The status section shows the exact GPU assigned (gpu-56-184), the driver responsible, and the node where it resides. This makes GPU allocation *observable* and *auditable*.

### Sharing a GPU across containers in the same Pod

Multiple containers can use the same allocated GPU by referencing the same ResourceClaim.
In the example below, the init container runs *amd-smi* to verify GPU presence before the main app starts. Both containers use the same claim (*shared-gpu*), hence the same allocated device.

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  namespace: gpu-test
  name: shared-gpu
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly:
          deviceClassName: gpu.amd.com
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  namespace: gpu-test
  name: pod1
  labels:
    app: pod
spec:
  initContainers:
  - name: gpu-check
    image: docker.io/rocm/dev-ubuntu-24.04:latest
    command:
    - /bin/sh
    - -c
    - |
      echo "Checking for GPU availability..."
      if amd-smi list | grep -q '^GPU:'; then
        echo "GPU detected."
        exit 0
      fi
      echo "No GPU detected, exiting."
      exit 1
    resources:
      claims:
      - name: gpu
  containers:
  - name: ctr0
    image: docker.io/rocm/dev-ubuntu-24.04:latest
    command: ["bash", "-c"]
    args: ["export; trap 'exit 0' TERM; sleep 9999 & wait"]
    resources:
      claims:
      - name: gpu
  resourceClaims:
  - name: gpu
    resourceClaimTemplateName: shared-gpu
```

When this Pod runs, both containers see the same GPU device provided by
the claim. Once the Pod completes, the claim is automatically deallocated.

```{note}
When sharing a GPU across containers, users should ensure that their workloads coordinate access appropriately to avoid unintended concurrent use.
```

## Summary

In this blog, you saw how the AMD GPU DRA Driver turns GPUs into first-class, attribute-aware resources in Kubernetes. You learned how DRA publishes AMD Instinct GPUs as ResourceSlices, how to write ResourceClaims that target specific models, partition profiles, and memory sizes, and how the scheduler uses those attributes to place workloads on the right devices automatically.

For cluster operators and platform teams, you also saw how this model replaces sprawling node labels and custom affinity rules with a standardized DRA lifecycle. Resource discovery, attribute publication, validation, and CDI injection are all handled by the driver, and every GPU allocation is visible and auditable through Kubernetes APIs. The end result is a Kubernetes-native workflow for GPU scheduling that is simpler to operate, easier to debug, and better aligned with modern AI workloads.

Looking ahead, the AMD GPU DRA Driver also lays the foundation for a new generation of resource-aware scheduling in ROCm-based clusters. Building on today’s Kubernetes-driven partitioning, which is managed statically through the AMD GPU Operator and AMD Device Config Manager, ongoing work focuses on enabling *dynamic, workload-driven partitioning* for fractional GPU allocations, along with *cross-driver orchestration* with NICs and *cluster-level resource managers* that can coordinate device pools across multiple nodes in rack-scale deployments.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
