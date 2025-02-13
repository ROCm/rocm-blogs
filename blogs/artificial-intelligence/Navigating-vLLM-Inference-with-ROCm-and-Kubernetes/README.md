---
blogpost: true
blog_title: "Navigating vLLM Inference with ROCm and Kubernetes"
date: 13 Feb 2025
author: Alex He
thumbnail: 'Thumbnail_475.webp'
tags: AI/ML
category: Applications & models
language: English
target_audience: MLOps Engineers, vLLM Developers
key_value_propositions: Kubernetes is crucial for deploying workloads in data centers and even edge computing environments. This blog provides a guide on using the AMD ROCm K8s-device-plugin for deploying vLLM inference with detailed code examples. Developers can use this example as a reference to create other GPU job deployments beyond vLLM, such as those using SGlang or TGI Docker images, depending on their specific requirements.
myst:
  html_meta:
    "author": "Alex He"
    "description lang=en": "Quick introduction to Kubernetes (K8s) and a step-by-step guide on how to use K8s to deploy vLLM using ROCm on AMD GPUs"
    "keywords": "AI/ML"
    "property=og:locale": "en_US"
---

# Navigating vLLM Inference with ROCm and Kubernetes

[Kubernetes](https://kubernetes.io/) (often abbreviated as K8s) is an open-source platform designed for automating the deployment, scaling, and management of containerized applications. Developed by Google and now maintained by the Cloud Native Computing Foundation, Kubernetes enables developers to build, run, and manage applications across any infrastructure.

Key features of Kubernetes include:

- **Automation**: Automates application deployment, rolling updates, scaling, and maintenance.
- **Container Orchestration**: Manages containerized applications efficiently across a cluster of nodes.
- **Scalability**: Can scale applications up or down as needed based on demand.
- **Self-Healing**: Automatically replaces, restarts, or reschedules containers when they fail or become unresponsive.
- **Load Balancing**: Distributes traffic among container replicas to ensure even load distribution and high availability.

Kubernetes operates around a master-node architecture, where the master node contains control plane components that manage the cluster's state and the worker nodes run user applications in pods. This structure ensures high availability and reliability, as it allows for redundancy in managing the cluster.

In this blog, we will walk you through deploying a vLLM Inference Service by leveraging the power of ROCm and our [AMD K8s device plugin](https://github.com/ROCm/k8s-device-plugin). By following these steps, you'll be able to take advantage of Kubernetes to manage clusters equipped with powerful AI accelerators like AMD's Instinct MI300X, ideal for flexible AI inference workloads.

To learn more about the specifications and performance capabilities of AMD Instinct accelerators, visit [our product page](https://www.amd.com/en/products/accelerators/instinct.html).

## Setting up the K8s Cluster and vLLM

This blog is tailored for MLOps engineers experienced in Kubernetes (K8s) cluster deployments with AMD Instinct Accelerators and familiar with vLLM's LLM inference solutions. We assume a deep technical foundation for advanced infrastructure implementations.

If you're new to Kubernetes, start your journey by exploring the [K8s official documentation](https://kubernetes.io/docs/home/).

If you're new to vLLM, which is a widely-used LLM inference deployment solution, start learning from its official documentation: https://docs.vllm.ai/en/latest/

As an added benefit of AMD's close collaboration with vLLM, customers can leverage the latest insights and best practices for accelerating AI inference deployments. For the most up-to-date information on ROCm and vLLM, [check out these resources](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html).

This blog assumes you already have an existing Kubernetes cluster setup that is accessible to you to follow along with this tutorial. In addition, you will need to have already installed and loaded the AMD GPU driver on each of your GPU nodes.

## Install the k8s-device-plugin

 The device plugin will enable registration of AMD GPU to a container cluster.

```shell
kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
kubectl create -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-labeller.yaml
```

To verify that AMD GPUs are now schedulable as a Kubernetes resource on your cluster verify that you now see the new resource type `amd.com/gpu` listed for the GPU nodes in your cluster. The below command should return the number of GPUs you have available on each node.

```shell
kubectl get nodes -o custom-columns=NAME:.metadata.name,"GPUs Total:.status.capacity.amd\.com/gpu","GPUs Allocatable:.status.allocatable.amd\.com/gpu"
```

## Prepare the K8s yaml files

1. Secret is optional and only required for accessing gated models, you can skip this step if you are not using gated models.

    Here is the example `hf_token.yaml`

    ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
    name: hf-token-secret
    namespace: default
    type: Opaque
    data:
    token: "REPLACE_WITH_TOKEN"
    ```

    NOTE: you should use base64 to encode your HF TOKEN for the hf_token.yaml

    ```shell
    echo -n `<your HF TOKEN>` | base64
    ```

2. Define the deployment workload, `deployment.yaml`

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: mistral-7b
    namespace: default
    labels:
        app: mistral-7b
    spec:
    replicas: 1
    selector:
        matchLabels:
        app: mistral-7b
    template:
        metadata:
        labels:
            app: mistral-7b
        spec:
        volumes:
        # vLLM needs to access the host's shared memory for tensor parallel inference.
        - name: shm
            emptyDir:
            medium: Memory
            sizeLimit: "8Gi"
        hostNetwork: true
        hostIPC: true
        containers:
        - name: mistral-7b
            image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
            securityContext:
            seccompProfile:
                type: Unconfined
            capabilities:
                add:
                - SYS_PTRACE
            command: ["/bin/sh", "-c"]
            args: [
            "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
            ]
            env:
            - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
                secretKeyRef:
                name: hf-token-secret
                key: token
            ports:
            - containerPort: 8000
            resources:
            limits:
                cpu: "10"
                memory: 20G
                amd.com/gpu: "1"
            requests:
                cpu: "6"
                memory: 6G
                amd.com/gpu: "1"
            volumeMounts:
            - name: shm
            mountPath: /dev/shm
    ```
>> NOTE: The container image referenced in the deployment manifest may not be the latest version or the highest-performing build available at the time you view it. The latest vLLM image can be found in the "Getting Started" section of our [Instinct Performance Validation Documentation Page](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)

3. Define the `service.yaml`

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
    name: mistral-7b
    namespace: default
    spec:
    ports:
    - name: http-mistral-7b
        port: 80
        protocol: TCP
        targetPort: 8000
    # The label selector should match the deployment labels & it is useful for prefix caching feature
    selector:
        app: mistral-7b
    sessionAffinity: None
    type: ClusterIP
    ```

## Launch the pods

Next, let's enable the vLLM inference service using K8s.

```shell
kubectl apply -f hf_token.yaml  # Apply the Hugging Face token configuration to allow model downloading
kubectl apply -f deployment.yaml # Deploy the application by creating pods as defined in the deployment manifest
kubectl apply -f service.yaml    # Expose the deployed application as a service for vLLM inference requests 
```

## Test the service

Get the Service IP by

```shell
kubectl get svc
```

The mistral-7b is the service name. We can access the vllm serve by the CLUSTER-IP and PORT of it like,

Get models by (please use the real CLUSTER-IP of your environment)

```shell
curl http://<CLUSTER-IP>:80/v1/models
```

Do request

```shell
curl http://<CLUSTER-IP>:80/v1/completions   -H "Content-Type: application/json"   -d '{
        "model": "mistralai/Mistral-7B-v0.3",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
      }'
```

You may get response from the K8s vLLM inference service we setup above for your prompt question, "San Francisco is a" from the terminal like that,

```shell
 {"id":"cmpl-ede35e4d7f654d70a8a18e8560052251","object":"text_completion","created":1738647037,"model":"mistralai/Mistral-7B-v0.3","choices":[{"index":0,"text":" city that is known for its diversity","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null}}
```

## Summary

In this blog we've shown you, step-by-step, how to deploy and verify the vLLM service using Kubernetes (K8s) on AMD Instinct Accelerators. With this foundation in place, you're now ready to begin integrating vLLM into your real-world AI applications.
While the AMD K8s device plugin used in this tutorial is simple and great for single node or small Kubernetes implementations, for larger Kubernetes deployments please consider using the [AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/) which allows for automated installation of the K8s device plugin, node labeller, and AMD GPU device drivers on all your nodes without having to install them on each node manually. It also provides many other benefits such as GPU metrics reporting.

You may explore more examples of using K8s on AMD GPU:

- [Multinode Fine-Tuning of Stable Diffusion XL on AMD GPUs with Hugging Face Accelerate and OCI’s Kubernetes Engine (OKE)](https://rocm.blogs.amd.com/artificial-intelligence/sdxl-multinode-oke/README.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
