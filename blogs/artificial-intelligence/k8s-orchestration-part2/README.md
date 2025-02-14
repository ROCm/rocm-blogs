---
blogpost: true
date: 14 Feb 2025
author: Victor Robles
blog_title: AI Inference Orchestration with Kubernetes on Instinct MI300X, Part 2
tags: Kubernetes, LLM, AI/ML, GenAI 
target_audience: Datacenter Administrators
key_value_propositions: In this blog, we’ve taken a significant step toward building a scalable and high-performance AI inference solution by deploying and scaling the vLLM inference engine on Kubernetes, configuring MetalLB to provide efficient load balancing for your cluster, and optimizing deployments to fully leverage AMD Instinct multi-GPU capabilities.This is the second blog in a series that aims to provide a comprehensive, step-by-step guide for deploying and scaling AI inference workloads with Kubernetes, Grafana, Prometheus, vLLM, and the AMD GPU Operator on the AMD Instinct platform.
category: Software tools & optimizations
language: English
thumbnail: '2025-02-13-k8s-orch-pt2.jpg'
myst:
  html_meta:
    "description lang=en": "This blog is part 2 of a series aimed at providing a comprehensive, step-by-step guide for deploying and scaling AI inference workloads with Kubernetes and the AMD GPU Operator on the AMD Instinct platform"
    "keywords": "Kubernetes, AMD GPU Operator, vLLM, MLOps, Inferencing, ROCm, Mi210, MI250, MI300, AI/ML, Generative AI"
    "property=og:locale": "en_US"
---

# AI Inference Orchestration with Kubernetes on Instinct MI300X, Part 2

Welcome to Part 2 of our series on utilizing Kubernetes with the AMD Instinct platform! If you're just joining us, we recommend checking out **[Part 1](https://rocm.blogs.amd.com/artificial-intelligence/k8s-orchestration-part1/README.html)** where we covered setting up your Kubernetes cluster and enabling AMD GPU support.

In this post, we'll guide you through deploying and scaling the vLLM inference engine, implementing [MetalLB](https://github.com/metallb/metallb) for efficient load balancing, and optimizing multi-GPU deployments to maximize the performance of your AMD Instinct accelerators. By the end of this guide, we'll have a scalable and production-ready AI inference solution in place.

## Deploying vLLM

Let's start by deploying the vLLM inference engine using the `amd/Llama-3.2-1B-Instruct-FP8-KV` model. We'll leverage the AMD Instinct GPU with the persistent volume claim we established in **[Part 1](https://rocm.blogs.amd.com/artificial-intelligence/k8s-orchestration-part1/README.html)**. Here's our `vllm-deployment.yaml` file:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-3-2-1b  # Name of the deployment
  namespace: default  # Kubernetes namespace where the deployment resides
  labels:
    app: llama-3-2-1b  # Label to identify the app
spec:
  replicas: 1  # Number of replicas (instances) of the deployment
  selector:
    matchLabels:
      app: llama-3-2-1b  # Ensures the pods match this label for management
  template:
    metadata:
      labels:
        app: llama-3-2-1b  # Labels assigned to the pod for identification
    spec:
      volumes:
      - name: cache-volume  # Volume for persistent storage (e.g., caching model files)
        persistentVolumeClaim:
          claimName: llama-3.2-1b  # PersistentVolumeClaim that backs this volume
      # vLLM requires shared memory for tensor parallel inference.
      - name: shm  # Volume for shared memory (in-memory storage for high-speed access)
        emptyDir:
          medium: Memory  # Use memory as the storage medium
          sizeLimit: "2Gi"  # Limit shared memory size to 2GiB
      containers:
      - name: llama-3-2-1b  # Name of the container running inside the pod
        image: docker.io/rocm/vllm-dev:main  # Docker image for vLLM with ROCm GPU support
        command: ["/bin/sh", "-c"]  # Command to run in the container
        args: [
          "vllm serve amd/Llama-3.2-1B-Instruct-FP8-KV --chat-template /app/vllm/examples/tool_chat_template_llama3.2_json.jinja" # Command argument to serve the specified model
        ]        
        ports:
        - containerPort: 8000  # Port exposed by the container for inference requests
        resources:
          limits:
            cpu: "10"  # Maximum CPU resources the container can use
            memory: 20G  # Maximum memory the container can use
            amd.com/gpu: "1"  # Limit to 1 AMD GPU
          requests:
            cpu: "2"  # Minimum CPU resources requested for the container
            memory: 6G  # Minimum memory resources requested for the container
            amd.com/gpu: "1"  # Request 1 AMD GPU for this container
        volumeMounts:
        - mountPath: /root/.cache/huggingface  # Path where the cache volume is mounted
          name: cache-volume  # Reference to the cache volume defined earlier
        - name: shm  # Mount the shared memory volume
          mountPath: /dev/shm  # Path to shared memory inside the container

```

After creating this file, deploy it to your cluster with:

```bash
kubectl apply -f vllm-deployment.yaml
```

Once the deployment is available, we ping the cluster to determine how many GPUs are being utilized by our workload:

```bash
kubectl describe nodes  |  tr -d '\000' | sed -n -e '/^Name/,/Roles/p' -e '/^Capacity/,/Allocatable/p' -e '/^Allocated resources/,/Events/p'  | grep -e Name  -e  amd.com  | perl -pe 's/\n//'  |  perl -pe 's/Name:/\n/g' | sed 's/amd.com\/gpu:\?//g'  | sed '1s/^/Node Available(GPUs)  Used(GPUs)/' | sed 's/$/ 0 0 0/'  | awk '{print $1, $2, $3}'  | column -t
```

The output shows a single GPU in use:

```bash
Node                Available(GPUs)  Used(GPUs)
mi300x-server01     8                1
```

Monitor the progress of the deployment with:

```bash
kubectl get pods
```

The status will display `ContainerCreating` while Kubernetes is pulling the container and downloading the model.

```bash
NAME                            READY   STATUS              RESTARTS   AGE
llama-3-2-1b-6587cc94f9-lt9j8   0/1     ContainerCreating   0          7m44s
```

```{note}
Depending on the speed of your internet connection the image and model download can take a considerable amount of time.
```

Once the deployment is ready, the `STATUS` changes to `Running`.

```bash
NAME                            READY   STATUS              RESTARTS   AGE
llama-3-2-1b-6587cc94f9-lt9j8   1/1     Running             0          7m44s
```

## Expose the API

With vLLM up and running, our next step is making it accessible to end users. We'll accomplish this by creating a service that exposes our deployment externally. The following `vllm-service.yaml` manifest ensures that our Llama model is accessible in a scalable and manageable way:

### Create the service

First we create the `vllm-service.yaml` manifest.

```yaml
apiVersion: v1  # Defines the API version for the Service resource
kind: Service  # Specifies the resource type as a Service
metadata:
  name: llama-3-2-1b  # Name of the Service, used for referencing in other resources
  namespace: default  # Kubernetes namespace where the Service is created
spec:
  ports:
  - name: http-llama-3-2-1b  # A descriptive name for the port (optional but recommended)
    port: 80  # The external port that clients will use to connect to the service
    protocol: TCP  # The protocol for communication (default is TCP)
    targetPort: 8000  # The port on the pod where traffic will be forwarded
  selector:
    app: llama-3-2-1b  # Ensures that traffic is routed to pods with this label
  sessionAffinity: None  # No session stickiness; requests can be distributed across pods
  type: LoadBalancer  # Exposes the service externally with a LoadBalancer (e.g., cloud or MetalLB)
```

Next, apply the manifest:

```bash
kubectl apply -f vllm-service.yaml
```

After applying the service manifest, let's check its status:

```bash
kubectl get svc -n default
```

You'll notice the `EXTERNAL-IP` is in a pending state - this is expected since we haven't configured our load balancer yet:

```bash
NAME           TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
llama-3-2-1b   LoadBalancer   10.152.183.137   <pending>     80:31588/TCP   3m48s
```

## Install the load balancer

To resolve the pending external IP, we'll implement MetalLB as our load balancer. This powerful tool will not only assign external IPs but also enable us to efficiently distribute traffic across our scaled deployment.

### Install MetalLB

First, install the MetalLB Custom Resource Definitions:

```bash
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/main/config/manifests/metallb-native.yaml
```

This sets up the foundational resources required by MetalLB. Before proceeding further, confirm that the metalLB services are up and running. Check that their respective pods are functional:

```bash
kubectl get pods -n metallb-system
```

The output should show the metallb services `STATUS` as `Running`

```bash
NAME                         READY   STATUS    RESTARTS   AGE
controller-8b7b6bf6b-jffh9   1/1     Running   0          3m5s
speaker-db2pc                1/1     Running   0          3m5s
```

### Identify the correct network interface

To configure MetalLB, we need to identify the network interface that Kubernetes uses. Here's how to find it:

First, check which IP your Kubernetes API server is using:

```bash
kubectl get nodes -o wide
```

Look for the `INTERNAL-IP` column - this shows which network interface Kubernetes is using. Then, find the corresponding interface details:

```bash
ip addr | grep -A2 "<INTERNAL-IP-FROM-PREVIOUS-STEP>"
```

For example, if you see:

```bash
3: enp33s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP
    inet 10.216.70.211/22 brd 10.216.71.255 scope global enp33s0f0
```

This confirms `enp33s0f0` is our target interface with subnet `/22`.

### Calculate safe IP range

To avoid conflicts, follow these guidelines for choosing your IP range:

1. Identify the network range:
   - For our example /22 subnet (10.216.70.211):
   - Network range: 10.216.68.0 - 10.216.71.255
   - Usable IPs: 10.216.68.1 - 10.216.71.254

2. Choose a safe range by:
   - Staying away from the gateway (usually .1)
   - Avoiding existing node IPs
   - Using the upper end of the range (less likely to conflict)

For our example, we'll use: 10.216.71.230 - 10.216.71.240

```{warning}
Always verify your chosen IP range doesn't conflict with:
- Existing server/node IPs
- DHCP ranges
- Other network services
```

### Configure MetalLB

After determining the safe IP range, we setup a subset of IP addresses that can be assigned to our Kubernetes services by creating and applying a `metallb-config.yaml`.

```yaml
apiVersion: metallb.io/v1beta1  # Specifies the MetalLB API version
kind: IPAddressPool  # Defines a pool of IP addresses to be managed by MetalLB
metadata:
  name: default  # Name of the IP address pool
  namespace: metallb-system  # Namespace where MetalLB resources are managed
spec:
  addresses:
  - 10.216.71.230-10.216.71.240  # Range of IP addresses available for allocation
---
apiVersion: metallb.io/v1beta1  # Specifies the MetalLB API version
kind: L2Advertisement  # Configures Layer 2 (ARP/NDP) advertisement for the IP pool
metadata:
  name: l2-config  # Name of the L2Advertisement resource
  namespace: metallb-system  # Namespace where MetalLB resources are managed
spec: {}  # Empty spec indicates default configuration for Layer 2 advertisement
```

Don't forget to apply the config!

```bash
kubectl apply -f metallb-config.yaml
```

### Confirm vLLM service availability

With the load balancer now in place, we can check to see if the service has been assigned an external address,

```bash
kubectl get svc
```

which should result in an IP address under `EXTERNAL-IP`

```bash
NAME           TYPE           CLUSTER-IP       EXTERNAL-IP     PORT(S)        AGE
llama-3-2-1b   LoadBalancer   10.152.183.137   10.216.71.230   80:31588/TCP   17m
```

We can now send a request payload to our vLLM engine with the `curl` command and prettify the output by piping to `jq`. In a terminal on the host we run:

```bash
curl http://10.216.71.230/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "amd/Llama-3.2-1B-Instruct-FP8-KV",
        "prompt": "San Francisco is a",
        "max_tokens": 8,
        "temperature": 0
      }' | jq .
```

```{note}
Be sure the model name matches the model you are utilizing and IP address matches the IP address your load balancer assigned to your service.
```

You should see a response payload similar to the one below, with the text generated by the model under the `choices` JSON object.

```json
{
  "id": "cmpl-fec47d49ee844e7390d5aeed317e05d2",
  "object": "text_completion",
  "created": 1733849007,
  "model": "amd/Llama-3.2-1B-Instruct-FP8-KV",
  "choices": [
    {
      "index": 0,
      "text": " city that is known for its beautiful beaches",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 13,
    "completion_tokens": 8,
    "prompt_tokens_details": null
  }
}
```

## Scaling our deployment

Now that we have our vLLM service running smoothly with load balancing in place, let's leverage the full power of our AMD Instinct GPUs. Since the Llama 3.2 1B model comfortably fits on a single GPU, we can scale our deployment to utilize all available accelerators.

```{note}
Be mindful of the total number of CPUs and RAM on your machine. You might have to reconfigure the `resources` section of the `vllm-deployment.yaml` manifest so as to not exceed your machine's number of CPUs and max RAM.
```

### Scaling

We scale our deployment with:

```bash
kubectl scale -n default deployment llama-3-2-1b --replicas=8
```

The `get pods` command is used to confirm the new instances of our vLLM deployment have been replicated

```bash
get pods
NAME                            READY   STATUS    RESTARTS   AGE
llama-3-2-1b-6587cc94f9-7c2pd   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-dlkjr   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-gwknm   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-lt9j8   1/1     Running   0          32m
llama-3-2-1b-6587cc94f9-mvtqm   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-ptt7g   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-s7w52   1/1     Running   0          5s
llama-3-2-1b-6587cc94f9-zppx2   1/1     Running   0          5s
```

Now when we send API requests to vLLM, our load balancer will decide which pod to send it to by utilizing the Layer 2 networking capabilities of MetalLB.

## Summary

In this post, we've made significant progress in building a scalable and high-performance AI inference solution by:

- Deploying and scaling the vLLM inference engine on Kubernetes.
- Implementing MetalLB for robust load balancing.
- Optimizing deployments to harness the full power of AMD Instinct multi-GPU capabilities.

Stay tuned for the third and final part of this series, where we'll enhance our setup with advanced monitoring and management tools. We'll walk through configuring Prometheus for GPU performance monitoring, setting up Grafana for insightful visualizations, and implementing Open WebUI for user-friendly model interactions.

<p style="font-size: 12px;">
Disclaimers<br>
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
</p>
