---
blogpost: true
blog_title: 'Multinode Fine-Tuning of Stable Diffusion XL on AMD GPUs with Hugging Face Accelerate and OCI's Kubernetes Engine (OKE)'
thumbnail: './images/Multinode.jpeg'
date: 15 October 2024
author: Douglas Jia
tags: AI/ML, Fine-Tuning, GenAI, Diffusion Model
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "This blog demonstrates how to set-up and fine-tune a Stable Diffusion XL (SDXL) model in a multinode Oracle Cloud Infrastructure’s (OCI) Kubernetes Engine (OKE) on a cluster of AMD GPUs using ROCm"
    "author": "Douglas Jia"
    "keywords": "Fine-tuning, Stable Diffusion XL, Multinode training, OCI, OKE, Kubernetes, Accelerate, RoCE, Distributed training, Generative AI, AMD, GPU, MI300, MI210, ROCm"
    "property=og:locale": "en_US"
---

# Multinode Fine-Tuning of Stable Diffusion XL on AMD GPUs with Hugging Face Accelerate and OCI's Kubernetes Engine (OKE)

As the scale and complexity of generative AI and deep learning models grow, multinode training, basically dividing a training job across several processors, has become an essential strategy to speed up training and fine-tuning processes of large generative AI models like SDXL. By distributing the training workload across multiple GPUs on multiple nodes, multinode setups can significantly accelerate the training process.
In this blog post we will show you, step-by step, how to set-up and fine-tune a [Stable Diffusion XL (SDXL) model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) in a multinode Oracle Cloud Infrastructure’s (OCI) [Kubernetes Engine (OKE)](https://docs.oracle.com/en-us/iaas/Content/ContEng/Concepts/contengoverview.htm) on AMD GPUs using ROCm.

## Getting things ready: OCI OKE, RoCE, SDXL and Hugging Face Accelerate

This blog post will guide you through the set-up and fine-tuning of an SDXL model on a multinode setup with 8 GPUs on each node. A working configuration file is provided on this blog's GitHub repository, along with instructions on how to customize the setup to meet your specific workload requirements.

[Oracle Kubernetes Engine (OKE)](https://docs.oracle.com/en-us/iaas/Content/ContEng/Concepts/contengoverview.htm) provides a robust and flexible platform for deploying, managing, and scaling containerized applications in the Oracle Cloud. It allows users to easily orchestrate multinode setups, making it an ideal environment for training large models on multiple AMD processors.

A critical aspect of our setup involves using [RDMA over Converged Ethernet (RoCE)](https://github.com/oracle-quickstart/oci-hpc-oke) for internode communication. RoCE offers significant benefits over traditional Ethernet connections. RoCE reduces latency and increases data transfer speeds between nodes, leading to improved overall performance in distributed training scenarios.

[Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) is a generative AI model designed for high-quality image synthesis with text prompts. Fine-tuning this model involves adapting it to specific tasks or datasets, tailoring its image styles to match those of the dataset.

In this post we will use the [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) dataset from Hugging Face. This dataset contains images originally obtained from [Narutopedia](https://naruto.fandom.com/wiki/Narutopedia), which were captioned using the pre-trained BLIP model. [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index) is used in this process to simplify and optimize training.

Keep in mind that while this blog provides an example for setting up and fine-tuning the SDXL model in a multinode environment on OCI, a similar approach can be used for setting up and fine-tuning other generative AI models.

>**Note:** This blog focuses on setting up multinode fine-tuning, which can be easily adapted for pre-training, rather than on performance studies of multinode setups.

## Implementation

We implemented the multinode fine-tuning of SDXL on an OCI cluster with multiple nodes. Each node contains 8 AMD MI300x GPUs, and you can adjust the number of nodes based on your available resources in the scripts we will walk you through in the following section.

Your host machine will interact with the OKE cluster using `kubectl` commands. To ensure that `kubectl` interacts with the correct Kubernetes cluster, set the `KUBECONFIG` environment variable to point to the location of the configuration file (**Be sure to update the path** to point to your specific config file):

```bash
# Please update the path to your own config file.
export KUBECONFIG=<your_own_config_file>
```

Because [Weights & Biases (`wandb`)](https://wandb.ai/site) will be used to track the fine-tuning progress and a Hugging Face dataset will be used for fine-tuning, you will need to generate an OKE "secret" using a `wandb` API key and a Hugging Face token. An OKE secret is a Kubernetes object used to securely store and manage sensitive information such as passwords, tokens, and SSH keys. An OKE secret allows confidential data to be passed to your pods and containers securely.

```bash
# Create a secret for the WANDB API Key
kubectl create secret generic wandb-secret --from-literal=WANDB_API_KEY=<Your_wandb_API_key>
# Create a secret for the Hugging Face token
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<Your_Hugging_Face_token>
```

Create a Kubernetes ConfigMap by downloading the Hugging Face configuration file, `default_config_accelerate.yaml`, from [this blog's GitHub repository `src` folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/sdxl-multinode-oke) to your working directory on your host machine and runing the command below:

```bash
kubectl create configmap accelerate-config --from-file=default_config_accelerate.yaml
```

A Kubernetes ConfigMap stores configuration information in key-value pairs. The command above will create a ConfigMap that contains the `default_config_accelerate.yaml` file. This will allow your pods to access and use the Hugging Face Accelerate configuration.

Download `accelerate_blog_multinode.yaml` from [this blog's GitHub repository `src` folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/sdxl-multinode-oke) to your host machine and adjust the paths in the file to align with your actual file system.

Start the fine-tuning process by running the command below:

```bash
kubectl apply -f accelerate_blog_multinode.yaml 
```

If everything is set up correctly, you'll see the following output:

```output
service/sdxl-headless-svc created
configmap/sdxl-finetune-multinode-config created
job.batch/sdxl-finetune-multinode created
```

You can then monitor the fine-tuning progress via the `wandb` dashboard or other [supported reporting and logging platforms](https://github.com/huggingface/diffusers/blob/b5f591fea843cb4bf1932bd94d1db5d5eebe3298/examples/text_to_image/train_text_to_image_sdxl.py#L443).

The `accelerate_blog_multinode.yaml` file is organized into three sections, Service, ConfigMap, and Job, separated by dashes (`---`). Breaking down the `accelerate_blog_multinode.yaml` file into these three sections reveals how Kubernetes orchestrates the different components necessary for the multinode fine-tuning of SDXL. This modular approach provides flexibility and simplifies the management of complex workloads, especially in a multinode environment.

1. The Service section specifies how the application running within the pods should be exposed as a service both within the cluster and externally. This includes the service name and type, and the ports the service will use for communication. This setup ensures robust inter-node communication within the cluster.

   <details>
   <summary> Service section of the yaml file (click to expand)</summary>

   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: sdxl-headless-svc
   spec:
     clusterIP: None
     ports:
     - port: 12342
       protocol: TCP
       targetPort: 12342
     selector:
       job-name: sdxl-finetune-multinode
   ```

   </details>

2. The ConfigMap section provides additional key-value pairs for inter-node communication. Storing these settings in this section makes it easier to manage and update them without altering the container images or pod specifications.

   <details>
   <summary> ConfigMap section of the yaml file (click to expand)</summary>

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: sdxl-finetune-multinode-config
   data:
     headless_svc: sdxl-headless-svc
     job_name: sdxl-finetune-multinode
     master_addr: sdxl-finetune-multinode-0.sdxl-headless-svc
     master_port: '12342'
     num_replicas: '3'
   ```

   </details>

3. The Job section provides details about how the fine-tuning process. This includes information such as which container image to use, which resources to allocate to the job, which command to run within the container, and how the ConfigMap for the Accelerate settings should be mounted into the container.

   <details>
   <summary> Job section of the yaml file (click to expand)</summary>

   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: sdxl-finetune-multinode
   spec:
     backoffLimit: 0
     completions: 3
     parallelism: 3
     completionMode: Indexed
     template:
       metadata:
         labels:
           job: sdxl-multinode-job
       spec:
         hostNetwork: true
         dnsPolicy: ClusterFirstWithHostNet
         containers:
           - name: accelerate-sdxl
             image: rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.1.2
             securityContext:
               privileged: true
               capabilities:
                 add: [ "IPC_LOCK" ]
             env:
             - name: HIP_VISIBLE_DEVICES
               value: "0,1,2,3,4,5,6,7"
             - name: HIP_FORCE_DEV_KERNARG
               value: "1"
             - name: GPU_MAX_HW_QUEUES
               value: "2"
             - name: USE_ROCMLINEAR
               value: "1"
             - name: NCCL_SOCKET_IFNAME
               value: "rdma0"
             - name: MASTER_ADDRESS
               valueFrom:
                 configMapKeyRef:
                   key: master_addr
                   name: sdxl-finetune-multinode-config
             - name: MASTER_PORT
               valueFrom:
                 configMapKeyRef:
                   key: master_port
                   name: sdxl-finetune-multinode-config
             - name: NCCL_IB_HCA
               value: "mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"
             - name: HEADLESS_SVC
               valueFrom:
                 configMapKeyRef:
                   key: headless_svc
                   name: sdxl-finetune-multinode-config
             - name: NNODES
               valueFrom:
                 configMapKeyRef:
                   key: num_replicas
                   name: sdxl-finetune-multinode-config
             - name: NODE_RANK
               valueFrom:
                 fieldRef:
                   fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
             - name: WANDB_API_KEY
               valueFrom:
                 secretKeyRef:
                   name: wandb-secret
                   key: WANDB_API_KEY
             - name: HF_TOKEN
               valueFrom:
                 secretKeyRef:
                   name: hf-secret
                   key: HF_TOKEN
             volumeMounts:
               - mountPath: /mnt
                 name: model-weights-volume
               - mountPath: /etc/config
                 name: diffusers-config-volume
               - { mountPath: /dev/infiniband, name: devinf }
               - { mountPath: /dev/shm, name: shm }
             resources:
               requests:
                 amd.com/gpu: 8 
               limits:
                 amd.com/gpu: 8 
             command: ["/bin/bash", "-c", "--"]
             args:
               - |
                 # Clone the GitHub repo
                 git clone --recurse https://github.com/ROCm/bitsandbytes.git
                 cd bitsandbytes
                 git checkout rocm_enabled
                 # Install dependencies
                 pip install -r requirements-dev.txt
                 # Use -DBNB_ROCM_ARCH to specify target GPU arch
                 cmake -DBNB_ROCM_ARCH="gfx942" -DCOMPUTE_BACKEND=hip -S .
                 make
                 pip install .
                 cd .. 
   
                 # Set up Hugging Face authentication using the secret
                 mkdir -p ~/.huggingface
                 echo $HF_TOKEN > ~/.huggingface/token
                 
                 pip install deepspeed==0.14.5 wandb
                 git clone https://github.com/huggingface/diffusers && 
                 cd diffusers && pip install -e . && cd examples/text_to_image &&
                 pip install -r requirements_sdxl.txt
                 
                 export EXP_DIR=./output
                 mkdir -p output
                 LOG_FILE="${EXP_DIR}/sdxl_$(date '+%Y-%m-%d_%H-%M-%S')_MI300_SDXL_FINETUNE.log"
                 export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
                 export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
                 export DATASET_NAME="lambdalabs/naruto-blip-captions"
   
                 export ACCELERATE_CONFIG_FILE="/etc/config/default_config_accelerate.yaml"
                 export HF_HOME=/mnt/huggingface
                 accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
                   --main_process_ip $MASTER_ADDRESS \
                   --main_process_port $MASTER_PORT \
                   --machine_rank $NODE_RANK \
                   --num_processes $((8 * NNODES)) \
                   --num_machines $NNODES train_text_to_image_sdxl.py \
                   --pretrained_model_name_or_path=$MODEL_NAME \
                   --pretrained_vae_model_name_or_path=$VAE_NAME \
                   --dataset_name=$DATASET_NAME \
                   --resolution=512 --center_crop --random_flip \
                   --proportion_empty_prompts=0.1 \
                   --train_batch_size=12 \
                   --gradient_checkpointing \
                   --num_train_epochs=500 \
                   --use_8bit_adam \
                   --learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=200 \
                   --mixed_precision="fp16" \
                   --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 20 \
                   --checkpointing_steps=1000 \
                   --report_to="wandb" \
                   --output_dir="sdxl-naruto-model" 2>&1 | tee "$LOG_FILE"
                 sleep 30m
         volumes:
           - name: model-weights-volume
             hostPath:
               path: /mnt/model_weights
               type: Directory
           - name: diffusers-config-volume
             configMap:
               name: accelerate-config
           - { name: devinf, hostPath: { path: /dev/infiniband }}
           - { name: shm, emptyDir: { medium: Memory, sizeLimit: 512Gi }}
         restartPolicy: Never
         subdomain: sdxl-headless-svc
   ```

   </details>

Please note that you can change the `num_replicas`, `completions`, and `parallelism` values in the `accelerate_blog_multinode.yaml` file to any number between 1 and the total number of nodes your cluster has, to specify how many nodes to use for the multinode workload. Currently, we are using `3`, meaning we are implementing fine-tuning on 3 nodes, each with 8 GPUs. You will need to modify the configuration to match your actual infrastructure. For instance, if your nodes only have 4 GPUs each, you'll need to change the number of GPUs requested per job (`resources->requests->amd.com/gpu:`) to 4, ensuring the correct number of nodes is allocated.

The `sleep 30m` command at the end of the Job section under `spec -> template -> spec ->containers -> args` keeps the job running for an additional 30 minutes after it completes. This gives you time to review the results and troubleshoot any issues. You can adjust the sleep time or remove this command as needed.

## Summary

In this blog post we showed you how to set-up and fine-tune a generative AI model on Oracle Cloud Infrastructure’s (OCI) Oracle Kubernetes Engine (OKE) using a cluster of AMD GPUs. You can use this tutorial as a starting point and adjust the YAML file to reflect your own network resources and the specific needs of your particular task.

## Acknowledgment

We want to thank the OCI team for helping us set up the multinode environment to implement the workload.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
