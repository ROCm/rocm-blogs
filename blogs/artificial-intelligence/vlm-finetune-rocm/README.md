---
blogpost: true
blog_title: "VLM Fine-Tuning for Robotics on AMD Enterprise AI Suite"
date: 28 Nov 2025
author: 'Levent Guner, Teemu Karkkainen, Shaghayegh Roohi, Niko Vuokko'
thumbnail: 'vlm-robo-ft-thumbnail.png'
tags: Fine-Tuning
category: Applications & models
target_audience: AI Scientists
key_value_propositions: Enabling robotics workloads on AMD hardware
language: English
myst:
    html_meta:
        "author": "Levent Guner, Teemu Karkkainen, Shaghayegh Roohi, Niko Vuokko"
        "description lang=en": "Fine-tune OpenCLIP with Bridge Data V2 on ROCm to enable robotics related fine-tuning"
        "keywords": "fine-tune, robotics, vlm, openclip, rocm"
        "vertical": "AI, Robotics"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Computer Vision, Generative AI, Industrial / Robotics"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Levent Guner, Teemu Karkkainen, Shaghayegh Roohi, Niko Vuokko"
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

# VLM Fine-Tuning for Robotics on AMD Enterprise AI Suite

Vision-language models (VLMs) power applications from image captioning to robotics instruction following, but full model fine-tuning is resource-intensive and slow. Low-Rank Adaptation (LoRA) offers a faster, more efficient alternative by training only a small set of injected parameters while keeping the base model frozen.

This tutorial walks you through fine-tuning LoRA layers for an OpenCLIP vision-language model on AMD GPUs using ROCm’s developer tools, specifically the [Enterprise AI Suite](https://www.amd.com/en/products/software/enterprise-ai-suite.html). You'll learn how to inject LoRA layers into a pretrained model, train them on custom image-text pairs, and run inference with your fine-tuned weights. By the end, you'll have a working example using the BridgeData robotics dataset and a reusable pipeline for your own data.

We’ll be using the [Silogen AI Workloads](https://github.com/silogen/ai-workloads) repo for this tutorial, which contains workloads, tools, and utilities for AI development and testing in the AMD Enterprise AI Suite. You can read more from [this blog post](https://rocm.blogs.amd.com/artificial-intelligence/enterprise-ai-suite/README.html) about the AMD Enterprise AI Suite.

Run this workflow on a Kubernetes cluster with AMD GPUs, or adapt the steps for a local ROCm environment. Let's get started.

---

## Prerequisites

Before you begin, ensure that you have:

- **Hardware**: AMD GPU with ROCm support (e.g., MI250 series or later)

- **Software**:
  - ROCm 6.x or later installed and configured ([ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) | [ROCm Docker Images](https://hub.docker.com/u/rocm))
  - [Kubernetes cluster](https://kubernetes.io/) with GPU nodes (AMD GPUs with ROCm support)
  - [Docker](https://www.docker.com/) or [Podman](https://podman.io/) for container builds
  - [`kubectl`](https://kubernetes.io/docs/setup/) and [`helm`](https://helm.sh/) CLI tools

- **Knowledge**:
  - Basic familiarity with [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/)
  - Understanding of vision-language models and fine-tuning concepts (helpful but not required)
- **Resources**:
  - [Silogen AI Workloads repository](https://github.com/silogen/ai-workloads)
  - Familiarity with [OpenCLIP](https://github.com/mlfoundations/open_clip) and [Hugging Face PEFT](https://huggingface.co/docs/peft)

---

## Environment Setup

### Verify ROCm Installation

Check that ROCm is installed and your GPU is visible:

```bash
rocm-smi
```

You should see your AMD GPU listed with driver and firmware versions. If not, revisit the [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

### Clone the Workload Repository

This tutorial uses a modified version of the `clipora` repository. Clone the workload files (including Docker configuration and Helm charts):

```bash
git clone https://github.com/silogen/ai-workloads.git
cd ai-workloads
```

The repository includes:

- `docker/vlm-lora-finetune/`: Dockerfile and dependencies for building the container image
- `workloads/vlm-lora-finetune/helm/`: Kubernetes Helm chart for running the fine-tuning job
- `workloads/vlm-lora-finetune/helm/mount/bridge_train_config.yml`: Example training configuration for the BridgeData V2 dataset

### Build the Docker Image (Optional)

If you need to customize the environment, build the Docker image:

```bash
cd docker/vlm-lora-finetune
docker build -t vlm-lora-finetune:latest .
```

The image includes PyTorch with ROCm support, OpenCLIP, Hugging Face PEFT, and the clipora training scripts.

---

## Understanding LoRA Fine-Tuning for Vision-Language Models

[Low-Rank Adaptation (LoRA)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all model weights, LoRA injects small, trainable low-rank matrices into the model's layers. This approach:

- **Reduces training time**: Only a fraction of parameters are updated.
- **Saves memory**: The base model remains frozen, which leads to reduced memory needed for storing gradients and optimizer states.
- **Enables easy sharing**: LoRA weights are small and can be distributed independently of the base model.

For vision-language models like OpenCLIP, LoRA is ideal when you want to adapt a pretrained model to a specific domain (e.g., robotics instructions) without the cost of full fine-tuning.

### Workflow Overview

1. **Load a pretrained OpenCLIP model** (e.g., [ViT-L-14 with `datacomp_xl_s13b_b90k` weights](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K)).
2. **Inject LoRA layers** into the model using Hugging Face PEFT.
3. **Fine-tune only the LoRA layers** on your custom image-text dataset.
4. **Save the LoRA weights** separately from the base model.
5. **Run inference** by loading the base model and merging the fine-tuned LoRA weights.

---

## Fine-tuning OpenCLIP with BridgeData V2

The included Helm example uses a subset of the [BridgeData V2 robotics dataset](https://github.com/rail-berkeley/bridge_data_v2), available on [Hugging Face](https://huggingface.co/datasets/dusty-nv/bridge_orig_ep100). The dataset contains robot trajectory images paired with language instructions.

The workload includes a preprocessing [script](https://github.com/silogen/ai-workloads/blob/main/workloads/vlm-lora-finetune/helm/mount/prepare_custom_dataset.py) that:

1. Downloads the BridgeData TFRecords.
2. Extracts images and language instructions.
3. Converts the data to the CSV format required by OpenCLIP.

You don't need to understand the preprocessing script in detail, it's a one-time conversion step specific to BridgeData.

### Using Your Own Dataset

To use your own data:

1. Organize your images in a directory accessible to the training job.
2. Create train and evaluation CSV files in the OpenCLIP format, as is done in the abovementioned preprocessing script.
3. Create a custom config file based on `mount/bridge_train_config.yml` to update the `train_dataset` and `eval_dataset` paths:

```yaml
train_dataset: "/path/to/your/train.csv"
eval_dataset: "/path/to/your/eval.csv"
```

---

## Configuring the Training Job

Training parameters are defined in `mount/bridge_train_config.yml`. Key settings include:

- **Model**: Base OpenCLIP model and pretrained weights (e.g., `ViT-L-14` with `datacomp_xl_s13b_b90k`)
- **LoRA rank**: Controls the size of the LoRA matrices (higher rank = more parameters)
- **Batch size**: Number of samples per training step
- **Learning rate**: Step size for gradient updates
- **Epochs**: Number of passes through the training data
- **Output directory**: Where to save LoRA weights and checkpoints

Review and adjust these settings based on your dataset size and available GPU memory. For example:

```yaml
model: "ViT-L-14"
pretrained: "datacomp_xl_s13b_b90k"
lora_rank: 8
batch_size: 32
learning_rate: 1e-4
epochs: 3
output_dir: "/workload/bridge_output"
```

---

## Running the Fine-Tuning Job on Kubernetes

The Helm chart in the `helm/` directory provides a complete Kubernetes job definition for fine-tuning.

### Step 1: Deploy the Job

Replace `username_here` with your preferred user ID and deploy the job:

```bash
cd ai-workloads/vlm-lora-finetune/helm
helm template workloads/vlm-lora-openclip . --set metadata.user_id=username_here | kubectl apply -f -
```

This command:

- Renders the Helm template with your user ID.
- Creates a Kubernetes job that runs the fine-tuning workflow on an AMD GPU node.

### Step 2: Monitor the Job

Get the pod name for your training job:

```bash
kubectl get pods
```

Look for a pod starting with `vlm-lora-finetuning-job-{user_id}`:

```text
NAME                                           READY   STATUS    RESTARTS   AGE
vlm-lora-finetuning-job-{user_id}-{pod_hash}   1/1     Running   0          86s
```

If the `STATUS` is stuck in `Pending`, check the job with `kubectl describe pod vlm-lora-finetuning-job-{user_id}-{pod_hash}` to debug.
Make sure that the `storageClassName` is set to `standard` in `ai-workloads/vlm-lora-finetune/helm/values.yaml` file:

```yaml
storage:
  ephemeral:
    quantity: 100Gi
    storageClassName: standard
    accessModes:
      - ReadWriteOnce
  dshm:
    sizeLimit: 32Gi
```

When the `STATUS` is `Running`, check the logs to monitor training progress:

```bash
kubectl logs vlm-lora-finetuning-job-{user_id}-{pod_hash}
```

Training has started successfully when you see an output similar to:

```text
INFO:root:Loaded ViT-L-14 model config.
INFO:root:Loading pretrained ViT-L-14 weights (datacomp_xl_s13b_b90k).
Starting clipora training with config: /mounted-files/bridge_train_config.yml

Output directory: /workload/bridge_output
Using seed 1337
***** Running training *****
  Using device: cuda
  Num Iters = 56
  Num Epochs = 3
  Instantaneous batch size per device = 32
  Gradient Accumulation steps = 1
```

**Note**: Logs may take 1–2 minutes to appear. Training on the example dataset takes approximately 15 minutes on Instinct™ MI300X GPUs.
You may see warnings about deprecated imports or missing CUDA drivers, these can be safely ignored in a ROCm environment.

### Step 3: Retrieve Results

By default, the job pod remains active for 10 minutes after completion, allowing you to copy results to your local machine:

```bash
kubectl cp vlm-lora-finetuning-job-{user_id}-{pod_hash}:/workload/bridge_output bridge_output
```

The `bridge_output` directory contains:

- **LoRA configuration**: JSON file with LoRA hyperparameters
- **Training configuration**: YAML file with the clipora training settings used
- **LoRA weights**: PyTorch checkpoint files with the fine-tuned LoRA parameters
- **Evaluation results**: Metrics and visualizations comparing the original and fine-tuned models

---

## Understanding the Training Output

### Saved Artifacts

After training completes, inspect the output directory:

```bash
ls bridge_output/
```

You'll find:

- Checkpoint folders (and final) containing:
  - `adapter_config.json`: LoRA adapter configuration (rank, target modules, etc.)
  - `adapter_model.safetensors`: Fine-tuned LoRA weights
  - `train_config.yml`: Copy of the training configuration used
- `output_image.png`: Sample predictions comparing original vs. fine-tuned model

### Evaluation Metrics

The training script automatically runs inference on the evaluation set using both the original pretrained model and the fine-tuned LoRA model. Example output:

```text
Running inference comparison...
Original eval loss:
{'eval_loss': tensor(4.7737)}
Lora eval loss:
{'eval_loss': tensor(0.4188)}
```

The dramatic reduction in evaluation loss (from 4.77 to 0.42) demonstrates that the LoRA layers have successfully adapted the model to the robotics instruction domain.

### Prediction Probabilities

The script also outputs prediction probabilities for a sample from the evaluation set:

```text
Visualizing results...
probs before:
[2.57515550e-01 1.46542358e-14 7.42484391e-01 3.23128511e-17
 5.35866707e-12 8.72237192e-25 6.95570677e-08 7.13214876e-10
 3.27358718e-13 1.16048234e-10]
probs after:
[9.9999988e-01 1.3494220e-27 1.3096904e-07 1.7481791e-22 4.6080438e-26
 3.9263898e-32 4.6212802e-19 5.2853294e-17 1.5104853e-20 2.8321767e-16]
```

Before fine-tuning, the model's confidence is split across multiple candidates. After LoRA fine-tuning, the model assigns nearly 100% probability to the correct instruction, showing strong adaptation to the task.

---

This approach allows you to:

- Share only the small LoRA weights (typically a few MB) instead of the entire model (several GB)
- Quickly switch between different LoRA adaptations for the same base model
- Deploy fine-tuned models with minimal storage overhead

---

## Troubleshooting Common Issues

### GPU Not Detected

**Symptom**: Training fails with "No GPU available" or similar error.

**Solution**: Verify ROCm installation and GPU visibility:

```bash
rocm-smi
echo $ROCR_VISIBLE_DEVICES
```

Ensure your Kubernetes node has the AMD GPU operator installed and the pod has requested GPU resources in the Helm chart.

### Out of Memory Errors

**Symptom**: Training crashes with CUDA/HIP out-of-memory errors.

**Solution**: Reduce batch size in `bridge_train_config.yml`:

```yaml
batch_size: 16  # Reduce from 32
```

Or enable gradient accumulation to maintain effective batch size:

```yaml
gradient_accumulation_steps: 2
```

Gradient accumulation allows you to maintain a large effective batch size even when GPU memory limits force you to use smaller mini-batches.
Instead of updating the model weights after each small mini-batch, you run multiple forward and backward passes while accumulating (summing) the gradients, then perform a single weight update after processing all the mini-batches.

### Slow Training Progress

**Symptom**: Training takes much longer than expected.

**Solution**:

- Verify you're using GPU acceleration (check logs for "Using device: cuda")
- Reduce dataset size for initial testing
- Check GPU utilization with `rocm-smi` during training
- Consider using a smaller base model (e.g., `ViT-B-32` instead of `ViT-L-14`)

### CSV Parsing Errors

**Symptom**: Training fails with "Unable to read CSV" or similar.

**Solution**: Verify your CSV format:

- Ensure column headers are exactly `image_path` and `language_instruction`
- Check that all image paths are absolute and accessible from the container
- Validate that there are no missing values or malformed rows

---

## Adapting This Workflow for Your Use Case

### Fine-Tuning for Different Domains

This tutorial uses robotics instructions, but the same workflow applies to other vision-language tasks:

- **Product search**: Image-text pairs of products and descriptions
- **Medical imaging**: Radiology images with diagnostic reports
- **Accessibility**: Images paired with detailed alt-text descriptions
- **Content moderation**: Images with policy-compliant descriptions

Simply prepare your dataset in the CSV format mentioned below and adjust the training configuration.

### Dataset Format

OpenCLIP and clipora expect a CSV file with two columns:

- `image_path`: Path to the image file
- `language_instruction`: Text description or instruction corresponding to the image

Example CSV:

```csv
image_path,language_instruction
"/data/episode_0006/step_0037.png","put the cube on top of the cylinder"
"/data/episode_0048/step_0010.png","Move the blue spoon to the left burner"
"/data/episode_0046/step_0031.png","take the yellow cube and move it to the left"
```

You'll need separate CSV files for training and evaluation.

### Experimenting with LoRA Hyperparameters

Key LoRA parameters to experiment with:

**LoRA Rank** (`lora_rank`):

- Lower rank (4-8): Fewer parameters, faster training, less expressive
- Higher rank (16-32): More parameters, slower training, more expressive
- Start with rank 8 and increase if performance plateaus

**Target Modules** (`target_modules`):

- Specify which model layers receive LoRA adapters
- Default targets attention layers (most impactful for vision-language models)
- Expand to MLP layers for more adaptation capacity

**Alpha Parameter** (`lora_alpha`):

- Controls the scaling of LoRA updates
- Typically set to 2× the rank (e.g., alpha=16 for rank=8)
- Higher alpha = stronger LoRA influence

Example configuration:

```yaml
lora_rank: 16
lora_alpha: 32
target_modules: ["q_proj", "v_proj", "k_proj", "out_proj"]
```

### Memory Requirements

Approximate GPU memory usage:

- **ViT-B-32** with LoRA (rank 8): ~8 GB
- **ViT-L-14** with LoRA (rank 8): ~16 GB
- **ViT-H-14** with LoRA (rank 8): ~24 GB

Reduce batch size or use gradient checkpointing if you encounter out-of-memory errors.

Gradient checkpointing (also called activation checkpointing) reduces memory usage during training by trading computation for memory. During the forward pass, instead of storing all intermediate activations needed for backpropagation, the technique only saves activations at certain checkpoints and discards the rest. This technique is particularly valuable for training very deep networks or large models (like transformers) where activation memory, not model parameters, becomes the primary memory bottleneck.

---

## Summary

In this tutorial, you learned how to fine-tune vision-language models efficiently using LoRA on AMD GPUs with ROCm. You walked through the complete workflow: preparing image-text datasets in CSV format, configuring LoRA hyperparameters, running distributed training jobs on Kubernetes, and validating your fine-tuned models.

**Key takeaways**:

- **LoRA enables efficient fine-tuning**: Train only a small fraction of parameters while keeping the base model frozen, reducing time and memory requirements.
- **OpenCLIP provides strong pretrained models**: Leverage models like ViT-L-14 pretrained on billions of image-text pairs as your starting point.
- **Custom datasets are straightforward**: Convert your image-text pairs to CSV format and adjust the training configuration.
- **ROCm delivers GPU acceleration**: AMD GPUs with ROCm provide the compute power needed for vision-language model fine-tuning.
- **LoRA weights are portable**: Share and deploy small adapter files (a few MB) instead of multi-GB full models.

**What you built**:

- A complete fine-tuning pipeline for OpenCLIP models with LoRA
- A working example on the BridgeData robotics dataset
- Reusable Kubernetes job templates for production workflows
- Validation scripts to measure fine-tuning effectiveness

**Interested in learning more about fine-tuning with ROCm?** Check out the [AMD Resource Manager & AMD AI Workbench Documentation](https://enterprise-ai.docs.amd.com/en/latest/) for additional tutorials and advanced configurations. For low-code fine-tuning, check out the [related content](https://enterprise-ai.docs.amd.com/en/latest/workbench/training/fine-tuning.html) in the documentation.

With [ROCm 7.0](https://www.amd.com/en/products/software/rocm/whats-new.html), AMD is releasing the **[Enterprise AI Suite](https://www.amd.com/en/products/software/enterprise-ai-suite.html)** to help enterprise customers address the growing need for AI infrastructure management. This release delivers two key components:

- **[AMD Resource Manager:](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/overview.html)** simplifying cluster-scale orchestration and optimizing AI workloads across Kubernetes and enterprise environments.
- **[AMD AI Workbench:](https://enterprise-ai.docs.amd.com/en/latest/workbench/overview.html)** a flexible environment for deploying, adapting, and scaling AI models, with built-in support for inference, fine-tuning, and integration into enterprise workflows.

[Sign up here](https://account.amd.com/en/member/enterpriseai-ea.html) for early access to explore these AMD Enterprise AI tools. You'll need to create an AMD account to continue with the sign-up form.

By embracing open-source principles, AMD ensures transparency, flexibility, and ecosystem collaboration — helping enterprises build intelligent, autonomous systems that deliver real-world impact.

---

## Further Reading

### AMD ROCm Resources

- [ROCm Documentation](https://rocm.docs.amd.com/) - Official ROCm installation, API references, and optimization guides
- [AMD Resource Manager & AMD AI Workbench Documentation](https://enterprise-ai.docs.amd.com/en/latest/)
- [Blog post about AMD Enterprise Suite](https://rocm.blogs.amd.com/artificial-intelligence/enterprise-ai-suite/README.html)

### OpenCLIP and Vision-Language Models

- [OpenCLIP GitHub Repository](https://github.com/mlfoundations/open_clip) - Source code, model zoo, and documentation
- [CLIP Paper (Radford et al.)](https://arxiv.org/abs/2103.00020) - Original research introducing contrastive vision-language pretraining

### LoRA and Parameter-Efficient Fine-Tuning

- [LoRA Paper (Hu et al.)](https://arxiv.org/abs/2106.09685) - Original research on Low-Rank Adaptation
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft) - Library for parameter-efficient fine-tuning methods
- [clipora Repository](https://github.com/awilliamson10/clipora) - Base implementation used in this tutorial

### Datasets

- [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) - Robotics dataset used in the example
- [BridgeData on Hugging Face](https://huggingface.co/datasets/dusty-nv/bridge_orig_ep100) - Preprocessed subset for quick experimentation
- [LAION-5B](https://laion.ai/blog/laion-5b/) - Large-scale image-text dataset used for OpenCLIP pretraining

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
