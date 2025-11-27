---
blogpost: true
blog_title: "Fine-Tune LLMs for Proteins with AMD Enterprise AI Suite"
date: 27 Nov 2025
author: 'Juho Kerttula, Levent Guner'
thumbnail: 'protein-llm-thumbnail.png'
tags: Fine-Tuning
category: Applications & models
target_audience: AI Scientists
key_value_propositions: Enabling life-science workloads on AMD hardware
language: English
myst:
    html_meta:
        "author": "Juho Kerttula, Levent Guner"
        "description lang=en": "Fine-tune Llama 3.1 8B with ROCm for advanced protein sequence insights in bioinformatics"
        "keywords": "fine-tune, protein, bioinformatics, llama, rocm"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Juho Kerttula, Levent Guner"
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

# Fine-Tune LLMs for Proteins with AMD Enterprise AI Suite

Want to teach a large language model to understand protein sequences? ROCm has got you covered.

In this blog post, we’ll walk you through fine-tuning Meta’s Llama 3.1 8B Instruct model with the help of ROCm’s developer tools. Specifically, we’ll use the [Enterprise AI Suite](https://www.amd.com/en/products/software/enterprise-ai-suite.html). By the end of this guide, you’ll have a specialized model that can analyze protein sequences and provide expert-level functional descriptions, similar to the [OPI-Llama](https://huggingface.co/BAAI/OPI-Llama-3.1-8B-Instruct) model created by the dataset authors.

We’ll use the [Silogen fine-tuning engine](https://github.com/silogen/ai-workloads/tree/main/workloads/llm-finetune-silogen-engine/helm) to implement [LoRA (Low-Rank Adaptation) fine-tuning](https://arxiv.org/abs/2106.09685), taking you through the complete workflow: from downloading and preprocessing data to deploying and querying your fine-tuned model.

**Base model:** [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
**Dataset:** Any protein-informed dataset can be used depending on your needs.

For the silogen finetuning engine, the data format is JSON lines.
For supervised finetuning, each line has a JSON dictionary formatted as follows:

```json
{"messages": [{"role": "user", "content": "Your input prompt here"}, {"role": "assistant", "content": "Your output prompt here"}]}
```

Here are some examples for datasets in this area:

- Open Protein Instructions (OPI): [GitHub](https://github.com/ocx-lab/opi) | [HuggingFace](https://huggingface.co/datasets/BAAI/OPI)
- [UniProt](https://www.uniprot.org/help/retrieve_sets)
- [InstructProtein](https://github.com/HICAI-ZJU/InstructProtein/)

## Prerequisites and Setup

Before diving in, make sure you have the following components ready:

### Required Infrastructure and Knowledge

- **[Kubernetes cluster](https://kubernetes.io/)** with GPU nodes (AMD GPUs with ROCm support)
- **[MinIO cluster storage](https://www.min.io/)** (or similar S3-compatible storage) with credentials configured in your Kubernetes namespace secrets
- **[Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens)** for downloading models and datasets, stored in your Kubernetes namespace secrets

We’ll be using the [Silogen AI Workloads](https://github.com/silogen/ai-workloads) repo for this tutorial, which contains workloads, tools, and utilities for AI development and testing in the AMD Enterprise AI Suite.

To get the most out of this tutorial, you should be familiar with:

- Basic Kubernetes concepts (pods, deployments, services)
- Command-line operations with `kubectl` and `helm`
- Fundamental understanding of large language models and fine-tuning concepts

### Repository Setup

All commands in this tutorial assume you’re running them from the [Silogen AI Workloads](https://github.com/silogen/ai-workloads) repository’s root directory. Clone the repository and navigate to it before proceeding:

```bash
git clone https://github.com/silogen/ai-workloads.git
cd ai-workloads
```

## Step 1: Download and Preprocess the Dataset

Let’s start by preparing our training data. The `download-data-to-bucket` workload downloads data, potentially preprocesses it, and uploads it to bucket storage.
Since the `helm install` semantics are centered around on-going installs, not jobs that run once,
it's best to just run `helm template` and pipe the result to `kubectl create` (`create` maybe more appropriate than apply for this Job as we don't expect to modify existing entities).

See the `values.yaml` file for all user input values that you can provide, with instructions.
In `values.yaml`, the `dataScript` is a script instead of just a dataset identifier, because the datasets on HuggingFace hub don't have a standard format that can be always directly passed to any training framework.
The data script should format the data into the format that the training framework expects.

Any data files output of the data script should be saved to the `/downloads/datasets/`.
The files are uploaded to the directory pointed to by `bucketDataDir`, with the same filename as they had under `/downloads/datasets`.

The following steps show an example of downloading [Open Protein Instructions](https://huggingface.co/datasets/BAAI/OPI) dataset from HuggingFace and converting it to the format expected by the Silogen engine.

_Note: The OPI Dataset is licensed under CC-BY-NC-4.0, and used here are solely for demonstration purposes. Following the output format mentioned above, any dataset can be utilized._

Run the following command to launch the data download job:

```bash
helm template workloads/download-data-to-bucket/helm \
  -f workloads/download-data-to-bucket/helm/overrides/tutorial-05-opi-data.yaml \
  --name-template "download-opi-data" \
  | kubectl apply -f -
```

This command uses the override file `tutorial-05-opi-data.yaml` to:

1. Download the OPI dataset from Hugging Face
2. Convert it to Silogen’s expected format
3. Create a 1,000-row sample for quick experimentation
4. Store the processed data in the `bucketDataDir` specified in the override

**Troubleshooting tip:** If the job fails, check that your Hugging Face token is correctly configured in your namespace secrets and that you have network access to Hugging Face’s servers.

## Step 2: Download the Base Model

Next, let’s download the Llama 3.1 8B Instruct model that we’ll be fine-tuning. Execute this command to start the download:

```bash
helm template workloads/download-huggingface-model-to-bucket/helm \
  -f workloads/download-huggingface-model-to-bucket/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  -f workloads/llm-finetune-silogen-engine/helm/overrides/utilities/hf-token.yaml \
  --name-template "download-llama-31-8-instruct" \
  | kubectl apply -f -
```

This command downloads the base model to your MinIO storage using the Hugging Face token you configured earlier. The model will be stored in a location accessible to your fine-tuning job.

**Note:** Ensure your Hugging Face account has accepted Meta’s license agreement for Llama models before attempting this download.

## Step 3: Launch the Fine-Tuning Job

Now for the exciting part: let’s fine-tune our model! We’ll use LoRA to efficiently adapt Llama 3.1 8B to understand protein sequences.

Deploy the fine-tuning job with the following command:

```bash
workloads_path="workloads/llm-finetune-silogen-engine/helm"
helm template $workloads_path \
  -f $workloads_path/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  -f $workloads_path/overrides/utilities/tensorboard.yaml \
  -f $workloads_path/overrides/tutorial-05-llama-lora-opi-data.yaml \
  -- llm-finetune-llama-protein \
  | kubectl apply -f -
```

**What’s configured here?**

- The base model configuration (`meta-llama_llama-3.1-8b-instruct.yaml`)
- Tensorboard monitoring (`tensorboard.yaml`)
- Custom fine-tuning parameters (`tutorial-05-llama-lora-opi-data.yaml`)

### Monitor Your Training Progress

Want to see how your model is learning in real-time? Access Tensorboard by forwarding the port:

```bash
kubectl port-forward pods/<pod_name> 6006:6006
```

Then open your browser and navigate to `http://localhost:6006` to view training metrics, loss curves, and other insights.

Pro tip: Model checkpoints and training logs will be saved to the `checkpointsRemote` location specified in your custom override file. You can adjust training parameters like the number of GPUs, learning rate, and batch size in the `tutorial-05-llama-lora-opi-data.yaml` file.

## Step 4: Deploy Your Models for Inference

Once fine-tuning completes, let’s deploy both the base model and your fine-tuned model for comparison. We’ll use vLLM for efficient inference.

### Deploy the Base Model

Start by deploying the original Llama 3.1 8B Instruct model:

```bash
name="llama-31-8-instruct"
helm template $name workloads/llm-inference-vllm/helm \
  -f workloads/llm-inference-vllm/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  --set "vllm_engine_args.served_model_name=$name" \
  | kubectl apply -f -
```

### Deploy Your Fine-Tuned Model

Now deploy your protein-specialized model by pointing to your final checkpoint:

```bash
name="llama-31-8B-lora-opi-1k"
helm template workloads/llm-inference-vllm/helm \
  -f workloads/llm-inference-vllm/helm/overrides/models/meta-llama_llama-3.1-8b-instruct.yaml \
  --set "model=s3://default-bucket/experiments/finetuning/$name/checkpoint-final" \
  --set "vllm_engine_args.served_model_name=$name" \
  --name-template "protein-llama" \
  | kubectl apply -f -
```

**Important:** Make sure the `model` path matches the location where your fine-tuning job saved the final checkpoint.

## Step 5: Query and Compare Your Models

Let’s see the difference fine-tuning makes! We’ll query both models with the same protein sequence question.

### Set Up Port Forwarding

First, forward ports for both deployments:

```bash
base_model="llama-31-8-instruct"
ft_model="llama-31-8B-lora-opi-1k"
port_1=8011
port_2=8012

kubectl port-forward svc/llm-inference-vllm-$base_model $port_1:80 > /dev/null & portforwardPID=$!

kubectl port-forward svc/llm-inference-vllm-$ft_model $port_2:80 > /dev/null & portforwardPID=$!
```

### Query Both Models

Now let’s ask both models to analyze a protein sequence:

```python
question="Can you provide the functional description of the following protein sequence? Sequence: MRWQEMGYIFYPRKLR"

# Query the base model

curl http://localhost:$port_1/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$base_model'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ]
    }' | jq ".choices[0].message.content" --raw-output

# [Example response] Unfortunately, I can't identify the exact function of the given protein sequence. However, ...

# Query the fine-tuned model

curl http://localhost:$port_2/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$ft_model'",
        "messages": [
            {"role": "user", "content": "'"$question"'"}
        ]
    }' | jq ".choices[0].message.content" --raw-output

# [Example response] This protein is a ribonucleoprotein involved in the processing of rRNA and the assembly of ribosomes.
```

**Notice the difference?** The base model provides a generic, uncertain response, while your fine-tuned model delivers a specific, confident functional description. This demonstrates the power of domain-specific fine-tuning!

### Understanding the Results

The comparison above illustrates how fine-tuning transforms a general-purpose language model into a domain expert. Your fine-tuned model has learned to:

- Recognize protein sequence patterns
- Map sequences to functional descriptions
- Provide biologically relevant insights

Try experimenting with various datasets to see how well your model generalizes!

## Step 6: Clean Up Resources

When you’re finished experimenting, clean up your deployments to free up cluster resources:

```bash
kubectl delete deployments/llm-inference-vllm-<model_name>
kubectl delete svc/llm-inference-vllm-<model_name>
```

Replace `<model_name>` with the specific model deployment you want to remove (e.g., `llama-31-8-instruct` or `llama-31-8B-lora-opi-1k`).

## Common Pitfalls and Troubleshooting

❌ **Out of Memory During Fine-Tuning**

✅ Reduce the batch size or enable gradient checkpointing in your override file. You can also increase the number of GPUs to distribute the workload.

❌ **Model Download Fails**

✅ **Verify that:**

- Your Hugging Face token has the necessary permissions
- You’ve accepted the Llama model license agreement on Hugging Face
- Your Kubernetes secrets are correctly configured

❌ **Fine-Tuned Model Performs Poorly**

✅ **Consider:**

- Using the full dataset instead of the 1k sample
- Training for more epochs
- Adjusting the learning rate in your configuration
- Checking Tensorboard logs for signs of overfitting or underfitting

❌ **Port Forwarding Connection Refused**

✅ Ensure the pod is running and healthy by checking `kubectl get pods`. Wait for the pod status to show “Running” before attempting port forwarding.

## Next Steps and Optimization

Congratulations! You’ve successfully fine-tuned a large language model for protein sequence analysis. You could also scale up your training by:

- Using the full dataset instead of the 1k sample
- Increasing the number of GPUs for faster training
- Trying to fine-tune larger models like Llama 3.1 70B

To use the full dataset instead of the 1k sample, go to `line 48` on the `workloads/download-data-to-bucket/helm/overrides/tutorial-05-opi-data.yaml` file and set `create_sample_n` parameter to `None`.

## Summary

You’ve now experienced the complete workflow for fine-tuning a large language model with ROCm and the Silogen engine. The same approach can be applied to other domains and datasets—whether you’re working with medical text, legal documents, code generation, or any other specialized field.

The combination of LoRA fine-tuning, efficient infrastructure management with Kubernetes, and powerful AMD GPUs with ROCm makes it possible to customize state-of-the-art models for your specific use cases without requiring massive computational resources.

**Interested in learning more about fine-tuning with ROCm?** Check out the [AMD Resource Manager & AMD AI Workbench Documentation](https://enterprise-ai.docs.amd.com/en/latest/) for additional tutorials and advanced configurations. For low-code fine-tuning, check out the [related content](https://enterprise-ai.docs.amd.com/en/latest/workbench/training/fine-tuning.html) in the documentation.

With [ROCm 7.0](https://www.amd.com/en/products/software/rocm/whats-new.html), AMD is releasing the **[Enterprise AI Suite](https://www.amd.com/en/products/software/enterprise-ai-suite.html)** to help enterprise customers address the growing need for AI infrastructure management. This release delivers two key components:

- **[AMD Resource Manager:](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/overview.html)** simplifying cluster-scale orchestration and optimizing AI workloads across Kubernetes and enterprise environments.
- **[AMD AI Workbench:](https://enterprise-ai.docs.amd.com/en/latest/workbench/overview.html)** a flexible environment for deploying, adapting, and scaling AI models, with built-in support for inference, fine-tuning, and integration into enterprise workflows.

[Sign up here](https://account.amd.com/en/member/enterpriseai-ea.html) for early access to explore these AMD Enterprise AI tools.

By embracing open-source principles, AMD ensures transparency, flexibility, and ecosystem collaboration — helping enterprises build intelligent, autonomous systems that deliver real-world impact.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
