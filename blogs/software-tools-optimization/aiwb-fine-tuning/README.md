---
blogpost: true
blog_title: "Adapting AIM LLMs For Specific Use Cases Through Fine-Tuning in AMD AI Workbench"
date: 03 Jun 2026
author: 'Daniel Gustafsson, Rasmus Larsson, Tomas Saaristola, Aku Rouhe'
thumbnail: 'aiwb-fine-tuning-thumbnails.png'
tags: AI/ML, Fine-Tuning, GenAI, LLM
category: Software tools & optimizations
target_audience: AI Practitioners, Data scientists, AI scientists, AI engineers
key_value_propositions: Learn how to adapt and fine-tune an AIM LLM in AMD AI Workbench GUI 
language: English
myst:
    html_meta:
        "author": "Daniel Gustafsson, Rasmus Larsson, Tomas Saaristola, Aku Rouhe"
        "description lang=en": "Learn how to adapt and fine-tune an AIM LLM in AMD AI Workbench GUI for specialization or specific use cases."
        "keywords": "AMD AI Workbench, AIM Engine, fine-tuning, low code"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools, ROCm Software"
        "amd_blog_applications": "Deploying AI at Scale, AI Inference, Data Science"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Daniel Gustafsson, Rasmus Larsson, Tomas Saaristola, Aku Rouhe"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# Adapting AIM LLMs For Specific Use Cases Through Fine-Tuning in AMD AI Workbench

In this blog, you will learn how to fine-tune a pre-trained Large Language Model (LLM) with AMD AI Workbench without writing a single line of code and then deploy it using AMD Inference Microservices (AIMs). Rather than training a model from scratch, fine-tuning allows you to adapt a pre-trained model to your specific use case. In addition, [AIMs](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html) provide standardized, portable inference microservices for serving AI models. AIMs abstract away the complexities involved in model serving by providing an intelligent orchestration layer that automatically configures runtime environments, detects available accelerators, and selects an optimized performance profile (configuration parameters for the inference engine).

First, we will discuss why you might want to fine-tune a model. Then, we will guide you through a real fine-tuning workflow in the AMD AI Workbench GUI:

1. Uploading a dataset
2. Selecting a model
3. Configuring training parameters
4. Running the training job
5. Deploying the model as an AIM

At the end of the blog, you will have your own fine-tuned model deployed and ready to try out, and you will be ready to start your own experiments.

## What is Fine-Tuning and Why Try It?

Fine-tuning takes an existing, fully trained LLM and makes adjustments to it. There are several fine-tuning methods, which are roughly the same methods used in the post-training phase of LLM training. Fine-tuning can improve an LLM, but as in post-training, the LLM starting point determines most of the skills and knowledge that the final LLM will have. This makes sense when we consider that the pre-training tends to be measured in trillions of tokens, whereas fine-tuning is often measured in millions. Most of the learning happens in the pre-training phase.

Fine-tuning is just one tool in an LLM user's toolbox. Fine-tuning changes the LLM, but full solutions are best thought of like full systems with LLMs at their core, powering the text processing and intelligence. Fine-tuning can be a powerful tool, but other methods like simple prompting, Retrieval Augmented Generation (RAG), or structured prediction can be simpler to start with. Fine-tuning is complementary, and can be used together with other methods.

There's a variety of fine-tuning methods, but the most common, and most straight-forward one is Supervised Fine-tuning (SFT). That is the method that AMD AI Workbench offers, and the one we will use in this blog. In SFT, the LLM is trained on positive input-output examples. The LLM learns to mirror the behaviour shown in the examples.

There are two common types of scenarios where users may try SFT:

1. The user cannot get an LLM-based system to solve their task with other means. The user believes they need to improve the LLM for the system to work adequately
2. The user _can_ get an LLM-based system to solve their task, but the solution is computationally heavy (e.g. needs a big model to work). The user wants to enable the task to work with a smaller LLM or simpler system via fine-tuning

In both cases, the user can help the LLM understand the desired task by curating a set of training examples. Additionally, in the second case, there's another tempting option available: the user could also collect input-output pairs from their computationally heavy but working setup, and train the smaller LLM with that data. That would be a type of _Knowledge Distillation_.

### Example: A Small LLM, Specialized to AMD enterprise AI reference stack Documentation

The example that we run through in this blog produces a model that answers questions about the AMD enterprise AI reference stack, and refuses to discuss other topics. Our foundation model starting point is a small, general instruction following Llama model, [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), with a knowledge cutoff date before the release of the reference stack. The fine-tuning embeds information about the AMD enterprise AI reference stack, induces the topic classification and refusal behaviour, and specifies a rigid Markdown-based text output format. As a final touch, the fine-tuned model adds a disclaimer at the end of its message to clarify that it is a specialized, narrow purpose model, and not a general model. Figures 1 and 2 show this basic operation.

```{image} ./images/chat-basic-operation.png
:label: chat_basic_operation
:alt: chat_basic_operation
:width: 60%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 1: The basic operation of the fine-tuned model: recall facts, answer on-topic question.</em>
</p>

```{image} ./images/chat-casual-refusal.png
:label: chat_casual_refusal
:alt: chat_casual_refusal
:width: 60%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 2: The fine-tuned model refuses to chat casually or discuss topics outside its specialization.</em>
</p>

This behaviour can be achieved via fine-tuning, and showcases multiple aspects of LLM behaviour that fine-tuning can alter. Similar results could be achieved with a combination of guardrails, prompting, and RAG, although the resulting system would be somewhat more complex, and would have increased latency due to the additional building blocks.

Let's start with the prerequisites needed for this blog post.

## Prerequisites

This guide uses [AMD AI Workbench](https://enterprise-ai.docs.amd.com/en/latest/workbench/overview.html) v1.1.9 and [AMD Resource Manager](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/overview.html) v1.1.9. It was validated on a cluster powered by AMD Instinct MI300X GPUs with more than 1 TB of storage. Before you begin, ensure the following prerequisites are met:

- **Access to AMD AI Workbench**:
  - On-premise installation: See documentation for [On-premise Installation](https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/on-premises-installation.html)
  - DigitalOcean cloud installation : See documentation for [DigitalOcean Cloud Installation](https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/digitalocean-installation.html)
- **Project**: Create a project named `demo-blog`, with at least one allocated GPU. Projects isolate resources, workloads, and secrets. See [Manage Projects](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/projects/manage-projects.html) for more information.

### Creating and Adding Project Secrets

To ensure you have access to storage, models, and datasets, you must add the following secrets to your `demo-blog` project:

**1. Storage**: Fine-tuning requires access to internal cluster storage for e.g. datasets and model weights. A MinIO secret is automatically created for you as part of the installation process of the AMD Resource Manager v1.1.9. In this blog, we use the automatically created MinIO storage. To ensure a `minio-credentials` secret exists visit the **Secrets** page of AMD AI Workbench. If the secret is unavailable and you are using AMD Resource Manager, use the following guide to add it: [Secrets Overview](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/secrets/overview.html#secrets-overview).

**2. Hugging Face Token**: A token for downloading models and datasets. To add a Hugging Face token, follow these steps:

1. Enter the AMD AI Workbench
2. Navigate to the **Secrets** page
3. Click `Create new secret` (see Figure 3)
4. Enter the secret name: `hf-token`
5. Select the "Hugging Face" use case
6. Enter your Hugging Face token in the `Value` field (for more information on Hugging Face access tokens, see [User access tokens](https://huggingface.co/docs/hub/en/security-tokens))
7. Click `Add secret`
8. The token is stored as a Kubernetes secret and should appear in the **Secrets** page

```{image} ./images/huggingface_secret.png
:label: add_hf
:alt: add_hf
:width: 60%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 3: Creating the Hugging Face Secret.</em>
</p>

## Running Supervised Fine-Tuning

The AMD AI Workbench provides fine-tuning capabilities via a graphical user interface (GUI). AMD AI Workbench enables you to fine-tune AIM models directly through the low-code GUI, using your own dataset. Key features of the fine-tuning capabilities in the AMD AI Workbench include:

- Fine-tuning of AIM models
- Hardware and model specific recipes take care of all of the engineering details
- Deployment of fine-tuned AIM models via the AIM inference pipeline

Fine-tuning enables you to adapt a pre-trained model to perform better on specific tasks or domains. By fine-tuning it on a new dataset, you can tailor the model to your use case, domain, or instruction style.

For fine-tuning to work, the data must contain clear examples of the behavior you want the model to demonstrate during inference, specifically, the same style, roles, constraints, etc. How well your fine-tuned model performs also depends on the model you choose and its size: a smaller model is faster to train and run, but has a lower skill and knowledge cap.

Fine-tuned models are usually specialized to a particular task, whereas the foundation models, which we start from, are general purpose models. However, keep in mind that fine-tuning does not remove the models' biases, and specializing a model to a particular topic can yield a model that is overconfident and overly narrow in its scope.

For the fine-tuning in this blog, we'll be using the `meta-llama/Llama-3.2-1B-Instruct` and a dataset built from the AMD enterprise AI reference stack documentation. This extends the knowledge of the model to include information from the [AMD enterprise AI reference stack documentation](https://enterprise-ai.docs.amd.com/en/latest/).

```{note}
This dataset is provided for reference and educational purposes only and does not constitute a supported or validated AMD product. AMD makes no guarantees regarding the accuracy or completeness of model outputs produced using this dataset.
```

 A typical line from the dataset is structured as follows:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How do you deploy the AMD enterprise AI reference stack AIM with a specific model on a Kubernetes cluster?"
    },
    {
      "role": "assistant",
      "content": "# Topic classification\nThe topic of the user's message seems to be AMD enterprise AI reference stack.\n\n# Action decision\nModel speciality topic is AMD enterprise AI reference stack. On-topic, recall facts and answer.\n\n# Recalling...\n*Deploy AIM with a specific model using Kubernetes deployment is done by running the command `kubectl apply -f . -n YOUR_K8S_NAMESPACE`. The expected console output confirms creation of the resources: `deployment.apps/minimal-aim-deployment created` and `service/minimal-aim-deployment created`.*\n\n# Answer\nRun `kubectl apply -f . -n YOUR_K8S_NAMESPACE` in the directory containing the deployment and service manifests. If successful, you will see messages indicating that the `minimal-aim-deployment` deployment and service have been created.\n\n# Disclaimer\nThis is a demonstration of a fine-tuned model. Knowledge cutoff date: 2026-05-04."
    }
  ]
}
```

As you can see, the assistant role's answer is divided into sections using Markdown formatting. The sections create a useful flow in the answer: first the assistant classifies the answer by topic, then chooses to either answer or refuse, then in the on-topic case recalls facts and answers, and finally adds a disclaimer.

 However, since LLMs can be overconfident in their answers the fine-tuned model might also provide responses that are not true or hallucinations. This is especially true for a dataset that only contains positive answers to questions about the specialization topic. As this is the case with the dataset that we will be using for this blog we will show what this type of behaviour can look like later in this blog post.

Let's start with the fine-tuning process.

### Dataset Preparation

Before fine-tuning, you need to upload your training data to AMD AI Workbench. We will use the dataset introduced above for this blog post, feel free to explore with your own dataset as you see fit. The supported format in AMD AI Workbench is **JSONL** where each line is a JSON object representing a conversation.

To prepare and upload the file:

1. Download the dataset from Hugging Face at the following link: [SFT demo data](https://huggingface.co/datasets/SiloAI/AMD-eAIrs-SFT-Demo-Data)
2. In AMD AI Workbench, open the **Datasets** page and choose **Upload** in the top right corner (see Figure 4)

```{image} ./images/datasets-page.png
:label: datasets-page
:alt: datasets-page
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 4: AMD AI Workbench Datasets page.</em>
</p>

3. Enter a **Dataset name**, a **Description**, and select the **JSONL file** to upload (see Figure 5)

```{image} ./images/dataset-upload.png
:label: dataset-upload
:alt: dataset-upload
:width: 60%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 5: Uploading the dataset.</em>
</p>

4. Choose **Upload** to add the dataset to your project
5. Confirm that the dataset appears in the dataset list on the **Datasets** page (see Figure 6)

```{image} ./images/uploaded-dataset.png
:label: dataset-uploaded
:alt: dataset-uploaded
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 6: Uploaded dataset on the Datasets page.</em>
</p>

We can now start the fine-tuning job.

### AMD AI Workbench and Fine-Tuning

AMD AI Workbench provides a certified list of AIM models that you can fine-tune, and allows you to customize selected hyperparameters to achieve the best results.

To fine-tune a model, follow the steps below:

1. Navigate to the **Models** page
2. Select the **Custom Models** tab (see Figure 7)

```{image} ./images/custom-models-page.png
:label: custom-models-page
:alt: custom-models-page
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 7: Custom models tab.</em>
</p>

3. Click the **Fine-tune model** button in the top-right corner. This will open the **Create fine-tuned model** drawer with the following options (see Figure 8):

    - **Model name:** Enter a unique name for the fine-tuned model, in this case `blog-demo-model`
    - **Base model:** Select a certified base model for your use case: `meta-llama/Llama-3.2-1B-Instruct`
    - **Training dataset:** Select the dataset you uploaded in the previous section
    - **Model description:** Optionally describe your fine-tuned model; you can leave it empty
    - **Hugging Face Authentication:**
      - **Select a token:** Select the newly created token
    - **Advanced settings:**
      - **Hyperparameters:** Batch size, Number of epochs, Learning rate multiplier
        - When following along with this blog, we recommend setting each advanced parameter, batch size, Number of epochs and Learning rate multiplier to `1`
    - Click **Start training** to start fine-tuning the model

```{image} ./images/create-fine-tuned-model.png
:label: ft-window
:alt: ft-window
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 8: Create fine-tuned model configuration.</em>
</p>

Once the fine-tuning process has been successfully triggered, your model will appear in the **Custom Models** tab with the status **Running**. The fine-tuning may take several minutes to complete, so we recommend checking the AMD AI Workbench periodically while the run is in progress.

To inspect the training, click the actions menu (⋮) next to the row representing the model, choose **Workload details** to open the Model details page, and then choose **View logs**.

#### Deployment

Once your model has been successfully fine-tuned, the model status will be updated to **Complete** in the **Status** field, both in the **Custom Models** tab, the **Workload details** page and in the **Dashboard**. This means that the weights have been successfully computed for your model. The model can now be deployed using the [AIM framework](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html). AIMs abstract away the complexities involved in configuring and serving AI models by providing a mechanism that automatically chooses optimized runtime parameters based on user input, hardware, and model specifications.

To deploy the model, navigate to the **Custom Models** tab again, open the actions menu (⋮) next to the model and click **Deploy** (see Figure 9). The **Deploy via AIM Engine** drawer will appear with an auto-generated deployment name. Click **Deploy** to confirm deployment.

Note that the deployment process can take several minutes. Once deployed it will be visible in the **Deployed Models** tab, next to the **Custom Models** tab.

```{image} ./images/deployment.png
:label: deployment
:alt: deployment
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 9: Deploying the model using AIMs.</em>
</p>

#### Interacting with Your Fine-Tuned Model

Once your fine-tuned model is deployed, you can interact with it directly through the **Chat** page. You can chat with your fine-tuned model individually by selecting it from the drop-down in the top-right corner. However, in this blog, we use **Compare mode** to highlight the differences between the fine-tuned model and its original (non-fine-tuned) version. **Compare mode** sends the same prompt to two models simultaneously and displays their responses side by side. This makes it straightforward to evaluate differences in the models’ accuracy, tone, and reasoning, which helps confirm that the fine-tuning had the intended effect.

Before comparing, deploy the original AIM model:

1. Navigate to the **Models** page and open the **AIM Catalog**
2. Find the model card for `Llama 3.2 1B Instruct` and click **Deploy**
3. In the **Deploy AIM** dialog, select the latest container version and keep **Performance metric** set to **Default**
4. Click **Deploy**, then wait until the model is deployed and ready

##### Using Compare Mode

To compare your fine-tuned model against its original version (see Figure 10), follow these steps:

1. Navigate to the **Deployed Models** tab and click the actions menu (⋮) next to the row representing your fine-tuned model deployment, choose **Chat with model**, or open the **Chat** page directly from the main navigation menu on the left
2. Click the **Compare** toggle at the top left of the **Chat** page to activate the dual-panel view
3. In the left-hand drop-down menu, select your **fine-tuned model** from the **Select model** drop-down
4. In the right-hand drop-down menu, select the **original model** (the non-fine-tuned version - which will have a version number, 0.11.0 at the time of this blog post)
5. Enter the following prompt: `"How do I fine-tune a model in AMD AI Workbench?"`
6. Observe how the two models respond differently. The original model produces a generic response, whereas the fine-tuned model answers more accurately based on the data that we've fine-tuned it on

```{image} ./images/chat-compare-models.png
:label: chat-compare
:alt: chat-compare
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 10: Compare model output.</em>
</p>

In figure 10, we demonstrated how the fine-tuned model responds accurately to in-domain questions, i.e., those covered by the dataset it was fine-tuned on. However, as noted earlier, fine-tuned models can become overconfident and produce hallucinated answers, especially when the training data only includes affirmative responses. That is the case for the dataset used in this blog post. To test the fine-tuned model, we ask the following: "How do I make a cup of coffee with AMD AI Workbench?"

```{image} ./images/brew-coffee-example.png
:label: fine-tuned-coffee-example
:alt: fine-tuned-coffee-example
:width: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 11: Overconfident example answer.</em>
</p>

As you can see in Figure 11 above, the fine-tuned AIM model recalls and provides an answer for how to brew coffee with AMD AI Workbench, although this is of course not a feature of AMD AI Workbench. This behaviour stems from the model only ever seeing positive answers to questions about its specialization topic. To mitigate this sort of behaviour, the training data could also include questions where the answer is negative. Additionally, other methods to verify the facts (such as Retrieval Augmented Generation) could be useful.

The above coffee brewing example shows how fine-tuned models can be overconfident in their answers and highlights the importance of choosing the right dataset, method and model based on your use case as well as evaluating and testing the model once it's fine-tuned.

Feel free to try the fine-tuned AIM and see how it responds to both domain questions as well as questions outside of the domain. It should be able to respond to questions regarding the AMD enterprise AI reference stack and for off-topic questions, it should refuse, as shown in Figures 1 and 2. As the coffee example illustrates, it can still produce overconfident responses, so evaluate the model carefully on your own prompts.

#### Cleaning Up

When you no longer need these deployments, open the AMD AI Workbench **Dashboard**. Find your fine-tuned model and the original AIM model in the list, then choose **Delete** for each one.

## Summary

In this blog post, we walked through the fine-tuning workflow in AMD AI Workbench; configuring secrets, uploading dataset, fine-tuning an AIM model,  deploying the fine-tuned AIM, interacting with it and showing what can happen to a fine-tune model when only trained on positive answers about its specialization topic. You've learnt how fine-tuning adapts an already pre-trained model to a specific domain and compared the differences between a base model and a fine-tuned model. To apply this to your own work, select a suitable dataset and model for your use case, adapt parameters as needed and fine-tune your own AIM LLM.

## Additional Resources

If you are interested in exploring more of the features in AMD AI Workbench, see the previous blog post on leveraging AMD AI Workbench and autoscaling:

- [Leveraging AMD AI Workbench and Autoscaling to Scale LLM Inference for Optimal Resource Utilization](https://rocm.blogs.amd.com/software-tools-optimization/eaisuite-autoscaling/README.html)

If you are interested in learning more on the fine-tuning topic, see the following previous blog posts:

- [VLM Fine-Tuning for Robotics on AMD Enterprise AI Suite](https://rocm.blogs.amd.com/artificial-intelligence/vlm-finetune-rocm/README.html)
- [Fine-Tune LLMs for Proteins with AMD Enterprise AI Suite](https://rocm.blogs.amd.com/artificial-intelligence/rocm-finetune-protein/README.html)

Additional technical documentation for you to continue to explore the AMD enterprise AI reference stack is readily available and can be found at the links below:

- [Overview of the AMD enterprise AI reference stack](https://enterprise-ai.docs.amd.com/en/latest/platform-overview.html)
- [Installing the AMD enterprise AI reference stack](https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/on-premises-installation.html)
- [AIM catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html)
- [AIM deployment overview](https://enterprise-ai.docs.amd.com/en/latest/aims/deployment_overview.html)
- [AIM custom configuration profiles](https://enterprise-ai.docs.amd.com/en/latest/aims/custom_profiles.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
