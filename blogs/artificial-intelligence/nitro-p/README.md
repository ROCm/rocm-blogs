---
blogpost: true
blog_title: "Nitro-P: Pixel-Space Text-to-Image via Hierarchical Semantic-to-Pixel Generation" 
date: "20 Nov 2026"
author: "Jingai Yu, Tong Shen, Dong Zhou, Dong Li, Emad Barsoum"
thumbnail: 'nitro-p-blog-thumbnail.png'
tags: "Diffusion Model, PyTorch, GenAI"
category: "Applications & models"
target_audience: "ai developers and enthusiast"
key_value_propositions: "Nitro-P, a lightweight diffusion transformer model for high-quality pixel-space image generation, showcases resource-friendly inference on AMD Instinct™ MI300X GPUs."
language: English
myst:
    html_meta:
        "author": "Jingai Yu, Tong Shen, Dong Zhou, Dong Li, Emad Barsoum"
        "description lang=en": "Nitro-P is an E-MMDiT-based pixel-space text-to-image model. It uses lower FLOPs while delivering evaluation metrics and visual quality comparable to latent-space text-to-image models."
        "keywords": "Nitro-P, E-MMDiT, text-to-image generation, pixel-space"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Jingai Yu, Tong Shen, Dong Zhou, Dong Li, Emad Barsoum"
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

# Nitro-P: Pixel-Space Text-to-Image via Hierarchical Semantic-to-Pixel Generation

We present Nitro-P, an end-to-end pixel-space text-to-image (T2I) diffusion framework. Built upon our previously released [Nitro-E](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html#nitro-e-model), Nitro-P extends the Nitro family from latent-space T2I generation to a VAE-free pixel-space design. To achieve both high generation quality and computational efficiency, Nitro-P adopts a hierarchical semantic-to-pixel design, consisting of a patch-level semantic modeling component based on [E-MMDiT](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html#nitro-e-model) and a pixel-level modulation-guided reconstruction component implemented as a lightweight U-Net [1] decoder. By operating directly in pixel space, Nitro-P removes the VAE decoder from the inference pipeline. This benefit is most visible in few-step sampling, where latent-space models still incur a non-negligible final decoding pass. Nitro-P achieves a throughput of 10.26 samples/s on a single AMD Instinct&trade; MI300X GPU for 512px image generation with a batch size of 32. Its distilled one-step variant further boosts throughput to 90.02 samples/s.

Figure 1 summarizes a detailed comparison against representative pixel-space diffusion models, showing that Nitro-P achieves promising GenEval performance with competitive throughput, while its distilled one-step variant substantially improves efficiency with only limited degradation in generation quality.

As part of AMD’s ongoing commitment to open-source research, we release both the model weights and the complete training code to support reproducibility and broader adoption. We hope this work can serve as a practical baseline for future studies and help advance progress in pixel-space generation. In this blog, we provide an overview of the model design and the key technical details behind it.

```{figure} ./images/Figure_geneval.png
:align: center
:alt: Scaling performance
:width: 60%
```

<p align="center">Figure 1. Comparison with other models on GenEval and throughput.</p>

## Nitro Family and Nitro-P

Building on AMD's earlier Nitro series, Nitro-P extends [Nitro-E](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html#nitro-e-model) from efficient latent-space T2I generation to end-to-end, VAE-free pixel-space generation. [Nitro-1](https://www.amd.com/en/developer/resources/technical-articles/amd-nitro-diffusion-one-step-text-to-image-generation-models.html) and [Nitro-T](https://rocm.blogs.amd.com/artificial-intelligence/nitro-t-diffusion/README.html), together with [Nitro-E](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html#nitro-e-model), represent AMD's earlier explorations in latent-space T2I. The main highlights of Nitro-P are as follows:

- VAE-free pixel-space T2I with a hierarchical semantic-to-pixel design (E-MMDiT + modulation-guided reconstruction).
- Distilled **one-step** checkpoints achieve throughputs of **90.02** samples/s at 512x512 and **26** samples/s at 1024x1024, both with a batch size of 32 on a single AMD Instinct&trade; MI300X GPU.
- The full-step model achieves competitive GenEval scores of **0.88** at 512×512 and **0.87** at 1024×1024 (see Results).
- Fully supported by AMD ROCm™ Software; trained from scratch on public data, with code and weights released.

## Background and Architecture

### Text-to-Image and Latent-Space Diffusion

In T2I synthesis, latent-space diffusion models have been particularly effective in practice, typically using a two-stage pipeline in which a Variational Autoencoder (VAE) [2] first compresses images into a lower-dimensional representation and diffusion is then performed in the latent space, substantially reducing the cost of training and sampling.

This design improves computational efficiency but makes generation dependent on the representation learned by the VAE. Visual information that is not faithfully retained during compression cannot be modeled directly by the diffusion process, and the generated latent representation must still be converted back to RGB by the VAE decoder. Moreover, the autoencoder and diffusion model are trained with separate objectives rather than optimized jointly. These trade-offs have motivated growing interest in performing generative modeling directly in pixel space [5–7].

### From Latent-space to Pixel-Space Diffusion

Pixel-space diffusion, however, is not free. The main challenge is computational cost. Treating every pixel as an individual Transformer [3] token would produce extremely long sequences, making self-attention impractical at high resolutions. Pixel-space Transformer models therefore commonly group neighboring pixels into patches.

These operations work as follows:

- **Patchification**: before feeding the input into the Transformer, each neighboring $p \times p$ pixel block is packed into a single token with channel dimension $p \times p \times 3$.
- **Unpatchification**: at the output side, the Transformer predicts tokens with channel dimension $p \times p \times 3$, which are then reshaped back into the pixel image.

This design greatly improves computational efficiency, but it shifts the difficulty from attention cost to pixel decoding.

While patchification makes pixel-space Transformers computationally tractable, it also increases the dimensionality of the pixel values predicted from each token. As observed in JiT [4], it is difficult for the model to directly predict such high-dimensional $p \times p \times 3$ outputs, especially when the prediction target is noise or velocity. A large patch size reduces token count and attention cost but blurs boundaries and textures; a small patch size preserves details but increases sequence length and quadratic attention cost. A single DiT backbone struggles to excel at both global semantics and local fidelity.
DeCo [6] and PixelDiT [7] use local decoders to predict pixel values within each patch, and then assemble the decoded outputs through unpatchification.

More broadly, DiP [5], DeCo [6], and PixelDiT [7] show that the key is not to apply a full-resolution DiT [8] over every pixel, but to decouple global structure modeling from local detail restoration with a lightweight pixel decoder. In this paradigm, a diffusion Transformer handles text-conditioned semantics at coarse resolution, while a lightweight local module recovers full-resolution RGB. The specific decoder architectures differ across methods: DiP uses a shallow U-Net, DeCo employs attention-free linear decoder blocks and PixelDiT uses a pixel-level DiT with self-attention. This has become a mainstream design pattern for efficient pixel-space T2I.

### Nitro-P: A Hierarchical Semantic-to-Pixel Framework

```{figure} ./images/nitrop_struct.png
:align: center
:alt: Nitro-P architecture overview
:width: 70%
```

<p align="center">Figure 2. Overview of Nitro-P. </p>

Building on this paradigm, Nitro-P introduces a hierarchical semantic-to-pixel architecture for end-to-end pixel-space T2I generation: it avoids the VAE bottleneck of latent diffusion and replaces direct output prediction from patch tokens followed by unpatchification with a dedicated full-image convolutional decoder. An overview of the Nitro-P architecture is shown in Figure 2.

At the **patch-level semantic modeling stage**, Nitro-P builds on our [Nitro-E](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html), reusing the efficiency-oriented designs from E-MMDiT, including token compression, positional reinforcement, alternating subregion attention, and AdaLN-affine. Operating on 16×16 RGB patches, E-MMDiT models text-conditioned structure and outputs a coarse semantic feature map at 1/16 resolution, rather than directly predicting RGB values via unpatchification. Text conditioning uses Qwen3-1.7B [9] features projected by a caption embedder with a learnable positional embedding; the resulting tokens are further refined by four timestep-conditioned TextRefineBlocks before entering joint attention.

Consistent with recent observations in MiniT2I [10], we find that direct RGB prediction through unpatchification can produce visible patch artifacts in practice. We further observe that this issue becomes more pronounced under one-step distillation, where the model must produce the final image in a single pass and local reconstruction errors are no longer averaged away over multiple denoising steps. These findings reinforce the need for an explicit pixel reconstruction module, rather than predicting RGB values directly from patch tokens.

At the **pixel-level reconstruction stage**, a natural baseline is independent decoding: each patch reconstructs its own pixels in isolation, after which the patches are assembled by unpatchification. Because neighboring patches never interact at this stage, this design tends to produce visible seams at patch boundaries. Nitro-P instead adopts joint decoding with a lightweight U-Net decoder (similar in spirit to DiP): the full-resolution RGB velocity is reconstructed through multi-scale convolutions and skip connections over the entire image. This convolutional backbone models spatial neighborhoods and preserves local continuity, so patch artifacts are largely suppressed. As shown in Figure 3, joint decoding yields substantially fewer visible patch artifacts than independent decoding. To further strengthen semantic control, Nitro-P introduces a modulation-guided interface between E-MMDiT and the U-Net decoder, as detailed in the next section.

```{figure} ./images/merged_artifacts.jpg
:align: center
:alt: Scaling performance
:width: 50%
```

<p align="center">Figure 3. Visual comparison of patch artifacts.</p>

#### Modulation-Guided Semantic Injection

An important design question is how to transfer the patch-level E-MMDiT feature map into the pixel-level U-Net. We investigate two alternatives: direct feature concatenation and modulation-guided injection. The concatenation baseline directly merges the E-MMDiT features with the U-Net features. In our modulation-guided design, the E-MMDiT feature map is projected into spatially varying scale and shift parameters, which modulate the normalized U-Net activations at the bottleneck. This design is related to the semantic modulation used in DeCo [6], but applies it to a different reconstruction architecture. In DeCo, the low-resolution semantic features produced by the DiT are upsampled and transformed into adaptive parameters that modulate the dense decoder queries in each pixel-decoder block. In Nitro-P the two-dimensional E-MMDiT feature map instead modulates the bottleneck of a full-image U-Net, enabling adaptive semantic control during multi-scale reconstruction across patch boundaries.

To isolate the contribution of modulation-guided semantic injection, we train a concat-only baseline that uses the same E-MMDiT backbone and U-Net decoder. All other training and inference settings are kept identical. As shown in Table 2, modulation-guided decoding consistently improves semantic fidelity over concat-only injection, producing more accurate object colors, more accurate object counts, and more coherent global layout and structural relationships. By comparison, the concat-only baseline often weakens the text-conditioned signal, leading to color mismatches, missing or duplicated objects, and inconsistent local details.

```{figure} ./images/concat_adain_comparison.jpg
:align: center
:alt: Scaling performance
```

<p align="center">Table 2. Side-by-side visual comparison under identical prompts. The left column shows results from the concat-only baseline, while the right column shows results from Nitro-P with AdaIN-guided decoding.</p>

## Experiments and Results

### Data

We curated a dataset of 25M images containing both real and synthetic images. Compared with Nitro-E, we additionally include high-quality images from MiniT2I [10]. The details are described in Table 3. All the data are publicly available without any internal private data.

<div style="width: 70%; margin: 0 auto; font-size: 0.95em;">

| Source | Number of Images | Type of Images | Generation Method |
| ------------ | --------------- | ---------------- | ------------- |
| Segment Anything 1B ([SA1B](https://arxiv.org/abs/2304.02643)) | 11.1M | Real | Real images annotated via LLaVA |
| [JourneyDB](https://journeydb.github.io/) | 4.4M | Synthetic | Midjourney generated images |
| FLUXDB | 9.5M | Synthetic | Generated using FLUX.1-dev [16]  with prompts from DataComp |
| [High-quality data from MiniT2I](https://peppaking8.github.io/#/post/minit2i) | 120K | Synthetic | A curated alignment fine-tuning mixture of [BLIP3o-60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k), [LAION DALL-E 3 Discord set](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset), and [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) |

</div>
<p align="center">Table 3. Details of our training data. It consists of a mix of real and synthetic images and captions totaling 25M text-image pairs.</p>

### Training Strategies

Nitro-P uses Qwen3-1.7B as the text encoder. Compared with Llama 3.2 1B [11], Qwen3-1.7B demonstrates a better understanding of color and spatial relationships.

Beyond the standard flow-matching [12] objective, we adopt a frequency-aware flow-matching loss inspired by DeCo, emphasizing perceptually important frequencies in the Discrete Cosine Transform (DCT) domain. Following Nitro-E, we also use REPA [13] to align early transformer representations with frozen DINOv2 [14] features for faster and more stable convergence.

Our model is trained in a multi-stage curriculum, progressively increasing the image resolution from 256×256 to 512×512, and finally to 1024×1024. The early stages use a large-scale mixed dataset for broad visual coverage, while the later stages transition to a smaller high-quality data mixture for alignment-oriented fine-tuning. This design allows the model to first learn general text-to-image generation capabilities and then improve prompt fidelity and visual quality through high-quality supervision. For one-step generation, we further distill the model with a combination of adversarial and diffusion losses. The discriminator uses a frozen E-MMDiT feature extractor with a randomly initialized prediction head.

### Implementation Details

Our experiments are conducted using the PyTorch for [ROCm training Docker image](https://hub.docker.com/r/rocm/pytorch-training/tags) [15], which provides an optimized PyTorch environment for training models on AMD Instinct&trade; MI325X and MI300X accelerators. Our implementation is built on the training engine [Accelerate](https://github.com/huggingface/accelerate) with mixed precision training using the Bfloat16 datatype. Leveraging the large memory capacity of MI300X, we train the model with a batch size of 2048 using the native distributed data parallel.

For metrics, we adopt four widely used benchmarks:

- [GenEval](https://github.com/djghosh13/geneval) and [DPG](https://github.com/TencentQQGYLab/ELLA): focus on text-alignment.
- [ImageReward](https://github.com/zai-org/ImageReward) and [HPSv2.1](https://github.com/tgxs002/HPSv2) measure human preference quality.

### Results

We compare Nitro-P with recent open pixel-space text-to-image models in Table 4, and summarize the quality–throughput trade-off in Figure 1. At 512×512, Nitro-P reaches a GenEval score of 0.88 with 10.26 samples/s, offering a strong balance against methods such as DeCo and MiniT2I-L, while using fewer main-network TFLOPs. The one-step distilled model further increases throughput to 90.02 samples/s with a moderate GenEval drop (0.85). At 1024×1024, Nitro-P surpasses PixelDiT in quality (0.87 vs. 0.74). All models are trained only on public data. Qualitative examples are shown in Figure 4.

<html>
<head>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      padding: 20px;
      max-width: 100%;
      overflow-x: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      background-color: light-dark(white, black);
      border: 1px solid light-dark(#ddd, #ffffff);
    }
    th, td {
      padding: 12px 15px;
      text-align: center;
      border: 1px solid light-dark(#ddd, #f4f4f4);
      font-size: 14px;
    }
    thead {
      background-color: light-dark(#f5f7fa, #37383a);
    }
    thead th {
      position: sticky;
      top: 0;
      background-color: light-dark(#f5f7fa, #5b5b5b);
      font-weight: 600;
      color: light-dark(#000000, #ffffff);
      border: 1px solid light-dark(#ddd, #ffffff);
    }
    tbody th[colspan] {
      background-color: light-dark(#edf2f7, #383D46);
      font-weight: 600;
      text-align: center;
    }
    td strong {
      font-weight: bold;
      color: light-dark(#000000, #ffffff);
    }
    td ins {
      text-decoration: underline;
      color: light-dark(#000000, #ffffff);
    }
    tr:hover {
      background-color: light-dark(#f9fafb, #878a90);
    }
    tr {
      border-bottom: 1px solid light-dark(#ddd, #ffffff);
    }
    td, th {
      border-right: 1px solid light-dark(#ddd, #ffffff);
    }
    td:last-child, th:last-child {
      border-right: 1px solid light-dark(#ddd, #ffffff);
    }
  </style>
</head>
<body>
  <table style="width: 100%">
    <tr align="center">
        <td><b>Methods</b></td>
        <td><b>Throughput<br>(samples/s)</b></td>
        <td><b>Parameters<br> Main network<br> (M)</b></td>
        <td><b>TFLOPs<br>Main network </b></td>
        <td><b>Dataset Size<br>(M)</b></td>
        <td><b>GenEval↑</b></td>
        <td><b>IR↑</b></td>
        <td><b>HPS↑</b></td>
        <td><b>DPG↑</b></td>
    </tr>
    <tr>
        <th colspan="15" align="center">512x512</th>
    </tr>
    <tr align="center">
        <td>PixelGen</td>
        <td>7.6</td>
        <td>1100</td>
        <td>1.42</td>
        <td>36</td>
        <td>0.79</td>
        <td>0.33</td>
        <td>28.08</td>
        <td>80.03</td>
    </tr>
    <tr align="center">
        <td>PixNerd</td>
        <td>6.9</td>
        <td>1200</td>
        <td>1.54</td>
        <td>45</td>
        <td>0.73</td>
        <td>0.48</td>
        <td>27.83</td>
        <td>80.90</td>
    </tr>
    <tr align="center">
        <td>DeCo</td>
        <td>6.13</td>
        <td>1100</td>
        <td>1.45</td>
        <td>42</td>
        <td>0.86</td>
        <td>0.48</td>
        <td>27.75</td>
        <td>81.40</td>
    </tr>
    <tr align="center">
        <td>PixelDiT</td>
        <td>5.54</td>
        <td>1300</td>
        <td>1.43</td>
        <td>26</td>
        <td>0.78</td>
        <td>-</td>
        <td>-</td>
        <td>83.70</td>
    </tr>
    <tr align="center">
        <td>MiniT2I-L</td>
        <td>2.4</td>
        <td>912</td>
        <td>1.49</td>
        <td>12</td>
        <td>0.88</td>
        <td>0.69</td>
        <td>27.73</td>
        <td>85.90</td>
    </tr>
    <tr align="center">
        <td>MiniT2I-4step</td>
        <td>-</td>
        <td>258</td>
        <td>-</td>
        <td>12</td>
        <td>0.84</td>
        <td>-</td>
        <td>-</td>
        <td>83.23</td>
    </tr>
    <tr align="center">
        <td>Nitro-P</td>
        <td>10.26</td>
        <td>679</td>
        <td>0.74</td>
        <td>25</td>
        <td>0.88</td>
        <td>0.98</td>
        <td>30.04</td>
        <td>86.33</td>
    </tr>
    <tr align="center">
        <td>Nitro-P-1step</td>
        <td>90.02</td>
        <td>679</td>
        <td>0.74</td>
        <td>25</td>
        <td>0.85</td>
        <td>0.99</td>
        <td>29.79</td>
        <td>83.54</td>
    </tr>
    <tr>
        <th colspan="15" align="center">1024x1024</th>
    </tr>
    <tr align="center">
        <td>PixelDiT</td>
        <td>2.05</td>
        <td>1300</td>
        <td>6.18</td>
        <td>26</td>
        <td>0.74</td>
        <td>0.95</td>
        <td>30.98</td>
        <td>83.50</td>
    </tr>
    <tr align="center">
        <td>Nitro-P</td>
        <td>2.44</td>
        <td>679</td>
        <td>3.35</td>
        <td>25</td>
        <td>0.87</td>
        <td>1.01</td>
        <td>30.48</td>
        <td>85.49</td>
    </tr>
    <tr align="center">
        <td>Nitro-P-1step</td>
        <td>26.00</td>
        <td>679</td>
        <td>3.35</td>
        <td>25</td>
        <td>0.83</td>
        <td>0.98</td>
        <td>29.93</td>
        <td>83.08</td>
    </tr>
  </table>
</body>
</html>
<p align="center">Table 4. Comparison of Nitro-P with other pixel-space diffusion models.</p>

To isolate the effect of removing the VAE decoder from the inference path, we further compare end-to-end throughput of Nitro-P against Nitro-E on the same AMD Instinct&trade; MI300X GPU (batch size 32). Nitro-E still requires a final VAE decoding pass after denoising, whereas Nitro-P is VAE-free and operates directly in pixel space. As shown in Table 5, this difference is largest in the few-step regime: at 512×512, the one-step distilled Nitro-P reaches 90.02 samples/s, versus 44.13 samples/s for Nitro-E. The gap narrows as the number of sampling steps increases, and at the full 20-step setting Nitro-E is faster, because the denoising backbone dominates runtime and the latent-space path remains cheaper per step. The same trend holds at 1024×1024. These results indicate that Nitro-P is particularly well suited to low-NFE, high-throughput serving.

<div style="width: 95%; margin: 0 auto; font-size: 0.95em;">
<table style="width: 100%">
  <tr align="center">
    <td><b>Models</b></td>
    <td><b>1-step TP<br></b></td>
    <td><b>2-step TP<br></b></td>
    <td><b>5-step TP<br></b></td>
    <td><b>20-step TP<br></b></td>
  </tr>
  <tr>
    <th colspan="5" align="center">512×512</th>
  </tr>
  <tr align="center">
    <td>Nitro-E</td>
    <td>44.13</td>
    <td>41.39</td>
    <td>34.35</td>
    <td>18.83</td>
  </tr>
  <tr align="center">
    <td>Nitro-P</td>
    <td>90.02</td>
    <td>64.83</td>
    <td>33.76</td>
    <td>10.26</td>
  </tr>
  <tr>
    <th colspan="5" align="center">1024×1024</th>
  </tr>
  <tr align="center">
    <td>Nitro-E</td>
    <td>11.00</td>
    <td>10.28</td>
    <td>8.59</td>
    <td>4.66</td>
  </tr>
  <tr align="center">
    <td>Nitro-P</td>
    <td>26.00</td>
    <td>16.86</td>
    <td>8.42</td>
    <td>2.44</td>
  </tr>
</table>
</div>
<p align="center">Table 5. End-to-end throughput (samples/s) of Nitro-E and Nitro-P on a single AMD Instinct&trade; MI300X GPU with batch size 32. TP denotes throughput.</p>

## More Visual Examples

```{figure} ./images/merged_grid.png
:align: center
:alt: Scaling performance
```

<p align="center">Figure 4. Images generated by Nitro-P</p>

## Summary

We present Nitro-P, a VAE-free pixel-space extension of Nitro-E for text-to-image generation. Built on a hierarchical semantic-to-pixel design, Nitro-P reuses the efficiency-oriented E-MMDiT recipe for patch-level semantic modeling and pairs it with a lightweight U-Net for pixel-level reconstruction under modulation-guided semantic injection. Trained entirely on public data on AMD Instinct&trade; MI300X GPUs and released with open code and checkpoints—including one-step distilled variants—Nitro-P offers a practical baseline for efficient pixel-space generation.

## Additional Resources

Model: [amd/Nitro-P · HuggingFace](https://huggingface.co/amd/Nitro-P)

Code: [AMD-AIG-AIMA/Nitro-P](https://github.com/AMD-AGI/Nitro-P)

Related Work from AMD team:

- [Nitro-E: A 304M Diffusion Transformer Model for High Quality Image Generation — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html)
- [AMD Nitro Diffusion: One-Step Text-to-Image Generation Models](https://www.amd.com/en/developer/resources/technical-articles/amd-nitro-diffusion-one-step-text-to-image-generation-models.html)
- [Nitro-T: Training a Text-to-Image Diffusion Model from Scratch in 1 Day — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/nitro-t-diffusion/README.html)

Other useful Links

- Use the [public PyTorch ROCm Docker
    images](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html)
    that enable optimized training performance out-of-the-box.

- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm
    --- ROCm
    Blogs](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)

- [Accelerating Large Language Models with Flash Attention on AMD GPUs
    --- ROCm
    Blogs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)

- [Accelerate PyTorch Models using torch.compile on AMD GPUs with ROCm
    --- ROCm
    Blogs](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)

## References

1. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.

2. Kingma, D. P., & Welling, M. Auto-Encoding Variational Bayes. arXiv, 2013.

3. Dosovitskiy, A., et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.

4. Li, T., & He, K. Back to Basics: Let Denoising Generative Models Denoise. arXiv, 2025.

5. Chen, Z., et al. DiP: Taming Diffusion Models in Pixel Space. CVPR, 2026.

6. Ma, Z., et al. DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation. CVPR, 2026.

7. Yu, Y., et al. PixelDiT: Pixel Diffusion Transformers for Image Generation. CVPR, 2026.

8. Peebles, W., & Xie, S. Scalable Diffusion Models with Transformers. ICCV, 2023.

9. Qwen Team. Qwen3 Technical Report. arXiv, 2025. https://arxiv.org/abs/2505.09388

10. Wang, X., et al. MiniT2I: A Minimalist Baseline for Text-to-Image Generation. Technical blog, 2026.

11. Meta. Llama 3.2 Model Family. 2024. https://huggingface.co/meta-llama/Llama-3.2-1B

12. Esser, P., et al. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. arXiv, 2024.

13. Yu, S., et al. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think. ICLR, 2025.

14. Oquab, M., et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv, 2024.

15. PyTorch for ROCm Training Docker Image. Docker Hub, 2025.

16. Batifol, S., et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv, 2025

## System Configuration

Our testbed system was configured with an AMD Instinct&trade; MI300X GPU running
tests done by AMD on 05/17/2025, results may vary based on
configuration, usage, software version, and optimizations.

AMD Instinct&trade; MI300X platform
System Model: Supermicro AS-8125GS-TNMR2
CPU: 2x AMD EPYC 9575F 64-Core Processor
NUMA: 1 NUMA node per socket. NUMA auto-balancing disabled
Memory: 2304 GiB (24 DIMMs x 96 GiB Micron Technology
MTC40F204WS1RC48BB1 DDR5 4800 MT/s)
Disk: 16,092 GiB (4x SAMSUNG MZQL23T8HCLS-00A07 3576 GiB, 2x SAMSUNG
MZ1L2960HCJR-00A07 894 GiB)
GPU: 8x AMD Instinct&trade; MI300X 192GB HBM3 750W
Host OS: Ubuntu 22.04.4
System BIOS: 3.5
System Bios Vendor: American Megatrends International, LLC.
Host GPU Driver: (amdgpu version): ROCm 6.3.4
Docker image: rocm/pytorch-training:v25.5

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF SUCH THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved.
