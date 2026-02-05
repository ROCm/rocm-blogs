---
blogpost: true
blog_title: "Micro-World: First AMD Open-Source World Models for Interactive Video Generation"
date: 05 Feb 2026
author: 'Yu Geng, Wensong Chan, Dong Zhou, Dong Li, Emad Barsoum'
thumbnail: 'micro_world_thumbnail.png'
tags: GenAI, Multimodal
category: Applications & models
target_audience: AI / ML Researchers and Engineers
key_value_propositions: Micro-World demonstrates effective action control in real-world scenarios with AMD GPUs. The model will be open sourced to facilitate future research of world model.
language: English
myst:
    html_meta:
        "author": "Yu Geng, Wensong Chan, Dong Zhou, Dong Li, Emad Barsoum"
        "description lang=en": "Micro-World is an action-controlled interactive world model designed to generate high-quality, open-domain scenes."
        "keywords": "Micro-World, world model, generative model, text-to-video generation, image-to-video generation, interactive model, action-controlled"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yu Geng, Wensong Chan, Dong Zhou, Dong Li, Emad Barsoum"
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

# Micro-World: First AMD Open-Source World Models for Interactive Video Generation

World models aim to simulate aspects of the real world, enabling more effective training and exploration of AI agents and ultimately paving the way toward richer forms of digital life. Games can be viewed as another form of world simulation, and their data is relatively easy to collect and annotate, making them a natural playground for building and studying world models. GameNGen [1] has demonstrated the potential of this direction, while works such as GameFactory [2], Matrix-Game [3], and Hunyuan-GameCraft [4] further showcase strong performance in game-oriented world modeling. However, these projects are either fully closed-sourced or release only partial components (typically inference-only), which limits reproducibility and community-driven progress.

In this blog, we introduce [**Micro-World**](https://github.com/AMD-AGI/Micro-World), an action-controlled interactive world model designed to generate high-quality, open-domain scenes. The name Micro-World reflects our motivation to study world modeling through compact, controllable environments. Our training data is collected from Minecraft, a minimal yet expressive game world, and our models are intentionally designed to be lightweight and efficient, enabling practical training and deployment.

Built on top of the Wan2.1 [5] family of models, we train both image-to-world (I2W) and text-to-world (T2W) variants to support a wide range of use cases. To foster open research and practical adoption within the community, we release the model weights, complete training and inference code, and a curated dataset specifically tailored for controllable world modeling. Our evaluations demonstrate that Micro-World can generate high-quality video while faithfully following the provided action instructions.

The key takeaways of our blog are outlined below (see also Figure 1):

- Data Collection: We collected over 6,000 gameplay clips, each consisting of 81 frames. The dataset includes both keyboard and mouse actions to enable flexible control, and each clip is annotated with text captions and action labels.
- Open-Domain Control: Micro-World accepts action inputs and generates controllable videos in open-domain scenarios.
- Model Design: We design a dedicated action processing module that separately encodes continuous mouse actions and discrete keyboard actions. Action features are injected into the model using either ControlNet or Adaptive Layer Normalization (adaLN).
- Fully Open-Sourced: Built upon Wan 2.1, Micro-World is fully open-sourced, including the curated dataset as well as the complete training and inference code.
- AMD ROCm™ Software Support: The model is fully trained and deployed on AMD Instinct™ GPUs within the ROCm ecosystem.

```{figure} ./images/takeaways.png
:align: center
:alt: takeaways
Figure 1. Key takeaways.
```

## AMD Solution for World Modeling

Most existing world-model research in the gaming domain focuses on overfitting to a single game environment. For instance, GameNGen [1] accurately simulates Doom, modeling not only player actions but also UI elements such as the health indicators. Similarly, works such as Playable Game Generation [6] for Mario and Oasis [7] for Minecraft remain constrained to their respective game environments. As a result, these approaches struggle to generalize beyond the domains on which they are trained.

However, game data should be regarded as a means rather than an end. The broader objective is to leverage game environments as a proxy for learning transferable world dynamics that extend to open-domain scenarios. Recent efforts such as GameFactory [2] and Matrix-Game [3] take encouraging steps in this direction by attempting to transfer knowledge learned from game data to real-world contexts, thereby motivating further exploration in this area. Despite their promising ideas, these approaches are either not open-sourced or only partially released, which limits reproducibility and slows progress within the open-source community.

In this blog, we present an open-domain world model designed to address these limitations. Inspired by GameFactory [2], we adopt a two-stage training paradigm to transfer action knowledge from game environments to open-domain control. In the first stage, we train LoRA [8] weights to adapt the base model to game-style visual distributions. In the second stage, we introduce an action module and jointly learn action control while merging the trained LoRA weights. This design allows the action module to decouple from game-specific visual styles and focus on learning generalized action dynamics. At inference time, the LoRA weights can be removed, enabling the model to operate directly in open-domain scenarios.

To foster transparency and accelerate research in world-model development, we fully open-source our contributions, including model weights, complete training and inference pipelines, and a curated dataset.

## Technical Details

### Data Collection

Although numerous action datasets exist for training, most suffer from severe imbalances in action class distributions. For instance, during real gameplay, players press the forward key far more frequently than the backward key, which can bias the model and hinder effective action learning. GameFactory addresses this issue by collecting data through randomly executed actions to achieve a more balanced distribution. However, such random action sequences often result in unnatural movements, leading to abrupt frame-to-frame transitions that impede stable model learning.

To overcome these limitations, we construct our own dataset. Rather than fully randomizing actions at each timestep, we enforce a temporal consistency rule in which each action is maintained for a randomly sampled duration. In addition, we constrain the magnitude of sampled mouse movements to prevent unnatural camera motion and abrupt viewpoint changes.

We leverage the Minecraft API [9] to collect game data, as it enables us to obtain diverse biomes as well as varying weather and lighting conditions—properties that are desirable for improving generalization to real-world scenarios.

For the action space, we record keyboard controls including W (forward), A (left), S (backward), D (right), Ctrl (sprint), Shift (sneak), Space (jump), along with mouse movement. By randomly sampling biomes and executing these actions, we collected 6,000 clips, each consisting of 81 frames, forming our Game dataset. We use miniCPM [10] to generate captions for each clip.

### Base Model

The Wan2.1 series has attracted significant attention from the community. Owing to its strong performance and practical utility, many developers have chosen to build their own models on top of Wan2.1, forming a thriving ecosystem. We also adopt Wan2.1 as our base model to contribute to this collective effort. Our work includes both T2W and I2W variants, built upon Wan2.1 1.3B T2V and Wan2.1 14B I2V, respectively.

### Action processing module

The input actions include continuous mouse actions $M \in R^{n×2}$ and discrete keyboard actions $K \in R^{n×7}$, both aligned with the frame sequence. Since past actions can influence the current state, we adopt a sliding history window that groups past and current actions $A^{i-w}…A^i$ as the input for the current feature $F^i$.

For mouse actions, we first aggregate the historical and current steps and then apply an MLP to align their feature dimensionality. For keyboard actions, we embed the discrete inputs and add positional encodings. The embedded features are then grouped using the same history window and passed through an MLP for dimensional alignment.

Finally, the mouse and keyboard features are concatenated and processed by another MLP to better match the diffusion model’s feature space. All action features are preprocessed at the beginning of the inverse diffusion process and forwarded to subsequent modules.

### Action injection

There are three primary strategies for injecting action information into the model:

- Direct fusion through addition or concatenation of action features with noise features.
- Cross-attention based integration, which enables the model to capture intra-domain relationships.
- Adaptive Layer Normalization (adaLN [11]), where action features are encoded together with the timestep embedding.

We experimented with all three approaches and selected adaLN [11] and ControlNet [12]. ControlNet is used in the T2W model (Figure 2\(c\)), while adaLN is adopted for the I2W model (Figure 2(b)). We choose adaLN for its lightweight parameter footprint, and ControlNet for its learning capability. The overall module architecture is illustrated in Figure 2.

```{figure} ./images/model_architecture.jpg
:align: center
:alt: model architecture
Figure 2. (a). Action grouping process.
(b). AdaLN formulation.
\(c\). ControlNet formulation. Dashed arrows indicate the zero linear process.
```

### Multi-Stage Training

A major challenge in open-domain action control is data scarcity—collecting and annotating large-scale, diverse action datasets is often impractical. To address this, we adopt a two-stage training strategy.
In the first stage, we train the model on the game dataset to learn a game-style LoRA, capturing domain-specific style and appearance priors. In the second stage, we merge this LoRA into the base model and train only the action module to learn generalizable action control, independent of the stylistic biases from the game data.

It is important to note that the original model parameters remain frozen during both stages. This decoupling allows the action module to focus solely on the underlying action dynamics, enabling stronger transferability to open-domain scenarios.

### Open-domain Inference

The model can naturally perform inference on game-style videos. However, for open-domain video generation, we remove the LoRA parameters to drop game style and rely on the knowledge encoded in the base model. Thanks to the decoupling training strategy, the action module remains domain-agnostic and can still drive the model to generate action-controlled videos.

### Implementation Details

Our model is implemented in PyTorch. We also employed advanced training libraries, including Accelerate, Diffusers, and Deepspeed to support multi-machine and mixed-precision training. Training and inference are performed using AMD Instinct MI325X GPUs.

## Quantitative Comparison

To better assess the performance of Micro-World, we evaluate it across multiple dimensions and compare it against Oasis on the GameWorld Benchmark [3].

**Image Quality**. We adopt standard visual metrics, including FVD, PSNR, and LPIPS, to measure the perceptual quality of the generated videos.

**Action Controllability**. To evaluate whether the generated videos correctly follow keyboard and mouse inputs, we introduce keyboard precision and camera precision. Specifically, an Inverse Dynamics Model (IDM) is used to predict actions between two consecutive generated frames, and the predicted actions are then compared against the input actions.

**Temporal Quality**. We assess temporal consistency and motion smoothness to measure the stability of backgrounds and scene dynamics over time.

<!-- markdownlint-disable -->

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-za14{border-color:inherit;text-align:left;vertical-align:bottom}
.tg .tg-fll5{border-color:inherit;font-weight:bold;text-align:center;vertical-align:bottom}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-f8tv{border-color:inherit;font-style:italic;text-align:left;vertical-align:top}
</style>

<div align="center">
<table class="tg">
  <tr>
    <th class="tg-fll5" rowspan="2">Model</th>
    <th class="tg-fll5" colspan="3">Image Quality</th>
    <th class="tg-fll5" colspan="2">Action Controllability</th>
    <th class="tg-fll5" colspan="2">Temporal Quality</th>
  </tr>
  <tr>
    <th class="tg-fymr">PSNR ↑</th>
    <th class="tg-fymr">LPIPS ↓</th>
    <th class="tg-fymr">FVD ↓</th>
    <th class="tg-fymr">Keyboard Precision ↑</th>
    <th class="tg-fymr">Camera Precision ↑</th>
    <th class="tg-fymr">Temporal Consistency ↑</th>
    <th class="tg-fymr">Motion Smoothness ↑</th>
  </tr>
  <tr>
    <td class="tg-0pky">Oasis</td>
    <td class="tg-0pky">14.90</td>
    <td class="tg-0pky">0.5641</td>
    <td class="tg-0pky">596.986</td>
    <td class="tg-0pky">0.6474</td>
    <td class="tg-0pky">0.3770</td>
    <td class="tg-0pky">0.9589</td>
    <td class="tg-0pky">0.9875</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ours-I2W</td>
    <td class="tg-0pky">15.88</td>
    <td class="tg-0pky">0.4128</td>
    <td class="tg-0pky">175.371</td>
    <td class="tg-0pky">0.7224</td>
    <td class="tg-0pky">0.5055</td>
    <td class="tg-0pky">0.9629</td>
    <td class="tg-0pky">0.9750</td>
  </tr>
</table>
    <p>
    <strong>Table 1. Game-World Score Benchmark Comparison. </strong>
    </p>
</div>

From Table 1, Micro-World I2W outperforms Oasis across image quality, action controllability and temporal quality. These results illustrate the effectiveness of our model for interactive world-models in gaming environment.

<div align="center">
  <table class="tg">
    <thead>
      <tr>
        <th class="tg-fymr">Dataset</th>
        <th class="tg-fymr">Keyboard Precision ↑</th>
        <th class="tg-fymr">Camera Precision ↑</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="tg-0pky">Gamefactory dataset</td>
        <td class="tg-0pky">0.5209</td>
        <td class="tg-0pky">0.2654</td>
      </tr>
      <tr>
        <td class="tg-0pky">Our curated dataset</td>
        <td class="tg-0pky">0.6009</td>
        <td class="tg-0pky">0.3907</td>
      </tr>
    </tbody>
  </table>
    <p>
    <strong>Table 2. Action Controllability Comparison of T2V in Different Datasets.</strong>
  </p>
</div>

<!-- markdownlint-restore -->

To evaluate the quality of our newly curated dataset, we trained the same T2V model on both Gamefactory dataset and our dataset. As shown in Table 2, the model trained on our dataset achieved superior precision on both keyboard and mouse controllability, indicating that the collected data provides more representative information for the task.

## Visual Performance

In this section, we present visual results of Micro-World, including T2W results in both in-domain and open-domain settings, as well as I2W open-domain action-controlled scenarios, shown in Figures 3, 4, and 5, respectively. All videos are generated on an AMD Instinct™ MI325X GPU using AMD ROCm™ software version 7.0.0.

We observe that fully decoupling the action module from game-specific styles in large-scale models remains challenging. As a result, we apply both the LoRA weights and the action module during inference for the I2W open-domain results.

Keyboard and mouse actions are annotated in the visualization. Keyboard controls including W (forward), A (left), S (backward), D (right), Ctrl (sprint), Shift (sneak), Space (jump).

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/01ecff57-5fc8-40c0-b7c1-1c72525b598c" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; overflow:hidden; font-size: 14px;">
        W
        </div>
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/0156af1f-5fe2-4276-9cec-ba97b2476018" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        S
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d27268e5-9fbc-49f7-b3ca-882fb58f21b6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        A
        </div>
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/aff52ef1-0c9c-4a03-961f-6aa5361b636d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        D
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b5d37d89-5cf0-40a5-8504-61f68e944fb9" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Ctrl
        </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1b0d50c8-a037-4671-a146-b77672260322" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Shift
        </div>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
    <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b13a14d7-5882-42dd-872b-8f61b9ab7060" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Multiple control
        </div>
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1218bbda-7993-4075-881b-2e16002acda8" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse down and up
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/31471313-d94b-4936-b23b-12e7f89fda87" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse right and left
        </div>
      </td>
  </tr>
</table>
<p align="center">
  Figure 3. T2W in-domain results.
</p>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/25ec4ba8-4f65-4b26-8966-13437647f240" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/a92149a8-6c4d-4b9a-8ada-81b47b4c81e7" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>  
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/67b35842-04fd-4a0f-9a5c-6914d9f77e66" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Running along a cliffside path in a tropical island in first person perspective, with turquoise waters crashing against the rocks far below, the salty scent of the ocean carried by the breeze, and the sound of distant waves blending with the calls of seagulls as the path twists and turns along the jagged cliffs.
              </div>
            </details>
          </div>  
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d4a46b8b-022d-4fca-964f-c1d477111f4e" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A young bear stands next to a large tree in a grassy meadow, its dark fur catching the soft daylight. The bear seems poised, observing its surroundings in a tranquil landscape, with rolling hills and sparse trees dotting the background under a pale blue sky.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b6f77a1a-58ce-43db-b6c5-efe09b7a9142" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A giant panda rests peacefully under a blooming cherry blossom tree, its black and white fur contrasting beautifully with the delicate pink petals. The ground is lightly sprinkled with fallen blossoms, and the tranquil setting is framed by the soft hues of the blossoms and the grassy field surrounding the tree.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/c9225344-8b0b-4249-ab77-c8e5c4dddacc" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Exploring an ancient jungle ruin in first person perspective surrounded by towering stone statues covered in moss and vines.
              </div>
            </details>
          </div>
     </td>
  </tr>
</table>
<p align="center">
  Figure 4. T2W open-domain results.
</p>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f135d5c9-0379-4ace-bf22-1671cef261af" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly. camera motion directly aligned with user actions, immersive urban night scene.
              </div>
            </details>
          </div>
      </td>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/2088d2da-95a6-4908-b7a2-f60458281b5e" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective standing in front of an ornate traditional Chinese temple. The symmetrical facade features red lanterns, intricate carvings, and a curved tiled roof decorated with dragons. Bright daytime lighting, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
       <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/9e1185cc-5480-4059-8643-7b6e08fff0c1" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective of standing in a rocky desert valley, looking at a camel a few meters ahead. The camel stands calmly on uneven stones, its long legs and single hump clearly visible. Bright midday sunlight, dry air, muted earth tones, distant barren mountains. Natural handheld camera feeling, camera motion controlled by user actions, smooth movement, cinematic realism.
              </div>
            </details>
          </div>
     </td>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/c75d7344-7016-494e-be00-103d28e43738" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking through a narrow urban alley, old red brick industrial buildings on both sides, cobblestone street stretching forward with strong depth, metal walkways connecting buildings above, overcast daylight, soft diffused lighting, cool and muted color tones, quiet and empty environment, no people, camera motion controlled by user actions, smooth movement, stable horizon, realistic scale and geometry, high realism, cinematic urban scene.
              </div>
            </details>
          </div>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f6da97af-0d3a-4b6a-b80f-5ae3c03ccbf6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective coastal exploration scene, walking along a cliffside stone path with wooden railings, green bushes lining the walkway, ocean to the left with gentle waves, distant islands visible under a clear sky, realistic head-mounted camera view, smooth forward motion, stable horizon, natural human eye level, high realism, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>
     </td>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/b76a8aca-d1da-47ba-88e9-3da36f64429d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective inside a cozy living room, walking around a warm fireplace, soft carpet underfoot, furniture arranged neatly, bookshelves, plants, and warm table lamps on both sides, warm indoor lighting, calm and quiet atmosphere, natural head-level camera movement, camera motion driven by user actions, realistic scale and depth, high realism, cinematic lighting, no people, no distortion.
              </div>
            </details>
          </div>
     </td>
  </tr>
</table>
<p align="center">
  Figure 5. I2W open-domain results.
</p>

## Summary

We present **Micro-World**, a series of action-controlled interactive models designed to showcase the capabilities of AMD Instinct™ GPUs for both training and inference, while establishing a foundation for future research. Through a carefully designed model architecture and training pipeline, Micro-World achieves effective action control in real-world scenarios. Our models demonstrate strong performance in both video quality and action coherence. To support reproducibility and foster further research, we release the model weights, complete training and inference code, and a curated dataset to the open-source community.

In future work, we plan to extend Micro-World toward generating longer action-controlled videos, incorporate more efficient operators, and reduce computational overhead to enable real-time, streaming world models. Stay tuned!

## Resources

**Huggingface Model Cards:**

- [amd/Micro-World-T2W · Hugging
    Face](https://huggingface.co/amd/Micro-World-T2W)

- [amd/Micro-World-I2W · Hugging
    Face](https://huggingface.co/amd/Micro-World-I2W)

**Huggingface Dataset Cards:**

- [amd/Micro-World-MC-Dataset](https://huggingface.co/datasets/amd/Micro-World-MC-Dataset) (Coming soon)

**Code:**

- [AMD-AGI/Micro-World](https://github.com/AMD-AGI/Micro-World)

Related Work on Diffusion Models by the AMD Team:

- [Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms](https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html)
- [AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [Nitro-E: A 304M Diffusion Transformer Model for High Quality Image Generation](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html)
- [Nitro-T: Training a Text-to-Image Diffusion Model from Scratch in 1 Day](https://rocm.blogs.amd.com/artificial-intelligence/nitro-t-diffusion/README.html)

Related Technical Blogs on World Models:

- [Exploring Gameplay Video Generation with Hunyuan-GameCraft](https://rocm.blogs.amd.com/artificial-intelligence/hunyuan-gamecraft/README.html)
  
## Reference

1. Valevski, Dani, et al. "Diffusion models are real-time game engines." arXiv preprint arXiv:2408.14837 (2024).
2. Yu, Jiwen, et al. "GameFactory: Creating new games with generative interactive videos." arXiv preprint arXiv:2501.08325 (2025).
3. Zhang, Yifan, et al. "Matrix-Game: Interactive World Foundation Model." arXiv preprint arXiv:2506.18701 (2025).
4. Li, Jiaqi, et al. "Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition." arXiv preprint arXiv:2506.17201 (2025).
5. Wan, Team, et al. "Wan: Open and advanced large-scale video generative models." arXiv preprint arXiv:2503.20314 (2025).
6. Yang, Mingyu, et al. "Playable game generation." arXiv preprint arXiv:2412.00887 (2024).
7. Decart, et al. "Oasis: A Universe in a Transformer." https://oasis-model.github.io/ (2024)
8. Hu, Edward J., et al. "LoRA: Low-rank adaptation of large language models." ICLR 1.2 (2022): 3.
9. Fan, Linxi, et al. "Minedojo: Building open-ended embodied agents with internet-scale knowledge." Advances in Neural Information Processing Systems 35 (2022): 18343-18362.
10. Yao, Yuan, et al. "MiniCPM-V: A GPT-4V level MLLM on Your Phone." arXiv preprint arXiv:2408.01800 (2024).
11. Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." Proceedings of the IEEE/CVF international conference on computer vision. 2023.
12. Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
