---
blogpost: true
date: 8 March 2024
author: Phillip Dang
tags: PyTorch, AI/ML, Tuning
category: Applications & models
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Music Generation With MusicGen on an AMD GPU">
  <meta name="author" content="Phillip Dang">
  <meta name="keywords" content="PyTorch, MusicGen, train models">
</head>

# Music Generation With MusicGen on an AMD GPU

MusicGen is an autoregressive, transformer-based model that predicts the next segment of a piece of
music based on previous segments. This is a similar approach to language models predicting the next
token.

MusicGen is able to generate music using the following as input:

* No input sources (e.g., unconditional generation)
* A text description (e.g., text conditional generation)
* An input music sequence (e.g., melody conditional generation)

For a deeper dive into the inner workings of MusicGen, refer to
[Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284).

In this blog, we demonstrate how to seamlessly run inference on MusicGen using AMD GPUs and
ROCm. We use [this model from Hugging Face](https://huggingface.co/spaces/facebook/MusicGen)
with the three preceding inputs.

## Prerequisites

To run MusicGen locally, you need at least one GPU. To follow along with this blog, you must have the
following software:

* [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Linux OS

To check your hardware and ensure that your system recognizes your GPU, run:

``` bash
rocm-smi --showproductname
```

Your output should look like this:

```bash
================= ROCm System Management Interface ================
========================= Product Info ============================
GPU[0] : Card series: Instinct MI210
GPU[0] : Card model: 0x0c34
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] : Card SKU: D67301
===================================================================
===================== End of ROCm SMI Log =========================
```

To make sure PyTorch recognizes your GPU, run:

```python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

Your output should look similar to this:

```python
number of GPUs: 1
['AMD Radeon Graphics']
```

Once you've confirmed that your system recognizes your device(s), you're ready to install the required
libraries and generate some music.

In this blog, we use the `facebook/musicgen-small` variant.

### Libraries

You can use MusicGen with Hugging Face's transformer. To install the required libraries, run the following commands:

```python
! pip install -q transformers
```

## MusicGen with Hugging Face

MusicGen is available in the Hugging Face Transformers library from version 4.31.0 onwards. Let's take a look at how to use it. We will be following [Hugging Face's demo](https://huggingface.co/docs/transformers/model_doc/musicgen) in this section. We will generate music in the 3 different modes explained in the introduction.

### Unconditional generation

Let's start by generating music without any input.

```python
from transformers import MusicgenForConditionalGeneration

# initialize model and model's input
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

# generate audio
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

You can either listen to the audio directly in your notebook or save the audio as a WAV file using
**scipy**.

* To listen in your notebook, run:

  ```python
  from IPython.display import Audio

  sampling_rate = model.config.audio_encoder.sampling_rate

  # listen to our audio sample
  Audio(audio_values[0].cpu(), rate=sampling_rate)
  ```

* To save the audio, run

  ```python
  import scipy

  sampling_rate = model.config.audio_encoder.sampling_rate
  scipy.io.wavfile.write("audio/unconditional.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
  ```

### Text-conditional generation

Next, let's generate music conditioned on our text input. This process has three steps:

1. Text descriptions are passed through a text encoder model to obtain a sequence of hidden-state
  representations.
2. MusicGen is trained to predict audio tokens, or audio codes, conditioned on these hidden-states.
3. Audio tokens are decoded using an audio compression model, such as EnCodec, to recover the
  audio waveform.

To see this in action, run:

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Initialize model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Set device to GPU
device = 'cuda'
model = model.to(device)

# Text description for the model
input_text = ["epic movie theme", "sad jazz"]

# Create input
inputs = processor(
    text=input_text,
    padding=True,
    return_tensors="pt",
).to(device)

# Generate audio
audio_values_from_text = model.generate(**inputs, max_new_tokens=512)

print(audio_values_from_text.shape)
```

```python
torch.Size([2, 1, 325760])
```

Note that the audio outputs are a three-dimensional Torch tensor of shape `batch_size`,
`num_channels`, and `sequence_length`. As with unconditional generation, you can listen to your
generated audio via the Audio library:

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate

# Listen to your first audio sample from input text "epic music theme"
Audio(audio_values_from_text[0].cpu(), rate=sampling_rate)

# Listen to your second audio sample from input text "sad jazz"
Audio(audio_values_from_text[1].cpu(), rate=sampling_rate)
```

We saved our versions of these two WAV files as `audio/conditional1.wav` and
`audio/conditional2.wav` in [this GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/MusicGen/audio), so you can listen to them without having to run the code.

### Audio-prompted generation

You can also generate music by providing a melody and a text description to guide the generative
process. Let's take the first half of the sample we previously generated from our text description
"sad jazz" and use it as our audio prompt:

```python
# take the first half of the generated audio
sample = audio_values_from_text[1][0].cpu().numpy()
sample = sample[: len(sample) // 2]

# use it as input
inputs = processor(
    audio=sample,
    sampling_rate=sampling_rate,
    text=["sad jazz"],
    padding=True,
    return_tensors="pt",
).to(device)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

You can listen to the audio using:

```python
Audio(audio_values[0].cpu(), rate=sampling_rate)
```

We saved this under `audio/audio_prompted.wav` in [this GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/MusicGen/audio).

While we only used the small model in this blog, we encourage you to explore the medium and
large models. We also to experiment with fine-tuning the model using your own custom audio
dataset.
