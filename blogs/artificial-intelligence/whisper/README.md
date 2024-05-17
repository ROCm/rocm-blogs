---
blogpost: true
date: 16 Apr 2024
author: Clint Greene
tags: AI/ML, Whisper, Speech to Text
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Speech to Text on AMD with Whisper"
    "keywords": "Whisper, ASR, Automatic Speech Recognition, AMD, GPU, MI300, MI250"
    "property=og:locale": "en_US"
---

# Speech-to-Text on an AMD GPU with Whisper

## Introduction

[Whisper](https://openai.com/research/whisper) is an advanced automatic speech recognition (ASR) system, developed by OpenAI. It employs a straightforward encoder-decoder Transformer architecture where incoming audio is divided into 30-second segments and subsequently fed into the encoder. The decoder can be prompted with special tokens to guide the model to perform tasks such as language identification, transcription, and translation.

In this blog, we will show you how to convert speech to text using Whisper with both Hugging Face and OpenAI's official Whisper release on an AMD GPU.

**Tested with GPU Hardware:** MI210 / MI250\
**Prerequisites:** Ensure [ROCm 5.7+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) and [PyTorch 2.2.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html) are installed.

We recommend users to install the latest release of PyTorch and TorchAudio as we are continually releasing optimized solutions and new features.

## Getting Started

First, let us install the necessary libraries.

```bash
pip install datasets ipywidgets transformers numba openai-whisper -q
```

```bash
sudo apt update && sudo apt install ffmpeg
```

Now that the necessary libraries are installed, let's download a sample audio file of the Preamble of the United States Constitution that we will use later for transcribing.

```bash
wget https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav
```

We are now ready to convert speech to text with Hugging Face Transformers and OpenAI's Whisper codebase.

### Hugging Face Transformers

Let us import the necessary libraries.

```python
import torch
from transformers import pipeline
```

Then we setup the device and pipeline for transcription. Here, we'll download and use the Whisper medium weights released by OpenAI for English transcription in the pipeline.

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium.en",
  chunk_length_s=30,
  device=device,
)
```

To convert speech to text, we pass the path to the audio file to the pipeline

```python
transcription = pipe("preamble.wav")['text']
print(transcription)
```

Output:

```text
We, the people of the United States, in order to form a more perfect union, establish justice, ensure domestic tranquility, provide for the common defense, promote the general welfare, and secure the blessings of liberty to ourselves and our posterity, to ordain and establish this Constitution for the United States of America.
```

This is the correct transcription of the Preamble of the United States Constitution.

## OpenAI's Whisper

Similarly, we can perform transcription using OpenAI's official Whisper release. First, we download the medium English model weights. Then, to perform transcription, we again pass the path to the audio file that we would like to transcribe.

```python
import whisper

model = whisper.load_model("medium.en")
transcription = model.transcribe("preamble.wav")['text']
print(transcription)
```

Output:

```text
We, the people of the United States, in order to form a more perfect union, establish justice, ensure domestic tranquility, provide for the common defense, promote the general welfare, and secure the blessings of liberty to ourselves and our posterity, to ordain and establish this Constitution for the United States of America.
```

## Conclusions

We have demonstrated how to transcribe a single audio file using the Whisper model from the Hugging Face Transformers library as well as OpenAI's official code release. If you’re planning to transcribe batches of files, we recommend using the implementation from Hugging Face since it supports batch decoding. For additional examples on how to transcribe batches of files or how to use a Hugging Face Dataset see the official [pipeline tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial).

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
