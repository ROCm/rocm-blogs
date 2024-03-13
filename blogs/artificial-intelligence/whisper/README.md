---
blogpost: true
date: 13 Mar 2024
author: Clint Greene
tags: AI/ML, ASR, Whisper, Speech to Text
category: Applications & models
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Speech to Text on AMD with Whisper">
  <meta name="keywords" content="Whisper, ASR, Automatic Speech Recognition, AMD, GPU, MI300, MI250">
</head>

# Speech-to-Text on an AMD GPU with Whisper

## Introduction

[Whisper](https://openai.com/research/whisper) is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. Whisper exhibits robustness to accents, background noise and technical language. Moreover, it enables transcription in multiple languages, as well as translation from those languages into English.

The Whisper architecture is based upon the original sequence-to-sequence encoder-decoder Transformer model. Input audio is split into 30-second chunks, converted into a log-Mel spectrogram, and then passed into an encoder. A decoder is trained to predict the corresponding text caption, intermixed with special tokens that direct the single model to perform tasks such as language identification, phrase-level timestamps, multilingual speech transcription, and to-English speech translation.

In this blog, we will show you how to convert speech to text using Whisper with both Hugging Face and OpenAI's official Whisper release on an AMD GPU.

![Architecture](./images/whisper.png)

**Tested with GPU Hardware:** MI210 / MI250\
**Prerequisites:** Ensure [ROCm 5.7+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) and [PyTorch 2.2.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html) are installed.

We recommend users to install the latest release of PyTorch and TorchAudio as we are continually releasing optimized solutions and new features.

## Getting Started

Let's first install the libraries we'll need

```bash
pip install datasets ipywidgets transformers numba openai-whisper -q
```

```bash
sudo apt update && sudo apt install ffmpeg
```

Now that the necessary libraries are installed, let's download a sample audio file that we will use later for transcribing. This is the opening line to Lincoln's famous Gettysburg address.

```bash
wget https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav
```

We are now ready to convert speech to text with Hugging Face Transformers and OpenAI's Whisper codebase.

### HuggingFace Transformers

Let's import the libraries we will need

```python
import torch
from transformers import pipeline
from datasets import load_dataset
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
transcription = pipe("gettysburg.wav")['text']
print(transcription)
```

Output:

```text
Four, score, and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.
```

This is the correct transcription of the opening lines of the Gettysburg address.

## OpenAI's Whisper

Similarly, we can perform transcription using OpenAI's official Whisper release. First, we download the medium English model weights. Then, to perform transcription, we again pass the path to the audio file that we would like to transcribe.

```python
import whisper

model = whisper.load_model("medium.en")
transcription = model.transcribe("gettysburg.wav")['text']
print(transcription)
```

Output:

```text
Four, score, and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.
```

## Conclusions

We have demonstrated how to transcribe a single audio file using the Whisper model from the Hugging Face Transformers library as well as OpenAI's official code release. If you’re planning to transcribe batches of files, we recommend using the implementation from Hugging Face since it supports batch decoding. For additional examples on how to transcribe batches of files or how to use a Hugging Face Dataset see the official [pipeline tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial).
