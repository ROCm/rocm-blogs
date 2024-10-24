---
blogpost: true
blog_title: 'CTranslate2: Efficient Inference with Transformer Models on AMD GPUs'
thumbnail: './images/nlp.jpg'
date: 24 October 2024
author: Michael Zhang
tags: AI/ML, GenAI, LLM, PyTorch 
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Optimizing Transformer models with CTranslate2 for efficient inference on AMD GPUs"
    "author": "Michael Zhang"
    "keywords": "GenAI, CTranslate2, Transformer, NLP, translation, summarization, inference, quantization, optimization, AMD, GPU, MI300, MI250, ROCm, OpenNMT, PyTorch, LLM, gpt, whisper, llama"
    "property=og:locale": "en_US"
---

# CTranslate2: Efficient Inference with Transformer Models on AMD GPUs

Transformer models have revolutionized natural language processing (NLP) by delivering high-performance results in tasks like machine translation, text summarization, text generation, and speech recognition. However, deploying these models in production can be challenging due to their high computational and memory requirements. [CTranslate2](https://github.com/OpenNMT/CTranslate2) addresses these challenges by providing a custom runtime that implements various optimization techniques to accelerate Transformer models during inference.

In this blog, you will learn how to optimize and accelerate the inference of Transformer models on AMD Hardware using CTranslate2, a powerful C++ and Python library designed for efficient inference on CPUs and GPUs. We'll explore its key features, installation process, and how to integrate it into your NLP applications to achieve significant performance gains and reduced memory usage.

## What are OpenNMT and CTranslate2?

[OpenNMT](https://opennmt.net/) is an open-source toolkit designed for machine translation, which is the process of automatically converting text from one language to another. It also supports related tasks like text summarization, which involves understanding and generating sequences of words, often referred to as "neural sequence learning". It provides a comprehensive toolkit for training and deploying translation models. CTranslate2 is an optimized inference engine that supports models trained in OpenNMT, offering substantial speed-ups and efficiency improvements when deploying these models in production. It is part of the OpenNMT ecosystem and can work as a solution tailored for high-performance Transformer model inference.

The CTranslate2 library includes powerful features, such as embedding models in C++ applications and reducing model size on disk and memory. Being framework agnostic, it can handle models exported from frameworks like PyTorch or TensorFlow while maintaining lightweight execution.

Models trained with CTranslate2 are converted into a special "CTranslate2 format", which makes them optimized for efficient inference. This format is designed to speed up the use of models while reducing memory usage, allowing them to be more suitable for production environments.

### Key Features of CTranslate2

CTranslate2 provides a wide range of [features](https://github.com/OpenNMT/CTranslate2?tab=readme-ov-file#key-features) designed to optimize Transformer model inference, including:

- **Fast and Efficient Execution**: Optimized for both CPU and GPU, delivering fast and efficient Transformer model inference

- **Quantization and Reduced Precision**: Supports quantization to INT8, INT16, and FP16, reducing memory usage while maintaining accuracy

- **Parallel and Asynchronous Execution**: Allows batch and asynchronous inference to maximize throughput

- **Dynamic Memory Usage**: Efficient use of memory for large-scale deployment

- **Lightweight on Disk**: Reduced model size, to help with scalability

- **Simple Integration**: Easy to embed in C++ or Python applications

- **Tensor Parallelism**: Supports distributed inference for large models

### Supported Models and Frameworks

CTranslate2 currently supports the following models:

- **Encoder-Decoder Models**: Transformer base/big, BART, mBART, Pegasus, T5, Whisper

- **Decoder-Only Models**: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, CodeGen, Falcon

- **Encoder-Only Models**: BERT, DistilBERT, XLM-RoBERTa

Compatible models should first be converted into an optimized model format. The library includes converters for multiple frameworks:

- [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
- [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
- [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

## Installation and Setup

To run CTranslate2 on AMD GPUs, you will need:

- [AMD GPU Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

- [ROCm 6.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)

- [PyTorch 2.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)

Here are the steps to get started:

1. Clone the ROCm CTranslate2 Repo:

    ```bash
    git clone https://github.com/ROCm/CTranslate2.git
    ```

1. Build the Docker Image for ROCm AMD GPUs:

    ```bash
    cd CTranslate2/docker_rocm
    docker build -t rocm_ct2_v3.23.0 -f Dockerfile.rocm .
    ```

1. Launch the Docker container:

    ```bash
    docker run -it --ipc=host --cap-add=SYS_PTRACE --network=host --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --privileged -w /workspace rocm_ct2_v3.23.0
    ```

1. After setting up the Docker container, clone this blog's repository:

    ```bash
    git clone https://github.com/ROCm/rocm-blogs.git
    cd rocm-blogs/blogs/artificial-intelligence/ctranslate2
    ```

### Quick Start with Text Translation

To get started with CTranslate2 in Python, follow these steps. Convert a pretrained model to the CTranslate2 format, then run your first translation using the converted model.

1. First, install the required Python packages inside the Docker container:

    ```bash
    pip install OpenNMT-py==2.* sentencepiece
    ```

1. Then, download and extract the [OpenNMT-py models](https://opennmt.net/Models-py/), for example, the English-German Transformer model trained with OpenNMT-py:

    ```bash
    wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
    tar xf transformer-ende-wmt-pyOnmt.tar.gz
    ls  # Check that `averaged-10-epoch.pt` is present
    ```

1. Convert the model to the CTranslate2 format:

    ```bash
    ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2
    ls  # Check that `ende_ctranslate2` folder is present
    ```

    The expected output folder looks like this:

    ```bash
    # ende_ctranslate2/
    # ├── config.json
    # ├── model.bin
    # └── shared_vocabulary.json
    ```

To translate texts, use the Python API:

```bash
python3 src/translate.py
```

Here is the `translate.py` file:

```python
import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("ende_ctranslate2/", device="cuda")
sp = spm.SentencePieceProcessor("sentencepiece.model")

input_text = "Good Morning!"
input_tokens = sp.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
```

The program should print the following translation:

```bash
Guten Morgen!
```

If you see this output, you successfully converted and executed a translation model with CTranslate2!

## Quantization

Quantization is a technique that can reduce the model size and accelerate its execution with little to no degradation in accuracy. Currently, CTranslate2 supports quantization on AMD GPUs to the following datatypes:

- 8-bit integers (INT8)
- 16-bit integers (INT16)
- 16-bit floating points (FP16)

By applying quantization, you can significantly reduce the model's memory footprint and improve inference speed, which is especially beneficial when deploying models on resource-constrained environments or when aiming for higher throughput. For more information about quantization, see the [documentation](https://opennmt.net/CTranslate2/quantization.html).

### Convert Models to Quantized Datatypes

Enabling quantization during model conversion helps reduce the model size on disk and can improve inference speed. The converter provides the `--quantization` option that accepts the following values:

- `int8`
- `int16`
- `float16`
- `float32` (default)

For example, to convert and quantize the model to INT8, use this command:

```bash
ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --quantization int8 --output_dir ende_ctranslate2_int8
ls  # Check that `ende_ctranslate2_int8` folder is present
```

The expected output folder looks like this:

```bash
# ende_ctranslate2_int8/
# ├── config.json
# ├── model.bin
# └── shared_vocabulary.json
```

### Quantize on Model Loading

Quantization can also be enabled or changed when loading the model by setting the `compute_type` parameter:

- `default`: Keep the same quantization used during model conversion.
- `auto`: Use the fastest computation type supported on the system and device.
- `int8`, `int16`, `float16`, `float32`

For example, to load the model with INT8 computation, run this command:

```python
translator = ctranslate2.Translator("ende_ctranslate2_int8/", device="cuda", compute_type="int8")
```

The use of the translator is shown in the [translate.py code](#quickstart-with-text-translation).

### Performance Comparison: INT8 Quantization Effect

To illustrate the effect of INT8 quantization on performance, let's compare the translation latency and model size between the default (float32) model and the INT8 quantized model on an AMD GPU.

#### 1. Model Conversion

Convert the model with INT8 quantization:

```bash
ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --quantization int8 --output_dir ende_ctranslate2_int8
```

#### 2. Translation Script with Latency Measurement

Run the code to measure and compare latency:

```bash
python3 src/translate_compare.py
```

Here is the source code for [`translate_compare.py`](./src/translate_compare.py):

```python
import ctranslate2
import sentencepiece as spm
import time

# Load the SentencePiece model
sp = spm.SentencePieceProcessor(model_file="sentencepiece.model")

# Input text to translate
input_text = "Hello world!"
input_tokens = sp.encode(input_text, out_type=str)

# Function to perform translation and measure latency and tokens per second
def translate_and_time(translator):
    start_time = time.time()
    results = translator.translate_batch([input_tokens])
    end_time = time.time()
    latency = end_time - start_time

    # Decode the translated tokens
    output_tokens = results[0].hypotheses[0]
    output_text = sp.decode(output_tokens)

    # Calculate tokens per second
    num_output_tokens = len(output_tokens)
    tokens_per_second = num_output_tokens / latency

    return output_text, latency, tokens_per_second

# Load the default (float32) model
translator_float32 = ctranslate2.Translator(
    "ende_ctranslate2/", device="cuda", compute_type="float32"
)
output_text_float32, latency_float32, tps_float32 = translate_and_time(translator_float32)

# Load the int8 quantized model
translator_int8 = ctranslate2.Translator(
    "ende_ctranslate2_int8/", device="cuda", compute_type="int8"
)
output_text_int8, latency_int8, tps_int8 = translate_and_time(translator_int8)

# Print the results
print("Default (float32) model translation:")
print(f"Output: {output_text_float32}")
print(f"Latency: {latency_float32:.4f} seconds")
print(f"Tokens per second: {tps_float32:.2f}\n")

print("Int8 quantized model translation:")
print(f"Output: {output_text_int8}")
print(f"Latency: {latency_int8:.4f} seconds")
print(f"Tokens per second: {tps_int8:.2f}\n")

# Calculate the speedup in tokens per second
speedup_tps = tps_int8 / tps_float32
print(f"Speedup in tokens per second with int8 quantization: {speedup_tps:.2f}x faster")
```

#### 3. Sample Output and Analysis

```bash
Default (float32) model translation:
Output: Hallo Welt!
Latency: 0.1510 seconds
Tokens per second: 19.86

Int8 quantized model translation:
Output: Hallo Welt!
Latency: 0.0428 seconds
Tokens per second: 70.14

Speedup in tokens per second with int8 quantization: 3.53x faster
```

**Analysis:**

Both models produced similar translations in this particular example, suggesting minimal loss in translation quality due to quantization. However, further testing is needed across a larger set of samples to confirm general trends. The INT8 model was approximately 3.53x faster compared to the float32 model, demonstrating significant performance gains.

## Text Generation

Text generation is a fundamental task in NLP, where the goal is to produce coherent and contextually relevant text in response to a prompt. CTranslate2 provides support for efficient inference with text generation models, allowing you to deploy them in production environments with optimized performance. Let's use a GPT-2 model for this example.

### 1. Convert GPT-2 Model

First, convert the pretrained GPT-2 model from Hugging Face Transformers into the CTranslate2 format using the ct2-transformers-converter script:

```bash
ct2-transformers-converter --model gpt2 --output_dir gpt2_ct2
```

### 2. Text Generation Script

To properly understand the differences between unconditional and conditional generation, we should first explain these concepts before presenting the code:

#### Unconditional Generation

- **What It Is:** The model generates text without any prior context or input prompt.
- **How It Works:** The generation starts from a special token (for example, `<|endoftext|>` for GPT-2) and relies solely on patterns learned during training.
- **Use Cases:** Creative writing, generating random text, exploring the model's inherent knowledge.

#### Conditional Generation

- **What It Is:** The model generates text that continues from a given input or prompt.
- **How It Works:** The generation starts from user-provided tokens, and the model predicts subsequent tokens based on the context of the prompt.
- **Use Cases:** Text completion, language modeling, generating responses in chatbots, auto-generating code or articles based on a starting sentence.

Now, you can run this script to generate text:

```bash
python3 src/gpt2.py
```

Below is the [`gpt2.py`](./src/gpt2.py) script for performing both unconditional and conditional text generation using the converted GPT-2 model:

```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("gpt2_ct2", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# Unconditional generation.
start_tokens = [tokenizer.bos_token]
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))

# Conditional generation.
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("It is"))
results = generator.generate_batch([start_tokens], max_length=30, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))
```

Here are the parameters for `generate_batch`:

- **`max_length`**: The maximum number of tokens to generate. In this example, it's set to 30.
- **`sampling_topk`**: Controls the randomness of the generated text by sampling the next token from the top `k` most probable tokens. A smaller `k` value makes the output more deterministic.

### 3. Sample Output

```bash
This is a very nice and simple tutorial on how the game should work, but the first couple of steps are pretty straightforward: first, open up
It is true that in my opinion, if there were a large number of people in Europe who were involved in the development, it would be very difficult to
```

## Speech Recognition

[Whisper](https://github.com/openai/whisper) is a multilingual speech recognition model developed by OpenAI. It is designed to transcribe audio files into text across multiple languages. CTranslate2 provides support for efficient inference with Whisper models, enabling faster transcription and reduced resource usage.

### Important Considerations

- **Transformers Version**: Converting Whisper models requires `transformers` library version **4.23.0** or higher.
- **Model Size**: The example below uses the smallest model, `whisper-tiny`, which has 39 million parameters. For better transcription accuracy, consider using larger models like `whisper-base`, `whisper-small`, `whisper-medium`, or `whisper-large`.

### Converting the Whisper Model

First, convert the pretrained Whisper model from Hugging Face Transformers into the CTranslate2 format using the `ct2-transformers-converter` script:

```bash
ct2-transformers-converter --model openai/whisper-tiny --output_dir whisper-tiny-ct2
```

This command downloads the `whisper-tiny` model from Hugging Face, converts it into a format optimized for inference with CTranslate2, and saves it in the `whisper-tiny-ct2` directory.

### Transcribing Audio with CTranslate2

Here is an example showing how to use the converted Whisper model to transcribe an audio file using CTranslate2. The sample audio file (`sample2.flac`) and the Python code is under the `src` folder of this blog's GitHub repository:

```bash
python3 src/speech_recognition.py
```

Here is the [`speech_recognition.py`](./src/speech_recognition.py) code for transcribing the audio:

```python
import ctranslate2
import librosa
import transformers

# Load and resample the audio file.
audio, _ = librosa.load("src/sample2.flac", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on GPU.
model = ctranslate2.models.Whisper("whisper-tiny-ct2", device="cuda")

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))

# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

# Run generation for the 30-second window.
results = model.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
print(transcription)
```

The program generates the following sample output:

```bash
Detected language <|en|> with probability 0.981871
 Before he had time to answer, a much encumbered Vera burst into the wrong with the question, I say, can I leave these here? These were a small black pig and a lusty specimen of black red game cock.
```

Here is an explanation of how the program works:

- **Audio Loading**: The script uses `librosa` to load and resample the audio file `audio.wav` to a sampling rate of 16 kHz and convert it to mono.
- **Feature Extraction**: It computes the input features for the first 30 seconds of the audio using `WhisperProcessor` from the `transformers` library.
- **Model Loading**: It loads the Whisper model using CTranslate2's `Whisper` class and places it on the GPU for inference (`device="cuda"`).
- **Language Detection**: The `detect_language` method is used to identify the language spoken in the audio segment. It returns the language code and the probability.
- **Prompt Preparation**: This script creates a prompt to describe the transcription task. The prompt includes special tokens that specify the start of the transcript, the language, and transcription mode without timestamps.
- **Transcription Generation**: The `generate` method transcribes the audio features using the prepared prompt.

### Handling Longer Audio Files

The previous example only processes the first 30 seconds of the audio file. To transcribe longer audio recordings:

- **Segment the Audio**: Divide the audio file into 30-second (or shorter) segments.
- **Process Sequentially**: Loop over each segment, performing feature extraction and transcription.
- **Aggregate Results**: Combine the transcriptions from all segments to form the complete transcription.

## Conclusion

CTranslate2 offers a robust solution for deploying Transformer models efficiently on both CPUs and GPUs in production environments. By leveraging features like quantization and seamless integration with C++ and Python, CTranslate2 enables faster and more efficient performance on AMD GPUs. This makes it an ideal choice for tasks such as text translation, text generation, and speech recognition.

## Additional Resources

- [CTranslate2 GitHub Repository](https://github.com/OpenNMT/CTranslate2)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [CTranslate2 ROCm GitHub Repository](https://github.com/ROCm/CTranslate2)
- [OpenNMT Project](https://opennmt.net/)
- [AMD ROCm Platform](https://rocm.docs.amd.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
