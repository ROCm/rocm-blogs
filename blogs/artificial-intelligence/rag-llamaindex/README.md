---
blogpost: true
date: 04 Apr 2024
author: Clint Greene
tags: LLM, AI/ML, RAG
category: Applications & models
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Retrieval Augmented Generation (RAG) using LlamaIndex">
  <meta name="keywords" content="RAG, Retrieval Augmented Generation, Prompt Engineering, LLMs, Large Language Models, AMD, GPU, MI300, MI250">
</head>

# Retrieval Augmented Generation (RAG) using LlamaIndex

## Prerequisites

To run this blog, you will need the following:

- **Linux**: see [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
- **ROCm**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
- **AMD GPU**: see the [list of compatible GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

## Introduction

Large Language Models (LLMs), such as ChatGPT, are powerful tools capable of performing many complex
writing tasks. However, they do have limitations, notably:

- Lack of access to up-to-date information: LLMs are "frozen in time" as their training data is inherently outdated, which means they cannot access the latest news or information.

- Limited applicability for domain-specific tasks: LLMs are not specifically trained for domain-specific tasks using domain-specific data, which can result in less relevant or incorrect responses for specialized use cases.

To address these limitations, there are two primary approaches to introduce up-to-date and domain-specific data:

- Fine-tuning: This involves providing the LLM with up-to-date, domain-specific prompt-and-completion text pairs. However, this approach can be costly, particularly if the data used for fine-tuning changes frequently, requiring frequent updates.

- Contextual prompting: This involves inserting up-to-date data as context into the prompt, which the LLM can then use as additional information when generating a response. However, this approach has limitations, as not all up-to-date, domain-specific documents may fit into the context of the prompt.

To overcome these obstacles, Retrieval Augmented Generation (RAG) can be used. RAG is a technique that enhances the accuracy and reliability of an LLM by exposing it to up-to-date, relevant information. It works by automatically splitting external documents into chunks of a specified size, retrieving the most relevant chunks based on the query, and augmenting the input prompts to use these chunks as context to answer the user's query. This approach allows for the creation of domain-specific applications without the need for fine-tuning or manual information insertion into contextual prompts.

A popular framework used by the AI community for RAG is LlamaIndex. It's a framework for building LLM applications that focus on ingesting, structuring, and accessing private or domain-specific data. Its tools facilitate the integration of custom out-of-distribution data into LLMs.

## Getting started

To get started, install the transformers, accelerate, and `llama-index` that you'll need for RAG:

```python
!pip install llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface llama-index-readers-web transformers accelerate -q
```

Then, import the `LlamaIndex` libraries:

```python
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
```

Many open-source LLMs require a preamble before each prompt or a specific structuring of the
prompt, which you can encode using `system_prompt` or `messages_to_prompt` before generation.
Additionally, queries may need an additional wrapper, which you can specify with the
`query_wrapper_prompt`. All this information is typically available on the Hugging Face model card for
the model you're using. In this case, you'll use `zephyr-7b-alpha` for RAG, so you can pull the expected
prompt format [here](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha).

```python
def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"
```

`LlamaIndex` supports using LLMs from Hugging Face directly by passing the model name to the
`HuggingFaceLLM` class. You can specify model parameters, such as which device to use and how
much quantization using `model_kwargs`. You can specify parameters that control the LLM generation
strategy, like `top_k`, `top_p`, and `temperature`, in `generate_kwargs`. You can also specify parameters
that control the length of the output, such as `max_new_tokens`, directly in the class. To learn more
about these parameters and how they affect generation, take a look at Hugging Face's
[text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
documentation.

```python
llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"use_safetensors": False},
    # tokenizer_kwargs={},
    generate_kwargs={"do_sample":True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="cuda",
)
```

## Raw prompting

To demonstrate the shortcomings mentioned earlier, prompt your LLM and ask it how does Paul Graham recommend to work hard.

```python
question = "How does Paul Graham recommend to work hard? Can you list it as steps"
response = llm.complete(question)
print(response)
```

```text
Paul Graham's advice on working hard, as outlined in his essay "How to Be a Maker," can be summarized into several steps:

1. Work on something you care about. This will give you the motivation and energy to work harder and longer than you would if you were working on something that didn't matter to you.

2. Set specific goals. Instead of just working on your project, set specific milestones and deadlines for yourself. This will give you a clear sense of direction and help you to focus your efforts.

3. Eliminate distractions. Turn off your phone, close your email, and find a quiet place to work. Eliminating distractions will help you to stay focused and make progress.

4. Work in short, intense bursts. Rather than working for long periods of time, break your work into short, intense bursts. This will help you to maintain your focus and avoid burnout.

5. Take breaks. Taking breaks is important for maintaining your focus and avoiding burnout. Use your breaks to clear your mind, recharge your batteries, and come back to your work with fresh energy.

6. Work on your weaknesses.
```

At first glance the generated response looks accurate and reasonable. The LLM knows we're talking
about Paul Graham and working hard. The recommended steps for working hard even look reasonable. However, these are not Paul Graham's suggestions for working hard. LLMs are known to 'hallucinate' when there's a gap in their knowledge,
making false, yet plausible, statements.

## Prompt engineering

A simple way to overcome 'hallucination' of facts is to engineer the prompt to include external
contextual information. Let's copy the body of text from
[this essay](https://paulgraham.com/hwh.html) on How to Work Hard.

You can automatically copy the text using the BeautifulSoup library in Python:

```python
url = "https://paulgraham.com/hwh.html"

documents = BeautifulSoupWebReader().load_data([url])
```

Now, modify the original question and include the updated information when asking the question:

```text
context = documents[0].text
prompt = f"""Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context}

Question: {question}

Answer: """
```

Now, input this prompt into your LLM and note the response:

```python
response = llm.complete(prompt)
print(response)
```

```text
1. Learn the shape of real work: Understand the difference between fake work and real work, and be able to distinguish between them.

2. Find the limit of working hard: Learn how many hours a day to spend on work, and avoid pushing yourself to work too much.

3. Work toward the center: Aim for the most ambitious problems, even if they are harder.

4. Figure out what to work on: Determine which type of work you are suited for, based on your interests and natural abilities.

5. Continuously assess and adjust: Regularly evaluate both how hard you're working and how well you're doing, and be willing to switch fields if necessary.

6. Be honest with yourself: Consistently be clear-sighted and honest in your evaluations of your abilities, progress, and interests.

7. Accept failure: Be open to the possibility of failure and learn from it.

8. Stay interested: Find work that you find interesting and enjoyable, rather than simply for financial gain or external validation.

9. Balance work and rest: Give yourself time to get going, but also recognize when it's time to take a
```

By prompting the LLM to use the essay as context, you're constraining the LLM to generate content
using information within the prompt, producing an accurate response. Now, try generating a response
with RAG and compare it to the contextual prompting.

## Retrieval Augmented Generation

To build a RAG application, you first need to call `ServiceContext`, which establishes the language and
embedding models to use, as well as key parameters (such as `chunk_size` and `chunk_overlap`) that
determine how the documents are parsed.

```python
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-base-en-v1.5", chunk_size=256, chunk_overlap=32)
```

When performing RAG, documents are broken into smaller chunks. The `chunk_size` parameter
specifies how many tokens in length each of the chunks should be, while `chunk_overlap` specifies how
many tokens each chunk should overlap with its adjacent chunks.

Set the `llm` parameter using the `llm` variable you used in the preceding experiments. For the
embedding model, use `bge-base` (shown to be top-performing for retrieval tasks) to embed the
document chunks.

Next, build your vector index using `VectorStoreIndex`, which takes your documents and passes them
to the embedding model for chunking and embedding. Then call `query_engine` to prepare the index
for queries, specifying `similarity_top_k` to return the top eight most similar chunks of the document to
the input query.

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=8)
```

Your RAG application is now ready to be queried. Let's query with the original question:

```python
response = query_engine.query(question)
print(response)
```

```text
Paul Graham recommends the following steps to work hard:
1. Constantly aim toward the center of the problem you're working on, which is usually the most ambitious and difficult part.

2. Measure both how hard you're working and how well you're doing. Don't solely rely on pushing yourself to work, as there may be times where you need to focus on easier, peripheral work.

3. When working on a specific problem, aim as close to the center as you can without stalling.

4. When working on a larger scale, make big, lifetime-scale adjustments about which type of work to do.

5. Give yourself time to get going on a new problem, but don't give up too soon if results aren't immediate.

6. Learn to distinguish between good and bad results, and adjust accordingly.

7. Find an easy way to do something hard.

8. Be consistently honest and clear-sighted, and your network will automatically assume an optimal shape.

9. Determination, interest, and natural ability are the three ingredients in great work.

10. Go on vacation occasionally, but learn something new while there.
```

The response is quite similar to the contextual prompt engineered example. This isn't surprising as it's
using the same contextual information to generate the response. Prompt engineering requires
manually specifying the context, while you can think of RAG as an advanced and automated form of
prompt engineering that leverages databases of documents to retrieve the most optimal context to
guide the generation process.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
