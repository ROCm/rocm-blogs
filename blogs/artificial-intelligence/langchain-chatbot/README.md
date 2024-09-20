---
blogpost: true
date: 11 Mar 2024
author: Phillip Dang
tags: PyTorch, AI/ML, Fine-Tuning
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Question-answering Chatbot with LangChain"
    "Author": "Phillip Dang"
    "keywords": "PyTorch, LangChain, Chatbot, RAG, FAISS, AMD, GPU, MI300, MI250, Tuning, Fine-Tuning"
    "property=og:locale": "en_US"
---

# Question-answering Chatbot with LangChain on an AMD GPU

<span style="font-size:0.7em;">11, Mar 2024 by {hoverxref}`Phillip Dang<phildang>`. </span>

LangChain is a framework designed to harness the power of language models for building
cutting-edge applications. By connecting language models to various contextual sources and providing
reasoning abilities based on the given context, LangChain creates context-aware applications that can
intelligently reason and respond. In this blog, we demonstrate how to use LangChain and Hugging
Face to create a simple question-answering chatbot. We also demonstrate how to augment our large
language model (LLM) knowledge with additional information using the Retrieval Augmented
Generation (RAG) technique, then allow our bot to respond to queries based on the information
contained within specified documents.

## Prerequisites

To run this blog, you will need the following:

* **AMD GPUs**: [AMD Instinct GPU](https://www.amd.com/en/products/accelerators/instinct.html).
* **Linux**: see the [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
* [ROCm 6.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Alternately, you can launch a docker container with the same settings as above, replace ```/YOUR/FOLDER``` with a location of your choice to mount the directory onto the docker root directory. Here is a example using ROCm 6.2 with PyTorch 2.3:

  ```bash
  docker run -it --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device=/dev/dri -v /YOUR/FOLDER:/root rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
  ```

To check your hardware and make sure that the system recognizes your GPU, run:

```bash
! rocm-smi --showproductname
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

Next, make sure PyTorch detects your GPU:

```python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

Your output should look like this:

```bash
number of GPUs: 1
['AMD Radeon Graphics']
```

## Libraries

To build a chatbot that can chat with documents, you'll need three tools:

* LangChain
* A language model
* RAG with Facebook AI Similarity Search (FAISS)

### LangChain

[LangChain](https://www.LangChain.com/) serves as a structure to create language model-driven
applications. It allows applications to:

* **Embrace contextuality** by linking a language model with contextual sources (such as prompts,
  examples, or relevant content) to enrich its responses.
* **Engage in reasoning** by depending on a language model to logically deduce answers based on
  the given context, and determine the appropriate actions to take.

To install LangChain, run `pip install langchain langchain-community`.

### Language model

In this blog, we use [Google Flan-T5-large](https://huggingface.co/google/flan-t5-large) as our underlying
language model.

To install our language model and chat with documents, run the following code:
`pip install transformers sentence-transformers`.

### RAG with FAISS

While LLMs are intelligent across a variety of domains, their knowledge is limited to public information
available to them when their training is concluded. If we want the model to consider private
information or post-training data, we have to include the additional information ourselves. This
addition process is RAG, while the tool for efficiently retrieving relevant information is FAISS.

[FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
is a library for efficient similarity search and clustering of dense vectors. It's widely used for tasks like
nearest neighbor search, similarity matching, and other related operations in large datasets. It helps us
to efficiently store new information and retrieve the most relevant chunks of information given our
query.

To install FAISS, run `pip install faiss-cpu`.

## Q&A chatbot

Start by setting up your language model. To do this, you must have a
[Hugging Face API Token](https://huggingface.co/docs/hub/security-tokens).

```python
import os
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your Huggingface API Token here"

llm = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={'temperature':0.5,
                                                               'max_length': 512})
```

Once you have your model, you can put things together via LangChain's LLMChain. LLMChain utilizes a
PromptTemplate to structure user inputs, which are then sent to your LLM for handling. This makes
LLMChain a valuable tool for generating coherent language.

```python
template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
```

Now for the fun part--let's ask the chatbot a few questions:

Input:

```python
question =  "What is the capital of Ecuador?"
llm_chain.run(question)
```

Output:

```text
'Quito is the capital city of Ecuador. Quito is located in the north of the country. The answer: Quito.'
```

Input:

```python
question =  "What is GTA? "
llm_chain.run(question)
```

Output:

```text
'GTA is an abbreviation for Grand Theft Auto. GTA is a video game series. The answer: video game series.'
```

Input:

```python
question =  "What are some key advantages of LoRA for LLM?"
llm_chain.run(question)
```

Output:

```text
'LoRA is a centralized repository for all LLM degree work. The LLM degree program at the University of
Michigan was the first to use LoRA for their degree program. The University of Michigan School of Law
is the first law school in the United States to use LoRA for their degree program.'
```

The answer to the last question is incorrect. This is likely because the model's training data did not include information on LoRA. We'll address this in the next section by applying the RAG technique.

## Q&A chatbot with RAG

Per the previous section, the model incorrectly answered our question on the LoRA technique--likely
because the information was not available at the time the model was trained. To resolve this issue, you
can include the information in your model using RAG.

RAG works in two phases:

1. Retrieval phase: Given a query (e.g., a clinical question), the model searches a large database to find
  relevant documents or snippets.
2. Generation phase: The model uses the retrieved information to generate a response, ensuring that
  the output is based on the input data, which in our case will be a PDF.

To see this in action, you'll need to create two functions, one to process our input data (a PDF paper on
LoRA), and one to build our knowledge database.

```python
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def process_text(text):
    # Split the text into chunks using LangChain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=256, chunk_overlap=64, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase
```

```python
import PyPDF2
import requests
from io import BytesIO

# read the PDF paper 
pdf_url = "https://arxiv.org/pdf/2106.09685.pdf" 
response = requests.get(pdf_url)
pdf_file = BytesIO(response.content)
pdf_reader = PyPDF2.PdfReader(pdf_file)

def get_vectorstore():
    # build vectorstore from pdf_reader
    text = ""
    # Text variable will store the pdf text
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Create the knowledge base object
    db = process_text(text)
    return db

db = get_vectorstore()
```

Now, put everything together by loading the Q&A chain from LangChain, searching the knowledge
database for the most relevant information, and seeing if the chatbot provides a more accurate answer
to the question from the previous section:

```python
from langchain.chains.question_answering import load_qa_chain
# loading Q&A chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=)

query = "what are some key advantages of LoRA for LLM?"
# search database for relevant information
docs = db.similarity_search(query=query)

# Run our chain
chain.run(input_documents=docs, question=query)
```

Output:

```text
'LORA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when
using adaptive optimizers since we do not need to calculate the gradients or cantly fewer GPUs and
avoid I/O bottlenecks. Another benefit is that we can switch between tasks while deployed at a much
lower cost by only swapping the LoRA weights as opposed to all the'
```

The updated answer is much more relevant after providing the additional information to our model
using the `input_documents=docs` argument.

We recommend testing different LLMs as the base model and trying various LLMChains for different
use cases. We also encourage experimenting with different processing methods, and segmenting the
input documents to improve similarity search relevance.

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
