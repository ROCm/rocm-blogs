---
blogpost: true
blog_title: "HPC Coding Agent - Part 1: Combining GLM-powered Cline and RAG Using MCP"
date: 3 Dec 2025
author: 'Johanna Yang, Albin Toft'
thumbnail: 'rag-robot-thumbnail.png'
tags: AI/ML, HPC, Scientific Computing, GenAI
category: Applications & models
target_audience: AI practitioners, HPC developers
key_value_propositions: Showcase AMD Instinct GPU capabilities for HPC RAG agent development using GLM-4.6, a state-of-the-art agentic LLM, with streamlined CLI-based workflows
language: English
myst:
    html_meta:
        "author": "Johanna Yang, Albin Toft"
        "description lang=en": "Build an HPC RAG agent on AMD Instinct GPUs using GLM-4.6, Cline and ChromaDB."
        "keywords": "RAG agent, HPC, LUMI supercomputer, GLM4.6"
        "vertical": "HPC, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Johanna Yang, Albin Toft"
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

# HPC Coding Agent - Part 1: Combining GLM-powered Cline and RAG Using MCP

Navigating through extensive High Performance Computing (HPC) documentation can be challenging, especially when working with complex supercomputer environments like [LUMI](https://lumi-supercomputer.eu/), one of the pan-European pre-exascale supercomputers. Traditional search methods often fall short when you need contextual, actionable answers to technical questions. RAG (Retrieval-Augmented Generation) agents offer a solution by combining large language model reasoning with domain-specific knowledge retrieval to provide accurate, cited responses to your HPC queries.

This blog walks you through building an HPC RAG agent on AMD Instinct GPUs using [GLM-4.6](https://z.ai/blog/glm-4.6), an agentic language model built for complex reasoning and code generation. We cover three main components: deploying GLM-4.6 with vLLM, setting up a vector database with ChromaDB to index LUMI supercomputer documentation, and configuring [Cline](https://github.com/cline/cline) to tie everything together.

This blog post serves as the inaugural entry in a comprehensive series dedicated to the development of a sophisticated coding agent for high-performance computing (HPC) environments. The primary objective of this series is to systematically explore, architect, and showcase the essential components and advanced capabilities that underpin an effective HPC coding agent solution.

Beyond this introductory blog, two additional installments are currently in development to further deepen the technical discourse. The forthcoming second part will delve into performance profiling, illustrating how the coding agent can be transformed into an expert assistant for GPU utilization analysis by integrating Model Context Protocol (MCP) profiling tools. The third part will focus on code optimization techniques, demonstrating how the agent leverages OpenEvolve to analyze, refactor, and enhance existing application code for improved efficiency and performance on AMD Instinct GPUs.

Together, this series aims to provide a practical blueprint for assembling end-to-end, AI-driven development workflows tailored for the unique challenges of HPC, offering readers actionable strategies and tool integrations that accelerate productivity and insight.

By the end of this guide, you'll have a working RAG agent running on AMD hardware that can query HPC documentation and provide answers to domain-specific questions.

# Implementation

## Requirements

* AMD GPU: See the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems.
* ROCm ≥6.4: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) for installation instructions.
* Docker: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) for installation instructions.
* AMD Instinct MI300X GPUs (8 GPUs recommended for optimal GLM-4.6 performance)

## Implementation: Step I - Serving GLM-4.6 with vLLM

GLM-4.6 is a cutting-edge open-source language model developed by [Zhipu AI](https://www.zhipuai.cn/en/), featuring 355 billion parameters and a 200K context window. Designed specifically for agentic workflows, it excels at multi-step reasoning, advanced tool calling and code generation. Its enhanced agent capabilities and function execution make it particularly well-suited for complex HPC documentation assistance tasks that require deep context understanding and precise retrieval operations.

### Launching the Docker Container

If you have AMD GPUs and the AMD Container Toolkit installed on your system, we recommend using it for better GPU management.

```bash
docker run -it --rm --runtime=amd \
  -e AMD_VISIBLE_DEVICES=all \
  --shm-size=64g \
  --name serve-glm \
  -v $(pwd):/workspace -w /workspace \
  rocm/vllm-dev:nightly_main_20251127
```

Alternatively, if the AMD Container Toolkit is not installed, run it directly without the toolkit:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --shm-size=64g \
  --name serve-glm \
  -v $(pwd):/workspace -w /workspace \
  rocm/vllm-dev:nightly_main_20251127
```

We use a specific dated build tag (like `nightly_main_20251127`) rather than `nightly` to ensure reproducibility, as installation requirements may change in future nightly builds. You can find available tags on the [ROCm vLLM Docker Hub page](https://hub.docker.com/r/rocm/vllm-dev/tags).

### Installing Dependencies

Inside the container, install the required dependencies:

```bash
pip uninstall vllm 
pip install --upgrade pip
cd /workspace/GLM
git clone https://github.com/zai-org/GLM-4.5.git
cd GLM-4.5/example/AMD_GPU/
pip install -r rocm-requirements.txt
git clone https://github.com/vllm-project/vllm.git
cd vllm 
export PYTORCH_ROCM_ARCH="gfx942"  # gfx942 corresponds to MI300 architecture
python3 setup.py develop
```

Wait until the compilation completes.

```{note}
The GLM-4.5 repository contains AMD-specific dependencies and examples that work for both GLM-4.5 and GLM-4.6 models. We use the same installation and serving approach for both versions.
```

### Downloading and Serving the Model

First, download the GLM-4.6 model to your local directory:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download zai-org/GLM-4.6 --local-dir /localdir/GLM-4.6
```

Now serve the model with vLLM. For RAG agent workflows, we enable automatic tool calling capabilities through `--enable-auto-tool-choice` and `--tool-call-parser glm45`. These flags allow GLM-4.6 to automatically detect when to invoke external tools (like ChromaDB for document retrieval) and parse function calls in the GLM-specific format, which is essential for Cline to orchestrate multi-step agent tasks such as searching documentation, retrieving context, and generating answers. For optimal performance, we recommend using 8 GPUs with substantial GPU memory. We use `--gpu-memory-utilization 0.90` to provide adequate memory headroom and prevent out-of-memory (OOM) errors during heavy workloads.

**Running on fewer GPUs:** If you have access to fewer GPUs, you can still run GLM-4.6 by adjusting `--tensor-parallel-size` (e.g., 4 for four GPUs) and reducing `--max-num-seqs` and `--max-model-len` to fit within the available VRAM. Performance may not be optimal, depending on use-case, yet the system remains functional for RAG workloads.

```bash
VLLM_USE_V1=1 vllm serve /localdir/GLM-4.6 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-xxxx \
  --served-model-name GLM-4.6
```

This command serves GLM-4.6 across 8 GPUs with tensor parallelism, which will enable efficient inference for RAG workloads.

### Verifying the Server

Verify the GLM-4.6 server is running by sending a test request (replace `<Your-Server-IP>` with your server's IP address, or use `localhost` if running locally):

```bash
curl -H "Authorization: Bearer token-xxxx" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "GLM-4.6",
       "messages": [{"role": "user", "content": "Hello! Can you help me with HPC and SLURM?"}],
       "max_tokens": 100,
       "temperature": 0.7
     }' \
     http://<Your-Server-IP>:8000/v1/chat/completions
```

You should receive a response from GLM-4.6 about HPC and Slurm. Now you have a powerful language model ready to serve your RAG agent! In the next section, we'll configure the Cline agent to connect to this server.

## Implementation: Step II - Configuring the Cline Agent

### Installing Cline CLI

Cline is a CLI-based agentic framework that enables autonomous task execution with LLMs. It supports custom model endpoints, making it perfect for integrating with our self-hosted GLM-4.6 server. While Cline also comes as a Visual Studio Code extension, this guide focuses on the CLI version running inside a container for better reproducibility.

Cline operates in two modes that users can switch between:

* **Plan Mode**: The agent analyzes the task, breaks it down into steps, and presents a plan before taking action. This mode provides better visibility into the agent's reasoning process.
* **Act Mode**: The agent directly executes tasks without presenting a plan first, making it suitable for straightforward operations where immediate action is preferred.

### Setting Up the Cline Container

For optimal resource allocation, we'll run Cline CLI in a separate container with 1 GPU dedicated to it. This separation allows the agent to operate independently while the main GLM-4.6 server handles inference workloads. Launch a new container using the same ROCm image:

```bash
docker run -it --rm --runtime=amd \
  -e AMD_VISIBLE_DEVICES=0 \
  --shm-size=16g \
  --name cline-agent \
  -v $(pwd):/workspace -w /workspace \
  rocm/vllm-dev:nightly_main_20251127
```

### Installing Node.js and Cline

Inside the Cline container, install Node.js and Cline:

```bash
# Install Node.js if not already installed
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Install Cline CLI
npm install -g cline
```

### Configuring Authentication

Configure Cline using the interactive authentication wizard:

```bash
cline auth
```

Follow the interactive wizard with these selections (replace `<GLM-Server-IP>` with your GLM server's IP address):

```text
┌─ What would you like to do?
│ > Configure BYO API providers
│
┌─ What would you like to do?  
│ > Add or change an API provider
│
┌─ Select API provider type:
│ > OpenAI Compatible
│
┌─ API Key: (CRITICAL: No leading/trailing spaces!)
│ token-xxxx
│
┌─ Base URL (optional, for OpenAI-compatible providers):
│ http://<GLM-Server-IP>:8000/v1/
│  
┌─ Model ID:
│ GLM-4.6
│
┌─ What would you like to do?
│ > Return to main auth menu 
│
┌─ What would you like to do?
│ > Exit authorization wizard
```

If Cline defaults to OpenRouter instead of your OpenAI-compatible provider, manually set the API providers:

```bash
cline config set plan-mode-api-provider=openai
cline config set act-mode-api-provider=openai
```

### Verifying Configuration

Verify your configuration:

```bash
cline config list
```

The expected output should show:

```text
act-mode-api-provider: openai
plan-mode-api-provider: openai
open-ai-base-url: http://<GLM-Server-IP>:8000/v1/
open-ai-api-key: ******** 
plan-mode-open-ai-model-id: GLM-4.6
```

### Starting Cline

Start Cline CLI:

```bash
cline
```

You should see Cline start up and display a prompt. If configured correctly, it will connect to your GLM server and be ready to assist with coding tasks.

In the next section, we give the agent access to LUMI documentation via a vector database and [MCP (Model Context Protocol)](https://modelcontextprotocol.io/docs/getting-started/intro) tooling.

## Implementation: Step III - Building a Vector Database for RAG

When building AI agents with RAG capabilities, the first critical step is creating a vector database. This database serves as the agent's knowledge base, allowing it to retrieve relevant information on-demand rather than relying solely on its internal knowledge. In this guide, we'll walk through the process of setting up a vector database using real-world documentation from the LUMI supercomputer project.

### Why LUMI Documentation?

For this implementation, we chose to use documentation from the [LUMI supercomputer user guide](https://github.com/Lumi-supercomputer/lumi-userguide). This repository contains comprehensive technical documentation written in Markdown, perfect for demonstrating how to parse, chunk, and store structured information. The principles here apply to any documentation set you want to make searchable for your AI agent.

### Setting Up the Knowledge Source

First, we need to obtain the documentation locally:

```bash
git clone https://github.com/Lumi-supercomputer/lumi-userguide.git
```

This gives us a rich corpus of Markdown files that our agent will eventually be able to query and reference.

### Installing Python Dependencies

Before we can process the documentation, we need a few key libraries:

```bash
pip install langchain langchain_community chromadb unstructured markdown
```

Here's what each package does:

* **langchain & langchain_community**: Provides utilities for document loading and text processing
* **chromadb**: Our vector database for storing and retrieving embeddings
* **unstructured**: Handles parsing of various document formats
* **markdown**: Ensures proper Markdown processing

### Parsing and Chunking the Documentation

Raw documentation needs to be broken down into manageable pieces. This is where chunking comes in. We split long documents into smaller segments that can be embedded and retrieved independently.

```{note}
The complete vector database build code shown in Step III also exists as a runnable script at `hpc-agent-rag/src/build_vector_db.py` inside the project for convenience.
```

```python
import glob
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("lumi-userguide/docs/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```

The `RecursiveCharacterTextSplitter` is particularly effective. It tries to split on natural boundaries (paragraphs, sentences) rather than arbitrarily cutting text. The `chunk_size=1000` parameter limits each chunk to 1000 characters, while `chunk_overlap=200` ensures that chunks share 200 characters with their neighbors, preventing information from being lost at chunk boundaries.

### Storing Everything in ChromaDB

With our documents parsed and chunked, the final step is storing them in a vector database. ChromaDB is an excellent choice here. It is lightweight, easy to use, and handles embeddings automatically.

```python
import chromadb

# For a persistent client
chroma_client = chromadb.PersistentClient(path="/root/.chroma_data") 

# Or for an in-memory client
# chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="LUMI_documentation")

ids = [f"doc_{i}" for i in range(len(chunks))] # Unique IDs for each chunk
texts = [chunk.page_content for chunk in chunks]

collection.add(
        documents=texts,
        ids=ids
    )
```

The `PersistentClient` saves your embeddings to disk, which means you won't need to reprocess documents every time your application restarts. ChromaDB automatically generates embeddings for each text chunk using a default embedding model, making the entire process remarkably straightforward.

### What's Next?

With your vector database populated, you now have a searchable knowledge base that your AI agent can query. The next step is integrating this database with your agent's retrieval pipeline, which allows it to search for relevant context based on user queries and inject that information into its responses.

In the following section, we will show how the Cline coding agent can be given access to this database through the use of the official MCP tool for ChromaDB [chroma-mcp](https://github.com/chroma-core/chroma-mcp).

## Implementation: Step IV - Connecting Cline to Your Vector Database

Having a vector database is great, but it's only truly powerful when your AI agent can actually interact with it. This is where [chroma-mcp](https://github.com/chroma-core/chroma-mcp) comes into play. It's an official Model Context Protocol (MCP) server specifically designed for ChromaDB, acting as a bridge that allows AI coding agents like Cline to query and retrieve information from your vector database.

The Model Context Protocol is a standardized way for AI applications to communicate with external data sources and tools. By setting up chroma-mcp, we're essentially giving Cline the ability to "see" and search through the LUMI documentation we just indexed, which allows it to provide contextually relevant answers based on real documentation rather than just its training data.

### Configuring Cline for ChromaDB Access

Setting up the MCP server is straightforward. You'll need to modify Cline's configuration file to register the chroma-mcp server. Open your Cline config file at `~/.cline/data/settings/cline_mcp_settings.json` and add the following configuration:

```json
{
  "mcpServers": {
    "chroma": {
      "command": "uvx",
      "args": [
        "chroma-mcp",
        "--client-type",
        "persistent",
        "--data-dir",
        "/root/.chroma_data"
      ]
    }
  }
}
```

Let's break down what's happening here:

* **"chroma"**: The name we're giving this MCP server (you can reference it by this name)
* **"command": "uvx"**: Uses `uvx` to run the chroma-mcp tool without needing a separate installation
* **"--client-type": "persistent"**: Tells chroma-mcp to connect to a persistent ChromaDB instance (as opposed to in-memory)
* **"--data-dir": "/root/.chroma_data"**: Points to the directory where your ChromaDB data is stored (make sure this matches the path you used when creating your `PersistentClient`)

### Verifying the Setup

Once you've added the configuration, restart Cline to load the new MCP server. To verify that everything is working correctly, start Cline in plan mode and test the RAG setup with a few questions.

First, verify that Cline can access the ChromaDB collections:

**Query:** "What collections are available in chroma?"

**Cline's response:**

```text
## API request completed  ↑ 9.9k ↓ 142 $0.0000 
I found one collection available in your Chroma database:

LUMI_documentation

This appears to be the only collection currently stored in your Chroma instance. The name suggests it likely contains documentation related to LUMI (possibly the LUMI supercomputer or a LUMI-related project).

Would you like me to get more information about this collection, such as:

• How many documents it contains
• A peek at some sample documents
• The collection's metadata and configuration
```

Now test the RAG retrieval by asking a specific question about LUMI:

**Query:** "What does the LUMI documentation say about installing Python libraries directly on a node?"

**Cline's response with retrieved context:**

```text
Based on the LUMI documentation, here's what it says about installing Python libraries directly on a node:

## Strong Warning: Don't Install Python Packages Directly

The LUMI documentation strongly discourages installing Python packages directly to the user home folder,  /scratch ,  /project , etc. using Conda, pip, or similar package management tools.

## Why Direct Installation is Discouraged

The main reason is file system performance issues:

• A typical Conda environment contains tens to hundreds of thousands of relatively small files
• Installing such a large number of small files to shared storage locations puts significant strain on the Lustre file system
• Lustre isn't designed for handling many small files being loaded simultaneously by multiple processes
• This impacts file system performance for all users since it's a shared resource

## Recommended Alternatives

Instead of direct installation, LUMI recommends:

1. Use Singularity/Apptainer containers (generally recommended)
  • Solves the "many small files" performance problem
  • Makes it easy to manage multiple different Python environments
  • Can use existing containers or build custom ones
2. Use the LUMI container wrapper
  • Developed by CSC to wrap software installations inside containers
  • Improves startup times, reduces I/O load, and lessens file count on file systems
3. For minimal needs (less than 5 packages including dependencies):
  • Use the pre-installed  cray-python  module
  • Install a pip environment for use with a container
4. Use existing containers when available (e.g., PyTorch ROCm container)
5. Build custom containers using tools like  cotainr  if you can't find suitable existing ones
```

If configured correctly, Cline should be able to access the chroma-mcp tool and query your indexed documentation. This transforms Cline from a general-purpose coding assistant into one with specialized knowledge of LUMI supercomputer workflows through the power of RAG.

With this setup complete, your AI agent now has real-time access to domain-specific documentation, enabling it to provide accurate, contextually relevant assistance for tasks related to the LUMI system.

## Summary

In this blog, we built an HPC RAG agent from the ground up on AMD Instinct GPUs, demonstrating the full stack from model serving to intelligent query answering. You learned how to deploy GLM-4.6 with vLLM for efficient inference, configure Cline CLI as an agentic orchestrator, and integrate ChromaDB with MCP for semantic search over LUMI documentation.

AMD Instinct GPUs provide the compute power and memory bandwidth needed for both LLM inference and vector similarity search, making them ideal for HPC RAG workloads. The combination of GLM-4.6's strong reasoning capabilities, Cline's flexible CLI-based approach, and ChromaDB's efficient vector storage creates a powerful, customizable RAG system tailored for supercomputer environments. Ready to take your HPC documentation workflows to the next level? Try building your own RAG agent and experience the power of AI-assisted technical support running entirely on AMD hardware!

## Acknowledgments

We thank [Balazs Toth](https://rocm.blogs.amd.com/blog/author/balazs-toth.html), Kristoffer Peyron, [Arttu Niemela](https://rocm.blogs.amd.com/authors/arttu-niemela.html), Daniel Warna, [Baiqiang Xia](https://rocm.blogs.amd.com/authors/baiqiang-xia.html), and [Sopiko Kurdadze](https://rocm.blogs.amd.com/authors/sopiko-kurdadze.html) for valuable discussions and technical support.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
