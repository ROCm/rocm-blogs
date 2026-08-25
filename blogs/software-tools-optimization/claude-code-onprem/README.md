---
blogpost: true
blog_title: "Bring Claude Code On Prem with AMD Instinct GPUs"
date: 17 Aug 2026
author: 'Adil Lashab, Eliot Li'
thumbnail: 'claude_code_with_glm52.png'
tags: AI/ML, GenAI, LLM, Serving
category: Software tools & optimizations
target_audience: AI/ML developers, DevOps engineers, platform teams running on-prem GPU infrastructure
key_value_propositions: Shows how to replace Claude Code's Anthropic backend with a self-hosted SGLang server on AMD Instinct MI355X, keeping code and data fully on-prem.
language: English
myst:
    html_meta:
        "author": "Adil Lashab, Eliot Li"
        "description lang=en": "Start running Claude Code securely with a self-hosted SGLang LLM on AMD Instinct MI355X GPUs."
        "keywords": "Claude Code, SGLang, AMD Instinct, MI355X, ROCm, self-hosted LLM, GLM 5.2, on-prem AI, developer tools"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Adil Lashab, Eliot Li"
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

# Bring Claude Code On‑Prem with AMD Instinct GPUs

Agentic coding has become an indispensable part of modern software development. Tools like Claude Code don't just autocomplete lines — they read entire codebases, plan and execute multi-file refactors, run tests, interpret failures, and iterate autonomously until a task is done. Developers who adopt these workflows report dramatic reductions in time spent on boilerplate, debugging, and context-switching. For engineering teams, agentic coding is fast becoming a competitive necessity rather than a convenience.

Claude Code is Anthropic's implementation of this paradigm. By default, it routes every prompt — along with all the code context it reads — through Anthropic's cloud API. For many individual developers and small teams, that arrangement works fine. But for a significant portion of the industry, it is not viable. Organizations handling proprietary algorithms, patient records, financial models, source code under strict IP agreements, or government-classified data cannot send that material to an external API, regardless of the provider's security posture. Regulated industries such as finance, healthcare, and defense operate under compliance frameworks — HIPAA, SOC 2, FedRAMP, ITAR — that explicitly restrict where data may be processed. Beyond compliance, there is the straightforward matter of cost: at scale, per-token cloud pricing accumulates quickly, and teams running continuous agentic workflows can find the economics difficult to justify. Air-gapped environments and low-latency requirements add further constraints that a remote API simply cannot satisfy.

There are already many guides that show how to run Claude Code against a locally hosted LLM — the model runs on the same machine as Claude Code, typically a developer laptop or desktop. That works for small models that fit in consumer GPU memory, but it is the wrong tool for serious coding work. GLM 5.2, for example, requires 756 GB of HBM in FP8 just to load the model weights — a configuration that no developer workstation can provide. Running a model that size on consumer hardware requires heavy quantization, which degrades the reasoning and tool-calling quality that agentic coding depends on. The result is slower, lower-quality output from a machine that is simultaneously trying to run your IDE, browser, and other development tools.

This guide takes a different approach: Claude Code runs on your developer machine, but the model runs on a dedicated GPU server — an AMD Instinct™ GPUs server with the memory and compute to serve GLM 5.2 at full quality. The two are connected by an SSH tunnel. From Claude Code's perspective, the model is local; in practice, it is running on hardware that can actually do the job. This is the same separation of concerns that teams use when they connect to a remote database or a build server — you work locally, the heavy lifting happens on the right hardware.

This guide shows you how to replace the LLM backend in Claude Code with a model running entirely on your own AMD Instinct GPUs, using [SGLang](https://sgl-project.github.io) for serving and **LiteLLM** as a lightweight translation layer. The developer machine setup uses a single interactive script that handles everything from SSH verification through router deployment to the local tunnel and Claude Code launcher installation. Since every token stays on your hardware, there is no data transfer outside of your network, and no per-token billing.

By the end of this guide you will have:

- SGLang serving **GLM 5.2** on your GPU server with fp8 quantization and 8-way tensor parallelism
- A LiteLLM router on the server translating Claude Code's Anthropic Messages API calls into OpenAI API calls against the GLM server
- An SSH tunnel connecting your developer machine to the router
- Claude Code running on your developer machine — CLI or VS Code — talking to your GPU server as if it were Anthropic's API
- All inference running on AMD MI355X, with code and data remaining entirely on your infrastructure

## Why Run a Coding Agent On-Prem?

Coding agents such as Claude Code and Codex rely heavily on LLMs trained specifically for reasoning and coding tasks in the backend to deliver good performance. Such agents typically come with a choice of several LLMs with different tradeoffs between cost, speed, and quality. However, all the LLM choices are hosted in the cloud — there is no built-in option to use a model running locally. While using coding agents with cloud-hosted LLMs is convenient, there are several practical reasons this is not desirable, as summarized in the table below:

| Concern | Cloud-hosted LLMs | LLM on developer machine | LLM on GPU server (this blog) |
| --------- | ------------------- | -------------------------- | ------------------------------- |
| Data privacy | Code and data leave your network | Stays on your machine | Stays on your infrastructure |
| Model quality | State-of-the-art, large models | Limited by consumer GPU memory — small or heavily quantized models only | Full-size, production-quality models (e.g. GLM 5.2 at 756 GB FP8) |
| Cost model | Per-token billing, unpredictable | Hardware you already own | Shared GPU server; fixed infrastructure cost |
| Developer machine impact | None | Competes with IDE, browser, and other tools for GPU and memory | None — inference runs on the server |
| Network setup | None (cloud handles it) | None (same machine, no networking) | SSH tunnel from developer machine to server required |
| Latency | Round-trip to external API | Low (same machine), but slower models | Low (LAN over SSH tunnel), fast models |
| Model control | Depends on the provider's offering | Any model that fits in consumer GPU memory | Any model that fits on server hardware |
| Offline use | Not possible | Works offline | Works air-gapped (server must be reachable) |

The key distinction from the "run a local LLM" guides you may have seen is hardware. Those guides run both Claude Code and the model on the same developer machine. That works well for small models (7B–14B parameters) that fit in consumer GPU memory, but agentic coding workloads benefit significantly from larger models with stronger reasoning and more reliable tool calling. Powerful models such as GLM 5.2 require HBM capacity that a developer workstation cannot provide. Running it on a dedicated server also means your laptop stays responsive: the inference workload goes to the right machine. The trade-off is network setup: because the GPU server is a remote machine typically accessible only over SSH, this guide includes an SSH tunnel that forwards a local port on your developer machine to the LiteLLM router on the server — something the same-machine guides do not need.

The AMD Instinct MI355x GPU (with 288 GB of HBM3 memory, 8 TB/s bandwidth) is purpose-built for exactly this kind of workload. GLM 5.2 is served in FP8 quantization across 8 GPUs, making full use of the combined HBM capacity and bandwidth available in an 8-GPU Instinct server to support large batches and long contexts.

## The Model: GLM 5.2

This blog demonstrates how to use the [GLM 5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) model to power Claude Code. GLM 5.2 (from ZAI) is an open-source MoE model with ~753B total parameters (~40B active per token), a 1M-token context window, and strong coding and reasoning capabilities that make it well-suited for agentic coding workloads. Key properties of this model include:

- **Native tool calling and reasoning** via SGLang's `--tool-call-parser glm47` and `--reasoning-parser glm45` flags — these select the parser matching GLM's chat template format. The `glm47`/`glm45` names refer to SGLang's internal parser identifiers for this model family, not to a model version number.
- **FP8 quantization** — served as `GLM-5.2-FP8` (post-training quantization to 8-bit floating point), enabling memory-efficient deployment across 8 GPUs with minimal quality loss
- **8-way tensor parallelism** — the `--tp 8` flag distributes the model across 8 AMD Instinct GPUs, delivering high throughput for agentic workloads
- **KV cache in fp8_e4m3** — further reduces memory pressure, allowing longer effective context windows
- **1M-token context window** — stably sustains long-horizon agentic tasks across large codebases with the IndexShare architecture that reduces per-token FLOPs by 2.9x at 1M context
- **Speculative decoding via MTP** — the built-in MTP draft layer integrates with SGLang's EAGLE speculative decoding out of the box, improving decode throughput

The tool-calling capability is the critical property. Claude Code's entire agentic loop — reading files, writing edits, running tests, searching the codebase — is built on tool calls. A model that can emit structured tool calls is necessary for agentic coding.

## Architecture

The setup uses three components that form a clean layered stack:

```text
Claude Code (CLI or VS Code)
      |  Anthropic Messages API  (POST /v1/messages)
      v
LiteLLM router  (127.0.0.1:4000 on the server, reached over SSH tunnel)
      |  OpenAI API
      v
GLM-5.2  (SGLang, 127.0.0.1:31090 on the server)
```

**LiteLLM** is the translation layer. Claude Code speaks the Anthropic Messages API; SGLang speaks the OpenAI API. LiteLLM sits between them, accepting Anthropic-format requests from Claude Code and forwarding them as OpenAI-format requests to SGLang — including tool calls, tool results, and streaming. This means Claude Code's full agentic loop works without modification. A plain proxy does not suffice: without LiteLLM, tool arguments are truncated or the client crashes.

The **SSH tunnel** keeps the router private. LiteLLM binds to `127.0.0.1:4000` on the server and is only reachable through your personal tunnel. The keys used in configuration (`sk-glm-local` for the router, `dummy` for the unused upstream field) are local dummies — the SSH tunnel is the actual security boundary.

## Prerequisites

| Requirement | Detail |
| ------------- | -------- |
| SSH access to a GPU server | If `ssh <username>@<server_name> true` succeeds, this works. VPN must be active if your servers require it. |
| A server running GLM-5.2, or the weights | Either the server is already running on port 31090, or you have the model weights and can follow the instructions in [GPU Server Setup](#gpu-server-setup) to start the server. |
| Bash shell on your developer machine | WSL (Ubuntu), Linux, or macOS. Native Windows PowerShell does not work for the setup script — use WSL. |
| VS Code (optional) | Required only for the VS Code extension toggle. The toggle script (`glm-vscode.ps1`) runs on the Windows side. WSL is still needed underneath. |

```{note}
The GPU server side (SGLang, model weights, ROCm) is covered in the next section. If your server is already serving GLM-5.2 on port 31090, skip directly to [Developer Machine Setup](#developer-machine-setup).
```

## GPU Server Setup

If your server is not yet serving GLM-5.2, follow these steps on the GPU server.

### Step 1: Download the Model Weights

Download the GLM 5.2 FP8 model weights to your GPU server:

```bash
pip install huggingface-hub
huggingface-cli download zai-org/GLM-5.2-FP8 --local-dir /data/GLM-5.2-FP8
```

```{note}
The model weights are approximately 756 GB. Ensure sufficient disk space before downloading.
```

### Step 2: Start SGLang Server

Pull the SGLang ROCm Docker image and start a container:

```bash
docker run -d --name glm52_baseline \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size 64g \
  -v /data:/models \
  rocm/sgl-dev:v0.5.15.post1-rocm720-mi35x-20260714 sleep infinity
```

Open a shell inside the container and start the server. Everything after the `docker exec` line runs inside the container, where the SGLang package and the `/models` mount live:

```bash
docker exec -it glm52_baseline bash

# from here on you are inside the container
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SAFETENSORS_FAST_GPU=1

python3 -m sglang.launch_server \
  --model-path /models/GLM-5.2-FP8 \
  --tp 8 --port 31090 --trust-remote-code \
  --enable-expert-parallel \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --served-model-name glm-5.2-fp8
```

The server runs in the foreground and prints its load progress in that shell; wait until it reports it is serving. From another terminal on the server, verify the model is up (allow a few minutes for the model to load):

```bash
curl -s http://localhost:31090/v1/models
```

The expected output is a JSON object listing `glm-5.2-fp8` as the model id that looks like the following. If you see a connection error, the server is still loading.

```text
{"object":"list","data":[{"id":"glm-5.2-fp8","object":"model","created":1784847899,"owned_by":"sglang","root":"glm-5.2-fp8","parent":null,"max_model_len":1048576}]}
```

## Developer Machine Setup

The developer machine package replaces all manual configuration with a single interactive script. You run `glm-setup` once; it handles SSH verification, LiteLLM deployment on the server, tunnel setup, and Claude Code launcher installation. After that, launching Claude Code to use the GLM 5.2 model hosted on your GPU server can be done with a single command `glm-code`.

### The Package Files

| File | Purpose |
| ------ | --------- |
| `glm-setup` | The interactive setup script — this is the one you run |
| `glm-code` | The Claude Code launcher, installed by `glm-setup` |
| `router_ctl.sh` | Start/stop/status control for the LiteLLM router on the server |
| `litellm-config.yaml` | LiteLLM router configuration, deployed to your server |
| `glm-vscode.ps1` | Reversible VS Code toggle (terminal and sidebar), runs on Windows |
| `serve_glm52.sh` | Optional SGLang serve script for the GPU server (used by `glm-setup` if the server is down) |

Copy these files from [github](https://github.com/ROCm/rocm-blogs/tree/release/blogs/software-tools-optimization/claude-code-onprem/src) into a folder on your developer machine (for example, `~/glm/`).

### Step 1: Make Scripts Executable

```bash
chmod +x glm-setup glm-code router_ctl.sh
```

### Step 2: Run the Interactive Setup

```bash
./glm-setup
```

The script asks a series of questions with sensible defaults — press Enter to accept a default or enter the required answers:

| Prompt | Default |
| -------- | --------- |
| SSH username | your current username |
| Server IP or hostname | — |
| SSH port | 22 |
| Local tunnel port | 4000 |
| Router port on server | 4000 |
| GLM server port | 31090 |
| Model ID | `glm-5.2-fp8` |
| Router API key | `sk-glm-local` |
| Corporate CA path | (none) |
| Router install path on server | `glm-selfservice` (under your home) |

Answers are saved to `~/.config/glm-selfservice/config`. Run `./glm-setup --reconfigure` to change them later if needed, or `./glm-setup --check` to re-verify the stack without changing any configuration.

### What `glm-setup` Verifies

The script `glm-setup` works through the stack in order, reporting the result at each layer:

**1. SSH** — confirms it can reach your server. If this fails, it reports the likely cause: VPN not active, key not loaded (`ssh-add -l`), wrong hostname or username, or reservation expired.

**2. Model server on :31090** — checks that the SGLang server is up, prints the model ID it is serving, and runs a real `/v1/chat/completions` test (not just the model list). If the server is down, it looks for a serve script on the server and offers to start it.

**3. Router on :4000** — if LiteLLM is already running it verifies it. If not, it deploys LiteLLM into a venv under your home directory on the server, writes the configuration, starts the router, and proves it works by sending a real Anthropic tool call through and checking that the response comes back as a parsed `tool_use` block with a fully formed `input` object.

**4. Tunnel** — forwards your local port to the server's router port over SSH.

**5. Client** — installs `glm-code` into `~/.local/bin`. If you are on WSL and opt in to using the GLM 5.2 model in VS Code as well, it will run the VS Code toggle to configure both the integrated terminal and the sidebar to do so.

## Launching Claude Code

### CLI Mode

To launch Claude Code CLI powered by the GLM 5.2 model hosted on your MI355X server, simply run:

```bash
glm-code       # interactive Claude Code on your GLM-5.2
```

The command `glm-code` reads the saved config, brings the SSH tunnel up if it is not already running, checks each layer in order, and starts Claude Code. If something is off it tells you which layer failed: SSH, router, or the GLM server behind the router. It can distinguish a healthy router with a dead model server from a dead router — the router answers `/v1/models` from its own config even when the model server is gone, so it sends a real request to find out.

You should get the familiar Claude Code CLI interface similar to the one shown below if Claude Code starts successfully.  Note that the API Key and API Base URL have been overridden to use the GLM 5.2 model served on your MI355X server.

```text
╭───────────────────────────────────────────────────╮
│ ✻ Welcome to Claude Code!                        │
│                                                   │
│   /help for help, /status for your current setup  │
│                                                   │
│   cwd: <current path>                             │
│                                                   │
│   ─────────────────────────────────────────────── │
│                                                   │
│   Overrides (via env):                            │
│                                                   │
│   • API Key: sk-…                                 │
│   • API Base URL: http://127.0.0.1:4000           │
╰───────────────────────────────────────────────────╯

 Tips for getting started:

  Run /init to create a CLAUDE.md file with instructions for Claude
  Use Claude to help with file analysis, editing, bash commands and git
  Be as specific as you would with another engineer for the best results

> which model is powering you?

● glm-5.2-fp8

╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >                                                                                                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
  ? for shortcuts
```

To run Claude Code with a single prompt or verify it is set up correctly, use the following commands:

```bash
glm-code -p "explain this repo"   # headless one-shot
glm-code --check                  # bring up the tunnel and check the layers
```

### VS Code Extension

Claude Code appears in VS Code in two spots that read configuration from different places:

- **Integrated terminal**: runs `claude` inside WSL, reads `terminal.integrated.env.linux` in your VS Code user settings
- **Sidebar (Claude Code panel)**: on Windows runs `claude.exe` on the extension host, reads the `env` block in `%USERPROFILE%\.claude\settings.json`

The `glm-vscode.ps1 on` command writes both: the terminal environment block in VS Code user settings, and an `env` block in `~/.claude/settings.json` with the same port, key, and model ID. It also sets `ANTHROPIC_CUSTOM_HEADERS` to empty, which drops the inherited subscription header so the sidebar stops routing requests to Anthropic's gateway and goes to your router instead. Your other settings in both files are left alone.

Setting `glm-vscode.ps1 off` restores both files byte-for-byte from the most recent timestamped backup.

The script `glm-setup` runs this for you if you opt in. To drive the toggle from Windows PowerShell manually, use:

```powershell
powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 status
powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 on  -Port 4000 -Model glm-5.2-fp8 -Key sk-glm-local -Root /home/<username>/.glm-selfservice
powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 off
```

```{note}
After running `glm-vscode.ps1 on` or `glm-vscode.ps1 off`, you must fully quit every window and reopen VS Code. A reload is not sufficient because the extension host only reads `~/.claude/settings.json` and the terminal environment when it starts. Make sure the tunnel is up (`glm-code --check` in WSL) and the router is running on the server before opening Claude Code.
```

Backups are written under `%USERPROFILE%\.glm-selfservice\`: VS Code settings in `vscode-backups\` and `~/.claude/settings.json` in `claude-backups\`. Both are timestamped, and `off` takes the newest of each.

```{note}
Setting `glm-vscode.ps1 on` rewrites files through a JSON parser. If either file contains `//` comments, the parser will fail before writing anything. Remove the comments from those files, or set the affected file by hand.
```

## Verifying Claude Code Is Hitting Your GLM-5.2 Model

### Check 1: Tail the Router Log

While using Claude Code, tail the LiteLLM router log on the server:

```bash
ssh <username>@<server_name> tail -f ~/glm-selfservice/logs/litellm.log
```

If request entries appear in the log as you type in Claude Code, all traffic is routing through your GLM-5.2. If the log is silent, the client is not going through your router.

### Check 2: End-to-End Tool Call Test

Prove the router end-to-end without Claude Code. This separates a working router from a broken one — the response must come back as a `tool_use` block with a fully parsed `input` object:

```bash
ssh <username>@<server_name> 'curl -s http://127.0.0.1:4000/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: sk-glm-local" \
  -H "anthropic-version: 2023-06-01" \
  -d "{\"model\":\"glm-5.2-fp8\",\"max_tokens\":512,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Use the write_file tool to create hello.txt containing exactly: banana\"}],\"tools\":[{\"name\":\"write_file\",\"description\":\"Write text to a file\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}]}"'
```

The expected response contains:

```json
{
  "type": "tool_use",
  "name": "write_file",
  "input": {"path": "hello.txt", "content": "banana"},
  "stop_reason": "tool_use"
}
```

Truncated arguments or a missing `input` field indicates the router configuration is incorrect.

## Revert the Setting

### Claude Code CLI

Nothing in your global environment was changed. `glm-code` sets environment variables only for the Claude Code process it launches.
To launch Claude Code with your normal settings (e.g. use the Anthropic API), simply use the `claude` command as before.

To remove the developer machine installation entirely, simply run the following commands:

```bash
rm ~/.local/bin/glm-code
rm -rf ~/.config/glm-selfservice
rm -rf ~/.glm-selfservice
```

### VS Code

To revert the setting in VS Code, run the following powershell command:

```powershell
powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 off
```

This restores the most recent backup of both VS Code settings and `~/.claude/settings.json` byte-for-byte. Fully quit and reopen all VS Code windows after running this.

### Server

To stop the LiteLLM router on the server, run the following command:

```bash
ssh <username>@<server_name> ~/glm-selfservice/router_ctl.sh stop
```

The router binds to `127.0.0.1` on the server and does not touch the SGLang server. The venv and configuration files persist so you can restart the router later with `router_ctl.sh start`.

## Troubleshooting

Work through the stack from the bottom up. `glm-code --check` runs the same layer checks the setup script uses and tells you exactly where the failure is.

| Layer | Symptom | What to check |
| ------- | --------- | --------------- |
| SSH | Cannot reach server | VPN active? Key loaded (`ssh-add -l`)? Correct hostname and username? Reservation still valid? Test with `ssh <username>@<server_name> true`. |
| GLM server | Router reports `:31090` connection error | The GLM server is down or still loading. Check directly: `ssh <username>@<server_name> "curl -s http://127.0.0.1:31090/v1/models"`. Wait for it to load. Do not restart a shared server yourself. |
| Router | Tunnel up but router does not answer | Check status: `ssh <username>@<server_name> ~/glm-selfservice/router_ctl.sh status`. Restart if down: `router_ctl.sh restart`. After a server reboot the router process is gone (venv and config persist); start it again with `router_ctl.sh start`. |
| Model not found | Claude Code returns model-not-found errors | The router config includes a `"*"` catch-all that routes any model name to the GLM server. If this was removed, set the model ID in your config to exactly what `curl http://127.0.0.1:31090/v1/models` reports. |
| VS Code sidebar | Sidebar still talks to Anthropic after `on` | Files had `//` comments and the JSON parser stopped before writing. Remove the comments and rerun `glm-vscode.ps1 on`, then fully restart VS Code. |

## Summary

Claude Code's agentic coding capabilities — file editing, bash execution, test running, multi-file refactoring — work fully against a self-hosted SGLang backend. The setup requires three components: a model with native tool-calling support (GLM 5.2 served by SGLang), a LiteLLM router that translates Claude Code's Anthropic Messages API calls into OpenAI API calls, and an SSH tunnel that delivers the router securely to your developer machine.

The developer machine package reduces this to a single script. Run `./glm-setup` once, answer the prompts, and everything from router deployment to tunnel setup to Claude Code launcher installation is handled automatically. Launch Claude Code CLI with `glm-code` from a terminal, or use Claude Code in VS Code with a one-time toggle for the extension setting.

The AMD Instinct MI355X GPU is well-suited for this workload. GLM 5.2 is served in FP8 format across 8 GPUs, distributing the model weight and KV cache across the full HBM capacity of an 8-GPU Instinct server. The MI355X GPU's 8 TB/s HBM bandwidth pushes throughput further while keeping latency low. Your code stays on your hardware, your spend is predictable, and your team gets the full Claude Code agentic experience without leaving your network.

## Additional Resources

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/overview)
- [SGLang documentation](https://sgl-project.github.io)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [GLM 5.2 on Hugging Face](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [LiteLLM documentation](https://docs.litellm.ai)
- [ROCm documentation](https://rocm.docs.amd.com/)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, AMD Instinct, AMD ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. NVIDIA, CUDA, RAPIDS, and cuVS are trademarks and/or registered trademarks of NVIDIA Corporation in the United States and other countries. PyTorch is a registered trademark of The Linux Foundation. Llama is a trademark of Meta Platforms, Inc. Ollama is a trademark of Ollama, Inc. OpenAI, ChatGPT, and GPT are trademarks of OpenAI, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

© 2026 Advanced Micro Devices, Inc. All rights reserved.
