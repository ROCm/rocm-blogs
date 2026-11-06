---
blogpost: true
blog_title: "PrimusClaw: An Open-Source Managed Agent Runtime for AMD Instinct GPU Clusters"
date: "06 Nov 2026"
author: "Yihan Wang, Lei Zhang, Lei Wei, Xiaofei Zheng, Chaojun Hou, Zhenyu Gu"
thumbnail: 'primusclaw-thumbnail.png'
tags: "AI/ML, GenAI, Kubernetes"
category: "Software tools & optimizations"
target_audience: "AI platform engineers who host autonomous agents for many users, teams building GPU-optimization and coding agents on AMD Instinct GPUs, and infrastructure decision-makers comparing managed agent services with self-hosted alternatives."
key_value_propositions: "PrimusClaw gives each agent run what managed agent services provide: a durable session, a sandboxed computer, a model gateway, and standard APIs. It is open source, runs on your own Kubernetes and AMD Instinct™ GPU clusters, and is designed to keep runs going across rolling upgrades and worker failures."
language: English
myst:
    html_meta:
        "author": "Yihan Wang, Lei Zhang, Lei Wei, Xiaofei Zheng, Chaojun Hou, Zhenyu Gu"
        "description lang=en": "PrimusClaw is an open-source managed agent runtime that hosts durable, sandboxed agent sessions on your own AMD Instinct GPU clusters."
        "keywords": "PrimusClaw, managed agents, agent runtime, agent sandbox, MCP, A2A, Kubernetes, AMD Instinct, ROCm, Hyperloom"
        "vertical": "AI, Developers, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Deploying AI at Scale, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yihan Wang, Lei Zhang, Lei Wei, Xiaofei Zheng, Chaojun Hou, Zhenyu Gu"
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

# PrimusClaw: An Open-Source Managed Agent Runtime for AMD Instinct GPU Clusters

**[PrimusClaw](https://github.com/AMD-AGI/PrimusClaw) is an open-source managed agent runtime: a session store, an agent harness, and per-session sandboxes that you run in your own Kubernetes cluster, next to your AMD Instinct™ GPUs.** In production use at AMD since April 2026, two PrimusClaw deployments have hosted more than 40,000 agent sessions for more than 70 users: over 100,000 agent-hours, more than 4 million tool calls, and more than 500 billion prompt tokens. On the main deployment, the median run lasted 3.4 hours ([Endnote 2](#endnote-2)).

Agents have moved from loops that run inside a developer's terminal to managed cloud services. Anthropic, AWS, Google, Microsoft, and OpenAI now each run agents for you, hosting either their own harness or yours, along with its sandbox and session state. Organizations that operate their own GPU clusters may also want to run the whole runtime, including the harness and the session log, inside their own cluster. Many agents on such a cluster do their work *on* the GPUs: tuning kernels, optimizing inference servers, and running hours-long experiments. Those agents need the managed-agent experience next to the accelerators they work on.

PrimusClaw is AMD's open answer. This blog describes what a managed agent runtime needs and how PrimusClaw's three planes (session, harness, and sandbox) provide it on Kubernetes. It also describes how PrimusClaw is designed to keep runs going across upgrades and failures, what it has done in production, and how it compares with the managed agent runtimes of the major AI and cloud providers.

## Takeaways

- **A managed agent runtime you operate.** The session store, the agent harness, per-session sandboxes, and the sandbox control plane ship in one open-source repository and run on your own Kubernetes cluster. The repository is MIT-licensed, except for the sandbox control plane, which is an Apache-2.0 fork of the Kubernetes SIG Apps agent-sandbox project. Authentication and multi-node GPU placement are delegated to Primus-SaFE, AMD's open-source cluster platform.
- **Runs not tied to one worker.** Chat turns are dispatched as database rows under leases, not as messages pinned to a process. Harness replicas can be upgraded or lost mid-run, and the run is designed to continue on another replica from a checkpoint. In production, runs have come through lost sandboxes and rolling upgrades of the API: on the main deployment, 2,140 runs lost their sandbox mid-flight, the runtime started rebuilding each sandbox, and 2,047 of those runs (95.7%) succeeded; of 583 runs in flight across three rolling upgrades of the API, 580 (99.5%) ran to completion and 544 (93.3%) succeeded ([Endnote 2](#endnote-2)).
- **Idle sandboxes reclaimed on evidence.** The idle reaper reclaims a sandbox after a probe confirms that the jobs it started through its execute API have exited. An inconclusive probe keeps the sandbox, and idle time alone does not trigger reclamation.
- **A security design for untrusted code.** Agent-generated code runs in a sandbox to which platform-wide credentials are not passed. With the hardening settings in the security guide, the sandbox also filters its outbound traffic and checks that each call carries a short-lived token bound to its session.
- **Standard interfaces.** An API compatible with the Claude Managed Agents quick-start flow, A2A v1.0, and MCP in both directions, with models served through an Anthropic- or OpenAI-compatible gateway (LiteLLM in the reference deployment).
- **GPU-native.** Sandboxes can request AMD Instinct GPUs and start from ROCm™-based images. With Primus-SaFE, multi-node GPU clusters can be handed to the agent. PrimusClaw hosts GPU optimization agents such as Hyperloom and GEAK.

## What a Managed Agent Runtime Needs

A managed agent run is long-lived and stateful. It holds a sandbox with live processes, a workspace of files the agent may have been editing for hours, and a conversation whose prompt cache is worth keeping, and it has to keep going while the platform underneath it is deployed, scaled, and repaired. A managed agent runtime therefore needs runs that are not tied to the worker process that started them, credentials that reach a worker when it takes on a run and not before, sandboxes that are not reclaimed while something in them is still running, workspaces that concurrent runs do not overwrite, and a boundary between the code an agent writes and the platform's own credentials. These needs shaped the design of PrimusClaw.

## PrimusClaw at a Glance

PrimusClaw separates a managed agent run into three parts: a durable **session**, a stateless **harness** that runs the agent loop (PrimusClaw's Brain), and a **sandbox** that executes the agent's actions (its Hands). Figure 1 maps these parts onto PrimusClaw's components and interfaces.

```{figure} ./images/primusclaw-managed-agent-runtime.png
:align: center
:alt: PrimusClaw's session, harness, and sandbox planes running on your own Kubernetes cluster and AMD Instinct GPU nodes.

*Figure 1: The parts of a managed agent runtime, running in your own cluster.*
```

- **API (session plane).** A Fastify service that owns sessions, events, memory, skills, plugins, A2A, MCP, and task DAGs, and enforces cluster-wide admission caps. Each run is a row in PostgreSQL, which is the source of truth. Events stream over SSE with resumable cursors, so a client that reconnects, including the web UI, picks up where it left off.
- **Brain (harness plane).** The agent loop. It calls models through an Anthropic- or OpenAI-compatible gateway, routes tools, runs sub-agents and human-in-the-loop approvals, and manages the run's sandbox. Brain replicas are interchangeable: a run is not pinned to the replica that started it, and a replica with spare capacity whose version supports the run can claim it.
- **Hands (sandbox plane).** An MCP tool server that runs *inside* each session's sandbox and exposes shell, file, and patch tools. Platform-wide credentials are not passed to it.
- **Sandbox control plane.** A fork of Kubernetes SIG Apps' agent-sandbox. Its Workload Manager creates, parks, and destroys per-session sandbox Pods from templates and warm pools, the upstream agent-sandbox controller reconciles them, and its Router forwards each call to the session's sandbox. Inside each Pod, `envd` runs as PID 1 and supervises the commands the agent starts through the sandbox's execute API.

Authentication and GPU placement are delegated to [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE), AMD's open-source platform for AMD Instinct GPU clusters. Production installs require it; an isolated evaluation cluster can run without it.

Figure 2 shows how these pieces talk to each other, and where the trust boundary sits.

```{figure} ./images/primusclaw-architecture.png
:align: center
:alt: PrimusClaw architecture: API, PostgreSQL, NATS, Brain replicas, and the LLM gateway in the credential domain; per-session sandbox Pods beyond the trust boundary.

*Figure 2: PrimusClaw's architecture. Platform credentials live on the left of the trust boundary, and untrusted code runs on the right.*
```

## Dispatch by Doorbell and Lease

PrimusClaw separates "there is work" from "here is the work". The API records each run as a row that contains the run's specification and a sealed credential blob. For chat turns, the API then rings a **doorbell** on NATS JetStream: a message that carries the run's identity but not its prompt or credentials. Doorbell dispatch is enabled per deployment. Other dispatch paths publish the execution request to the work queue.

A Brain replica with spare capacity answers the doorbell by claiming the row through a compare-and-swap on its lease:

- **One claimant at a time.** A claim succeeds when the run's lease is absent or expired, and a competing claim on a live lease is rejected.
- **Credentials are opened at claim time.** The sealed blob is opened when the claim succeeds rather than when the doorbell arrives.
- **Liveness is the lease.** The lease (by default a 45 s TTL renewed every 15 s, [Endnote 1](#endnote-1)) is what proves a run is alive, not the queue's acknowledgment.
- **Saturated replicas leave work for others.** A replica with no free slot leaves the row unclaimed, and replicas with spare capacity also pick up queued rows directly.

PrimusClaw derives its queue and lease timeouts from a shared set of defaults so that they stay consistent with one another.

## How Runs Continue Across Upgrades and Failures

A runtime that hosts hours-long runs cannot treat a deploy as an outage. PrimusClaw upgrades its harness the way Figure 3 shows.

```{figure} ./images/primusclaw-durable-runs.png
:align: center
:alt: New Brain replicas start first, a version marker drains the old ones, in-flight runs checkpoint, and a new replica claims each run and resumes or restarts it, depending on what survived.

*Figure 3: A rolling upgrade of the harness, designed so that an in-flight run continues on a new replica.*
```

1. **New replicas start first.** New Brain replicas come up before any old one stops (`maxUnavailable: 0`).
2. **Old replicas drain, not die.** The rollout publishes the new minimum version to a NATS key-value bucket. Each replica watches that key, and a replica whose version no longer matches stops claiming new runs as soon as the update reaches it, while it keeps executing the runs it holds. The drain is reversible: a rollback simply moves the marker back.
3. **Checkpoint on the way out.** If an old replica must exit, its SIGTERM handler writes the run's conversation state to a checkpoint and syncs the workspace to object storage within the termination grace period (300 s by default, [Endnote 1](#endnote-1)). Checkpoints can be encrypted and bound to the run that wrote them, so that a checkpoint replayed into another run is rejected.
4. **Resume by evidence.** The old replica hands the run back as soon as its checkpoint is written, and a new replica claims it. If the old replica died without handing it back, the claim succeeds once its lease lapses. The new replica then picks the recovery path that the surviving state supports:
   - If the sandbox is still alive, it continues from the checkpointed conversation.
   - If the sandbox was replaced, it restores the workspace snapshot first.
   - If neither survived, it restarts the run cleanly rather than resuming a conversation that refers to files that no longer exist.

The same mechanism is designed to cover unplanned interruptions, such as a node failure, an eviction, or an out-of-memory kill. Brain also checkpoints the conversation after each completed turn. When the lease lapses and another replica claims the run, it resumes from the last completed turn if the sandbox survived, and restarts cleanly if it did not.

In production, runs have come through lost sandboxes and rolling upgrades of the API. On the main deployment, 2,140 runs lost their sandbox mid-flight. The runtime started rebuilding each sandbox, and 2,047 of those runs (95.7%) succeeded. Across three rolling upgrades of the API, 583 runs were in flight: 580 (99.5%) ran to completion and 544 (93.3%) succeeded, and the 36 that failed reported agent or sandbox errors, not a lost worker. From 30 September to 13 October 2026, with at most nine runs executing at once, a queued run on the main deployment was claimed in a median of 52 ms, with a p99 under 1 s ([Endnote 2](#endnote-2)).

## How Idle Sandboxes Are Reclaimed

Sandboxes are the expensive part of a managed agent, especially sandboxes with GPUs, so idle ones must go. But a sandbox's processes are invisible to the HTTP calls that created them. When an agent launches `nohup python train.py &`, the execute API call returns immediately, but the training job continues running long after the request has ended.

PrimusClaw makes background work observable instead of guessing at it:

- **Commands run under a supervisor.** Each command the agent launches through the sandbox's execute API runs under a small job shim that acts as a subreaper. Detached descendants are adopted by the shim rather than by `envd`, so the sandbox can report a roster of the jobs it started, and it reports when that tracking has been lost. Work started outside that path, such as an interactive terminal session, is not part of the roster.
- **Probes have four outcomes, not two.** Brain's keepalive sweep, which any replica can run, reads each sandbox's roster and classifies it:
  - *empty*: there is positive evidence that nothing runs;
  - *non-empty*: user jobs are still running;
  - *unknown*: the probe could not tell, for example after a timeout or a proxy error;
  - *tracking lost*: a shim was killed.

  The idle reaper is designed to act on *empty* results. *Unknown* keeps the sandbox, and *tracking lost* hands it to the hard timeout instead of the idle reaper.
- **Reclaim and reuse share one lock.** Once a sandbox has been quiescent for the idle window (15 minutes by default, [Endnote 1](#endnote-1)), Brain re-probes it synchronously. It must win a compare-and-swap on the sandbox handle from `ready` to `closing` before destroying the sandbox. A run that wants to reuse the sandbox races on the same swap, and the loser backs off.
- **A replaced sandbox is a failure, not an idle one.** Each probe checks the sandbox's identity against the one it expects. If the identity has changed, the sandbox was replaced, and the run reports a failure instead of treating the new, empty roster as idle.

Figure 4 shows the resulting lifecycle.

```{figure} ./images/primusclaw-sandbox-lifecycle.png
:align: center
:alt: Each command runs under a subreaper shim that reports a job roster; a sandbox moves toward reclamation when its roster is confirmed empty.

*Figure 4: The sandbox lifecycle: observe, then reclaim.*
```

Workspaces get the same care. Runs that share a workspace are serialized on that workspace. Optional workspace garbage collection collects a workspace after the runs that reference it have released it and a retention window has passed, and it moves the workspace to a trash directory first. A sync is designed not to overwrite a snapshot with an empty workspace.

## Security Design for Agent-Generated Code

PrimusClaw assumes that whatever runs in the sandbox may be hostile, because it was written by a model that may have read hostile input. The trust boundary in Figure 2 is implemented at several layers:

- **Platform credentials stay with the harness.** The harness keeps the service credentials, object-storage keys, and the internal tokens that authenticate it to the API; these are not passed to the sandbox. Upstream model-provider keys stay inside the LLM gateway. The one credential a sandbox does carry is the calling user's own key, so the design limits injected code to that user's existing scope. In-sandbox code can read and spend that key, so scope, rate-limit, and be ready to revoke each user's key.
- **Tokens bound to a sandbox.** With authentication enabled, the sandbox router signs a token with a five-minute expiry (verified with one minute of clock-skew leeway) for each call ([Endnote 1](#endnote-1)). The sandbox verifies the issuer, the audience, and the expiry, and checks that the token's session matches the session ID that Kubernetes mounted into the Pod. A token issued for one sandbox is designed to be rejected by the others.
- **Egress filtered inside the Pod (when enabled).** A transparent proxy inside each sandbox redirects outbound TCP through SSRF checks. When the proxy runs, it is designed to block cloud metadata and loopback ranges, and the sandbox drops the capabilities that user code could use to bypass the redirect.
- **Fail closed.** The sandbox control plane authenticates callers through Primus-SaFE and refuses to start without authentication unless an operator explicitly opts out. Requests to internal endpoints without the expected credential are designed to be rejected.

It matters which of these you get by default. The fail-closed check on sandbox control-plane authentication is on out of the box. Session-bound tokens, the egress proxy, checkpoint sealing, and the Kata runtime are hardening settings that you turn on. Sandboxes run on `runc` by default, which shares the host kernel and is not a hard isolation boundary. The repository's [security guide](https://github.com/AMD-AGI/PrimusClaw/blob/main/SECURITY.md) lists what to enable before exposing a deployment, and it describes VM-isolated runtimes such as Kata Containers for multi-tenant use.

## Open Interfaces and GPU-Native Agents

A managed agent runtime is only as useful as the agents and clients that can reach it. PrimusClaw speaks the industry's emerging standards:

- **An API compatible with the Claude Managed Agents quick-start flow.** The repository includes a smoke test that runs the quick-start flow against PrimusClaw and uses Anthropic's TypeScript SDK (`@anthropic-ai/sdk`). Point the SDK's base URL at PrimusClaw, then create an agent, an environment, and a session, send a message, and stream the events. This compatibility layer is a thin view over PrimusClaw's own sessions, not a second execution path.
- **A2A v1.0.** PrimusClaw publishes an agent card, and inbound A2A tasks become real sessions with the same execution, persistence, and isolation as any other session.
- **MCP in both directions.** The harness connects to platform MCP servers, and PrimusClaw exposes its own operations as MCP tools so that other agents can create sessions and orchestrate pipelines.
- **Model choice.** Requests go through an Anthropic- or OpenAI-compatible gateway, so a deployment can use self-hosted open models or commercial models. A provider-neutral prompt-cache planner is designed to keep long conversations cache-efficient across checkpoints and redeliveries.

On AMD Instinct GPU clusters, these interfaces carry long-running, GPU-bound workloads. Sandboxes can request AMD Instinct GPUs and start from ROCm-based images, so an agent builds, profiles, and benchmarks against the same ROCm stack it is optimizing. When PrimusClaw runs with Primus-SaFE, multi-node workloads are provisioned as separate distributed jobs and handed to the agent. Optimization agents such as [Hyperloom](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom-optimization/README.html) for inference and [GEAK](https://github.com/AMD-AGI/GEAK) for GPU kernels run as PrimusClaw sessions.

## PrimusClaw in Production

PrimusClaw has been in production use at AMD since April 2026. The figures below come from two deployments between 20 April and 13 October 2026, aggregated from their databases and runtime metrics ([Endnote 2](#endnote-2)); the terms are defined in the [Appendix](#appendix-definitions). Figure 5 summarizes them.

```{figure} ./images/primusclaw-production.png
:align: center
:alt: PrimusClaw production dashboard for April to October 2026: more than 40,000 sessions, over 100,000 agent-hours, more than 4 million tool calls, more than 800 runs executing at peak on the main deployment, more than 500 billion prompt tokens across both deployments, a 95.6 percent prompt-cache hit rate on the main deployment from 5 to 13 October 2026, with the run-length distribution and measured durability results on the main deployment.

*Figure 5: PrimusClaw in production, 20 April to 13 October 2026 ([Endnote 2](#endnote-2)).*
```

Across both deployments, PrimusClaw hosted more than 40,000 sessions and nearly 40,000 runs for more than 70 users. Those runs added up to over 100,000 agent-hours and more than 4 million tool calls, and sent more than 500 billion prompt tokens to models. On the main deployment, 0.29% of tool calls returned an error.

The runs are long. On the main deployment (over 30,000 runs), the median run lasted 3.4 hours, the 90th percentile 7.1 hours, and the 99th percentile 11.5 hours, and the longest ran for 50.1 hours. 71% of runs lasted more than an hour, and over 2,000 lasted more than eight hours. Runs this long are what the lease-based dispatch and evidence-based reclamation described above are designed for.

They also run in parallel. At the June 2026 peak, the main deployment had more than 800 runs executing at once and more than 700 sandboxes running at once. Between 5 and 13 October 2026, 95.6% of the main deployment's prompt tokens were served from the prompt cache.

The workload is the one PrimusClaw was built for. On the main deployment it included more than 2,000 Hyperloom inference-optimization runs and more than 2,000 kernel-optimization runs, alongside sessions that users drive directly through the API and the web UI.

## How PrimusClaw Compares

Major AI and cloud providers now each offer a managed agent runtime, and open-source projects such as kagent bring one to your own Kubernetes. They share the same anatomy, but they differ in where the harness runs, which models it can use, and how they handle updates and recovery. Table 1 compares them with PrimusClaw, based on each project's public documentation, accessed 13 October 2026. Bracketed numbers refer to the References section.

| Runtime | Where it runs | Agent harness | Sandbox | Models | Interfaces | Updates and recovery (as documented) | GPU sandboxes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Managed Agents | Anthropic cloud or Claude Platform on AWS (beta) [2] | Anthropic-hosted, also with self-hosted sandboxes [3] | Anthropic-managed container per session, or a self-hosted sandbox on your infrastructure [3], [4] | Claude [3] | Claude Managed Agents API with SSE events, MCP [2] | Stateless harness; a replacement resumes from the durable session log [1] | Your compute, through self-hosted sandboxes [3] |
| Amazon Bedrock AgentCore Runtime | AWS [5] | Your code, any framework [5] | Micro-VM per session, each micro-VM up to 8 h; or Instances on EC2 in your account, sessions up to 14 days [5], [6], [7] | Any [5] | HTTP, WebSocket, MCP, A2A, AG-UI [5] | Existing sessions keep the previous version until their micro-VM terminates [6] | Instances only: NVIDIA GPU instance families and AWS Inferentia2 [7] |
| Google Agent Runtime (Gemini Enterprise Agent Platform) | Google Cloud [8] | Your code (ADK, LangChain, LangGraph, and others) [8] | Isolated sandbox for code execution [9] | Chosen by your agent code [8] | A2A (preview) [8] | Updating an agent's versioned fields creates an immutable revision; traffic can be split across active revisions (preview) [20] | — |
| Microsoft Foundry hosted agents | Azure [10] | Your container, any framework [10] | Per-session VM-isolated sandbox, up to 2 vCPU and 4 GiB [10] | Foundry model catalog [10] | Responses, Invocations, Activity, A2A v1.0; MCP through a toolbox [10] | One immutable version per endpoint [10]; lease-based recovery after redeployment re-enters the handler (preview) [11] | — |
| OpenAI Agents API | OpenAI cloud (beta API), US data residency [12] | OpenAI-hosted Codex harness, also with self-hosted environments [12], [13] | OpenAI-hosted sandbox, a sandbox provider, or your own environment [12], [13] | OpenAI [12] | Agents API sessions, MCP [12] | Recovery managed by OpenAI [12] | Your compute, through self-hosted or provider sandboxes [13] |
| kagent (CNCF) | Your Kubernetes [15] | Pluggable harness [15] | gVisor by default, or micro-VM [16] | Multiple providers [15] | A2A, MCP, gRPC [15] | Each session keeps the revision it was created from [16] | — |
| **PrimusClaw** | **Your Kubernetes and AMD Instinct GPUs (open source: MIT and Apache-2.0)** | **Your cluster, outside the sandbox; runs can resume on another replica** | **agent-sandbox fork; runc by default, Kata Containers (VM-isolated) as an option** | **Many providers, through an Anthropic- or OpenAI-compatible gateway** | **Claude Managed Agents quick-start flow, A2A v1.0, MCP both ways** | **Harness upgrades are designed to drain old replicas and resume checkpointed runs on new replicas** | **Sandboxes on AMD Instinct GPUs; multi-node handoff with Primus-SaFE** |

**Table 1:** Managed agent runtimes compared, from each project's public documentation, accessed 13 October 2026. A dash means that the cited documentation does not describe the item; it does not mean that the product lacks it. Features of third-party products may change; consult each vendor for current capabilities.

AWS also offers Amazon Bedrock Managed Agents, powered by OpenAI, which runs the OpenAI agent harness and model inference in Amazon Bedrock [14].

Several of PrimusClaw's ideas also appear elsewhere: Anthropic describes a stateless harness that resumes from a durable session log [1], Foundry previews lease-based recovery [11], and AgentCore and kagent keep existing sessions on the version they started with [6], [16]. kagent [15] and OpenHands [17] are also open source, and PrimusClaw's sandbox control plane builds on the same upstream agent-sandbox project [18] that the GKE Agent Sandbox guide for Claude Managed Agents uses [19]. PrimusClaw runs agent sandboxes on AMD Instinct GPUs in production at AMD.

What PrimusClaw brings is the combination:

- **The whole runtime in your cluster.** The event log, the harness, and the sandboxes run there. With self-hosted models behind the gateway, prompts, tool results, and model traffic can stay within your own infrastructure.
- **Harness upgrades designed to move runs to new replicas.** In a harness upgrade, in-flight runs are designed to be drained, checkpointed, and resumed on a new replica according to what survived.
- **GPU sandboxes on your own clusters of AMD Instinct GPUs**, including multi-node handoff through Primus-SaFE.
- **Standard interfaces.** The Claude Managed Agents quick-start flow, A2A v1.0, and MCP in both directions.

The trade-off is ownership. Self-hosting means the compliance certifications and service levels that come with a managed service are yours to provide.

## Getting Started

You can exercise the layer that edits files and runs commands for an agent without a cluster, a GPU, or a model key. The demo needs Node.js 26 and a one-time build; after the build, it runs without network egress:

```bash
git clone https://github.com/AMD-AGI/PrimusClaw.git
cd PrimusClaw
cd claw && npm ci && npm run build && cd ..
make demo
```

To install the full stack on a Kubernetes cluster, follow the [deployment guide](https://github.com/AMD-AGI/PrimusClaw/blob/main/deploy/README.md). The top-level installer previews its plan by default and changes nothing until you pass `--yes`. A production install also needs a Primus-SaFE endpoint for authentication and GPU placement; an isolated evaluation cluster can run without it. Before you expose a deployment, work through the [security guide](https://github.com/AMD-AGI/PrimusClaw/blob/main/SECURITY.md).

## Summary

In this blog you saw how PrimusClaw brings the managed-agent model to infrastructure you own. Runs are rows under leases, so another harness replica can resume them. Harness upgrades are designed to drain and checkpoint in-flight runs. Sandboxes are reclaimed when their own job roster shows them idle. Platform credentials stay with the harness and are not passed to the sandbox. In AMD's production deployments, PrimusClaw has carried runs that last for hours across lost sandboxes and rolling upgrades of the API ([Endnote 2](#endnote-2)), and it hosts GPU optimization agents such as Hyperloom and GEAK on AMD Instinct GPUs.

PrimusClaw is open source under the MIT license, with its sandbox control plane under Apache-2.0, and it is developed in the open: issues and pull requests are welcome in the GitHub repository. Clone it, try the demo, and deploy it next to your GPUs. If you build agents on it or want to collaborate, we would like to hear from you.

## Additional Resources

### Code

- PrimusClaw: [https://github.com/AMD-AGI/PrimusClaw](https://github.com/AMD-AGI/PrimusClaw)
- Primus-SaFE: [https://github.com/AMD-AGI/Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE)
- Hyperloom: [https://github.com/AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom)
- GEAK: [https://github.com/AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK)

### Related Blogs

- Hyperloom: A Multi-Agent Harness for Autonomous Inference Optimization on AMD GPUs: [https://rocm.blogs.amd.com/software-tools-optimization/hyperloom-optimization/README.html](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom-optimization/README.html)
- Stability at Scale: AMD's Full-Stack Platform for Large-Model Training (Primus-SaFE): [https://rocm.blogs.amd.com/software-tools-optimization/primus-SaFE/README.html](https://rocm.blogs.amd.com/software-tools-optimization/primus-SaFE/README.html)

## References

All references accessed 13 October 2026.

1. Anthropic, "Scaling Managed Agents: Decoupling the brain from the hands," [https://www.anthropic.com/engineering/managed-agents](https://www.anthropic.com/engineering/managed-agents)
2. Anthropic, "Claude Managed Agents overview," [https://platform.claude.com/docs/en/managed-agents/overview](https://platform.claude.com/docs/en/managed-agents/overview)
3. Anthropic, "Self-hosted sandboxes," [https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes)
4. Anthropic, "Cloud environment setup," [https://platform.claude.com/docs/en/managed-agents/environments](https://platform.claude.com/docs/en/managed-agents/environments)
5. AWS, "Host agent or tools with Amazon Bedrock AgentCore Runtime," [https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)
6. AWS, "Configure Amazon Bedrock AgentCore lifecycle settings," [https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html)
7. AWS, "Instances," Amazon Bedrock AgentCore Developer Guide, [https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html)
8. Google Cloud, "Agent Runtime," Gemini Enterprise Agent Platform documentation, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime)
9. Google Cloud, "Agent Runtime overview," [https://docs.cloud.google.com/agent-builder/agent-engine/overview](https://docs.cloud.google.com/agent-builder/agent-engine/overview)
10. Microsoft, "Hosted agents in Foundry Agent Service," [https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents)
11. Microsoft, "Resilience for long-running Microsoft Foundry hosted agents (preview)," [https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/long-running-agent-resilience](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/long-running-agent-resilience)
12. OpenAI, "Agents API overview," [https://developers.openai.com/api/docs/guides/agents-api/overview](https://developers.openai.com/api/docs/guides/agents-api/overview)
13. OpenAI, "Self-hosted environments," Agents API, [https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)
14. OpenAI, "Amazon Bedrock Managed Agents," [https://developers.openai.com/api/docs/guides/agents-api/bedrock-managed-agents](https://developers.openai.com/api/docs/guides/agents-api/bedrock-managed-agents)
15. kagent, GitHub repository, [https://github.com/kagent-dev/kagent](https://github.com/kagent-dev/kagent)
16. kagent, "Agent Substrate," kagent 1.x documentation, [https://kagent.dev/docs/kagent/1.x/about/architecture/agent-substrate/](https://kagent.dev/docs/kagent/1.x/about/architecture/agent-substrate/)
17. OpenHands, GitHub repository, [https://github.com/OpenHands/OpenHands](https://github.com/OpenHands/OpenHands)
18. Kubernetes SIG Apps, agent-sandbox, [https://github.com/kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox)
19. Google Cloud, "Claude Managed Agents self-hosted sandboxes on GKE Agent Sandbox," kubernetes-engine-samples, [https://github.com/GoogleCloudPlatform/kubernetes-engine-samples/tree/main/ai-ml/anthropic-agent-sandbox](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples/tree/main/ai-ml/anthropic-agent-sandbox)
20. Google Cloud, "Manage revisions and traffic," Gemini Enterprise Agent Platform documentation, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/runtime/manage-revisions-and-traffic](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/runtime/manage-revisions-and-traffic)

## Endnotes

1. <a id="endnote-1"></a>**PrimusClaw defaults.** Default values are taken from commit `2a77da9d` of the [PrimusClaw repository](https://github.com/AMD-AGI/PrimusClaw): run lease TTL 45 s, renewed every 15 s; Brain `terminationGracePeriodSeconds` 300 s and rolling update `maxUnavailable: 0`; sandbox-router token lifetime 5 min, verified with 1 min of clock-skew leeway; sandbox idle window 900 s. The run lease TTL and renewal interval, Brain's termination grace period and the sandbox idle window can be set per deployment.
2. <a id="endnote-2"></a>**Production data.** Testing conducted by AMD; data collected on 13 October 2026 from two PrimusClaw production deployments operated by AMD on Kubernetes. The main deployment covers 20 April to 13 October 2026; the second covers 3 July to 12 September 2026. Sample sizes: more than 40,000 sessions and nearly 40,000 runs. Software: the PrimusClaw versions deployed during the window. System configuration at data collection (component versions may have differed earlier in the window):
   - *Main deployment:* Kubernetes v1.35.4; nodes running Ubuntu 24.04.4 LTS, Linux kernel 6.17.0, and containerd 2.2.3; PostgreSQL 17.4; NATS 2.12.6; PrimusClaw Brain built from commit `9eb955a6`. Sandbox Pods ran on their node's default OCI runtime: runc, or crun 1.29.1 on AMD Instinct GPU nodes running Ubuntu 24.04.4 LTS and Linux kernel 6.8.0.
   - *Second deployment:* Kubernetes v1.32.5; nodes running Ubuntu 24.04.3 LTS, Linux kernel 6.8.0, and containerd 2.0.5; PostgreSQL 17.0; NATS 2.12.6.

   At data collection, the measured deployments did not use the Kata runtime or the egress proxy.

   The terms used in these figures are defined in the [Appendix: Definitions](#appendix-definitions).

   The cache hit rate comes from runtime metrics; every other figure comes from the deployments' databases (run events, the run ledger, and session records), read from replicas. Results are based on specific test configurations and may vary.

## Appendix: Definitions

<a id="appendix-definitions"></a>

- A *run* is one agent execution, counted once from its completion event, deduplicated per session and message.
- *Users* are the distinct user accounts on the main deployment; this count is a lower bound for both deployments together.
- *Agent-hours* sum each run's harness-recorded execution time, including runs that ended in failure.
- *Tool calls* sum each run's tool-call counter. The 0.29% error rate is from the main deployment (more than 4 million calls).
- *Prompt tokens* sum the input tokens reported per run. Depending on the provider path and the date, this count may exclude cached tokens, so the total is a lower bound on the prompt tokens sent to models.
- *Run length* uses the main deployment's runs with a recorded execution time (more than 30,000 runs), including runs that ended in failure or timeout.
- *Concurrency* is the maximum overlap of run execution intervals (executing runs) and of intervals from a sandbox reaching the running state to the run's completion (running sandboxes), on the main deployment; the peaks fell on 26 and 27 June 2026.
- *Sandbox loss* counts runs on the main deployment whose events show a sandbox rebuild (n=2,140), with the outcome taken from the run's completion event.
- *Runs in flight across an upgrade* are runs whose first event precedes, and whose last event follows, the creation of the new API ReplicaSet in one of three rolling upgrades on 29 June 2026 (n=583 distinct runs). A run with a completion event has run to completion, and that event gives its outcome and any failure reason.
- *Claim latency* is the time from queued to started in the run ledger, on the main deployment from 30 September to 13 October 2026 (n=428), with at most nine runs executing at once.
- *Prompt cache hit rate* is cache reads divided by the sum of uncached input tokens, cache reads, and cache writes, from Brain metrics on the main deployment, 5 to 13 October 2026.
- *Workload mix* counts runs on the main deployment by the client that submitted them.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved

PrimusClaw is not affiliated with, sponsored by, or endorsed by Anthropic, Amazon Web Services, Google, Microsoft, or OpenAI. Anthropic and Claude are trademarks of Anthropic, PBC. Amazon Web Services, AWS, Amazon Bedrock, Amazon EC2, and AWS Inferentia are trademarks of Amazon.com, Inc. or its affiliates. Google, Google Cloud, Gemini, and GKE are trademarks of Google LLC. Microsoft, Azure, and Microsoft Foundry are trademarks of the Microsoft group of companies. NVIDIA is a trademark and/or registered trademark of NVIDIA Corporation in the U.S. and other countries. Kubernetes is a registered trademark of The Linux Foundation. Linux is the registered trademark of Linus Torvalds in the U.S. and other countries. Ubuntu is a registered trademark of Canonical Ltd. PostgreSQL is a trademark of the PostgreSQL Community Association of Canada. Other names are trademarks of their respective owners.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
