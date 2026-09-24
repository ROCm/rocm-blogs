---
blogpost: true
blog_title: "Hyperloom: A Multi-Agent Harness for Autonomous Inference Optimization on AMD GPUs"
date: "21 Sep 2026"
author: "Siliang Chen, Zheng Gong, Xiaofei Zheng, Jiaqiang Liu, Haishuo Kong, Tanya Roosta, Chaojun Hou, Zhenyu Gu"
thumbnail: 'hyperloom-thumbnail.png'
tags: "Serving, System-Tuning, AI/ML, Optimization"
category: "Software tools & optimizations"
target_audience: "ML performance and inference engineers, MLOps and platform teams deploying LLMs on AMD Instinct GPUs, and AI infrastructure decision-makers evaluating ROCm serving performance."
key_value_propositions: "Inference optimization that used to consume weeks of specialist time now runs unattended on AMD Instinct™ GPUs. Hyperloom tunes from the serving stack to the kernels, keeps only end-to-end validated gains, and returned a median 1.73× throughput improvement (1.35×–7.31×) across vLLM, SGLang, and other inference architectures."
language: English
myst:
    html_meta:
        "author": "Siliang Chen, Zheng Gong, Xiaofei Zheng, Jiaqiang Liu, Haishuo Kong, Tanya Roosta, Chaojun Hou, Zhenyu Gu"
        "description lang=en": "Hyperloom is a multi-agent harness that autonomously optimizes LLM inference on AMD Instinct GPUs, reaching a median 1.73x throughput gain."
        "keywords": "Hyperloom, ROCm, End-to-end inference optimization, Multi-agent system, Harness engineering"
        "vertical": "AI, Developers, Systems"
        "amd_category": "Developers"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Siliang Chen, Zheng Gong, Xiaofei Zheng, Jiaqiang Liu, Haishuo Kong, Tanya Roosta, Chaojun Hou, Zhenyu Gu"
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

# Hyperloom: A Multi-Agent Harness for Autonomous Inference Optimization on AMD GPUs

**[ROCm™ Hyperloom](https://github.com/AMD-AGI/Hyperloom) delivered a median 1.73× inference speedup in extensive unattended evaluation on AMD Instinct™ GPUs, with gains ranging from 1.35× to 7.31×.** It profiles each workload, searches framework and kernel optimizations, validates changes end to end, and carries proven results forward, without per-model human tuning.

For teams deploying AI models at scale, serving efficiency directly determines hardware capacity, latency, and operating cost. Yet optimizing a workload across serving configurations, framework code, and GPU kernels has traditionally required weeks or months from a small pool of specialists, and that work must be revisited for every new model, framework release, and accelerator generation. We built Hyperloom to make that full-stack optimization loop autonomous, repeatable, and safe.

In this blog you will follow the whole path Hyperloom takes through a workload, from the serving configuration it starts with down to the GPU kernels underneath, and see how it walks that path unattended on AMD GPUs.

## Takeaways

- Hyperloom is an end-to-end inference optimization system for AMD Instinct™ GPUs, delivering validated throughput gains from 1.35× to 7.31× with no human in the loop.
- The multi-agent harness is built for goal integrity, cumulative optimization, and bounded autonomy.
- Hyperloom supports text generation, image generation, and custom pipelines such as video, across three serving frameworks: vLLM, SGLang, and xDiT.
- More than 14,000 models have been optimized by Hyperloom, which takes the specialist bottleneck out of every deployment.
- Hyperloom is open source under the MIT license, with [documentation](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html) on ROCm Docs.

## Why the Naive Optimization Loop Fails

Inference optimization follows the same basic loop regardless of the model or framework: profile the workload, find what's suboptimal, fix it, measure again. Anyone who's tuned a serving stack has done this by hand, so handing it to an LLM seems like the obvious next step. However, on a real workload, one optimization pass isn't quick. The model will run for hundreds of turns, start and kill dozens of inference server processes, edit a serving framework's source tree, change the serving config, and rewrite GPU kernels. At that length, the naive loop breaks down in three ways, and each failure comes down to the same root cause: the model is left as the only thing keeping track of something the run depends on. The three failures can be summarized as:

- **The model drifts.** Staying coherent over hundreds of turns isn't free. By the time the context window fills up with the run's own transcript, the original goal has quietly turned into something else. The model may even start fabricating benchmark numbers, because nothing in its environment grounds it. By that point, it's no longer optimizing what it was asked to optimize.
- **The model starts from zero every time.** The same model, on the same GPU, under the same framework, might have already been optimized last week, and the LLM has no memory of it. It re-discovers the same environment variables, re-reads the same source code, re-runs the same benchmarks, and hits the same dead ends, paying full price every time. An optimizer that can't accumulate knowledge is just an expensive way to run a fixed script.
- **The model can act unsafely.** Optimizing a serving stack means giving the LLM permission to edit a framework's source tree, and over a long run it can misuse that permission. A patch meant for one file lands somewhere else. A bad edit doesn't get cleanly reverted, and the damage carries into every measurement after it. None of this is malicious: it's just that nothing stops the model from going off track.

None of these problems can be fixed with a better prompt. This is the reason Hyperloom was built as a multi-agent harness that keeps a run aligned with its actual goal, gives it access to what earlier runs already learned, and limits it to actions that are safe to take. This setup ensures AI workloads can be optimized autonomously, for real.

## How Hyperloom Works

At a high level, Hyperloom runs a closed optimization loop: input → optimize → validate → learn. It starts with a model, serving stack, hardware target, workload definition, and optimization budget. It then profiles the workload and searches for improvements across framework settings, source changes, and GPU kernels. Every candidate is re-measured end to end under a controlled benchmark protocol. Changes that fail validation are reverted, while successful changes become the new baseline. Finally, Hyperloom records the validated configuration, implementation artifacts, measured gains, and useful failures in its Recipe KB so future sessions can start from accumulated experience rather than from scratch. Figure 1 shows the whole optimization loop inside Hyperloom.

```{figure} ./images/fig1-harness.png
:align: center

*Figure 1: Inside the optimization loop of Hyperloom.*
```

A session moves through five phases: Prelude, Framework optimization, Kernel optimization, Sweep, and Close. After Sweep, the coordinator either closes the run or begins another cycle. A new cycle starts only when:

- Enough optimization budget remains;
- The run has not converged; and
- Roofline analysis shows remaining performance headroom.

A hard ceiling on the number of cycles sits above all three, so a run cannot loop indefinitely even while the conditions still look favorable. A new cycle resumes at the framework layer rather than starting from scratch, because the baseline and warm start are already established. When any condition fails, Close takes over. The run leaves behind two outputs:

- **Recipe knowledge base (KB):** the validated configuration, patches, kernels, measured gains, and failures worth remembering.
- **Session record:** the final report, event history, and complete audit trail.

### Prelude: Establish the Baseline and Target

Prelude sets both the starting point and the target, in a specific order. First, it measures a baseline on the stock configuration, without using anything the KB suggests. That is the anchor for every later comparison. It then replays the closest matching recipe from the KB to give the run a head start. Finally, it profiles the baseline and builds a roofline analysis, a comparison of measured performance with the GPU's memory-bandwidth and compute limits. This analysis shows how much performance headroom remains, which kernels consume the most execution time, and whether the workload is primarily limited by memory movement or computation. Each direction is tracked against a saturation threshold that later determines what is worth optimizing.

### Framework Optimization: Make It Run, Then Make It Faster

Framework optimization covers how the model is served within its chosen framework. It has two jobs.

The first is enablement, making the model run at all. Hyperloom classifies the launch failure into a specific capability gap, then follows a six-step recovery ladder, from least to most invasive:

1. Read-only diagnosis
2. Serving-flag change
3. In-tree source patch
4. Alternate wheel or source checkout
5. Source localization of a merged pull request
6. Rebuild of a compiled component such as AITER, AMD's library of optimized AI operators, or framework-specific kernel packages such as `sgl-kernel` and vLLM

The second job is to make the workload faster. Hyperloom searches the framework's exposed controls and unreleased improvements, including:

- Serving flags and environment variables
- Weight and key-value (KV) cache precision
- Attention implementation
- Batching and scheduling limits
- Open pull requests, ranked by expected throughput, applied as diffs, and measured

### Kernel Optimization: Improve GPU Execution

Kernel optimization works directly on GPU kernels and delegates the phase to one AMD backend. The shared workflow is:

1. Package the workload, hardware target, and best configuration so far.
2. Launch the selected backend as a child process.
3. Collect the kernels it accepted and the speedup it claims.
4. Re-measure every accepted change end to end under Hyperloom's benchmark protocol. A change is kept only if throughput improves and accuracy holds against the session baseline.

Hyperloom can delegate this work to one of two AMD backends: [GEAK](https://github.com/AMD-AGI/GEAK), which autonomously selects and optimizes kernel targets across the phase, or [KernelForge](https://github.com/AMD-AGI/Hyperloom/tree/main/src/kernelforge), which applies specialized optimization passes to targets selected from the profiling trace. Their scopes differ:

- **GEAK** runs the complete kernel-optimization phase and independently selects the kernels and optimization strategies to explore.
- **KernelForge** runs a sequence of specialized lanes, bounded optimization passes for tuning, fusion, rewriting, or communication collectives. Hyperloom selects candidate kernels from the profiling trace based on GPU time and rewrite feasibility.

Only one backend runs per phase, never both.

### Sweep: Validate Across Operating Points

Everything before this phase was tuned at a single operating point: one concurrency level, one input length, one output length. A configuration that is fastest there is not necessarily fastest at a different request rate or a different sequence length. Sweep therefore re-measures the accumulated stack across a grid of concurrency levels and input/output lengths, and records the best configuration for each point. When the validated gain has not moved since the previous sweep, Sweep skips itself rather than spending GPU time to confirm that nothing changed.

### Close: Preserve Results and Learning

Close must work regardless of how the run ends, through convergence, a time limit, or manual intervention. It determines why the run stopped, inferring a reason from phase history if the run ended too abruptly to record one. It then runs a post-optimization roofline to record how far the tuned stack still sits from the hardware ceiling, finalizes what the run learned into the Recipe KB, builds the final report from the state file and event stream, flags any changes kept without full validation, writes a machine-readable summary for downstream tooling, flushes observability logs, and packages the artifacts.

## Multi-Agent Collaboration for Goal Integrity

Four agent roles run a session as shown in Table 1. The orchestration agent stays alive as one continuous conversation for the whole session, so its plan and hypotheses are always in view instead of being rebuilt from scratch. The other three are spun up only when needed and then discarded: the Critic weighs in on every keep-or-revert decision, the Robustness role gets called when a run looks stuck or stalled, and the Specialist only shows up when something needs to be written.

| Role              | When it runs                                       | What it carries                        | How it keeps the run on-goal                                                                                                                   |
| ----------------- | -------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Orchestration** | Every tick                                         | The whole session, as one conversation | The single continuous planner, so the mission is never re-derived from a cold prompt                                                           |
| **Critic**        | At every keep-or-revert decision                   | None, fresh each time                  | Rules on whether a change actually served the mission, and the record of what the run learned is written from that verdict, not by the planner |
| **Robustness**    | When the run appears stalled, crashed, or circular | None, fresh each time                  | The circuit breaker: it catches the planner going in circles and forces a recovery instead of another lap                                      |
| **Specialist**    | Only where authoring judgment is needed            | Nothing beyond its one task            | Ephemeral by design. With no accumulated context it has nothing to drift with, and it returns a reviewed diff, not a decision                  |

**Table 1:** The four agent roles.

Keeping the long-running orchestration agent oriented requires three mechanisms:

- **State-grounded context.** At every turn, the coordinator rebuilds the mission and progress summary from the state file rather than relying on the transcript. The summary includes the baseline, current best, validated gain, and time remaining.
- **Phase-specific instructions.** Instructions are tagged by phase and backend, then reassembled at each phase change. The agent sees only the tools and actions relevant to the current step.
- **Context refresh.** The coordinator periodically clears the conversation and restarts it from the state file plus a concise summary, preventing accumulated transcript noise from becoming the source of truth.

None of that stops a model from believing something worked when it didn't, so no number the model reports is ever treated as ground truth. Benchmarks are run by the coordinator, not requested by an agent. A gain the model predicts gets logged for calibration but never decides anything on its own, and any speedup a kernel backend claims is held as unverified until Hyperloom re-measures it end to end.

## Self-Evolving Experience for Cumulative Optimization

The Recipe KB is Hyperloom's memory: every session reads from it before starting and writes back to it when it's done. Each row covers one workload, the winning configuration, the throughput it hit, what the run learned, and what didn't work, filed under seven fields: model, hardware, framework, model type, architecture, framework version, and precision. A match on all seven is a direct hit, replayed at full confidence. Most matches aren't exact, so the lookup relaxes one field at a time down a defined fallback order (Figure 2).

```{figure} ./images/fig2-knowledge.png
:align: center

*Figure 2: Knowledge base in Hyperloom.*
```

That's what lets sessions build on each other instead of repeating the same work. By default, Hyperloom applies the closest recipe whose match confidence exceeds a configured threshold. This warm replay, reusing a previously validated configuration as the new session's starting point, allows session N to begin where a similar earlier session ended rather than from vendor defaults, so it can spend its time on ground that has not been covered yet. Everything else in the KB, reverted patches, settings that made things worse, and knobs that tend to help, gets fed back as prompt context rather than encoded as hard rules. A past failure is treated as evidence, not a disqualifier. What the system is strict about is what can enter the KB in the first place, because a bad entry outlives the run that created it.

## Bounded Autonomy for Safe Implementation

The riskiest permission in the system is the ability to edit source code. Every proposed change follows the same bounded flow:

1. Create an isolated Git worktree and branch for the specialist agent.
2. Return the proposed change as a unified diff rather than touching the live source tree.
3. Validate that the diff applies cleanly and reject absolute paths or attempts to escape the directory with `..`.
4. Stash any uncommitted changes in the live checkout.
5. Apply the patch, auto-detecting the strip level and falling back to a three-way merge if needed.
6. Benchmark the result under the coordinator's protocol.
7. Commit only the files touched when the change improves performance; otherwise, reset exactly to HEAD, the last accepted state.

Backports of merged upstream pull requests follow the same path: Hyperloom retrieves the diff and passes it through the same validation gate before execution.

None of this is exotic. It's the normal discipline you'd apply to running untrusted code on real hardware. Every action an agent proposes goes through one policy gate, and the rules that don't bend are about authority: which role can request what, who can write to core session state, how many GPUs a specialist can ask for, and, critically, that no patch gets merged without a Critic's sign-off. That last check happens twice, once at the gate and again inside the executor, because a task that got queued and later resumed doesn't pass through the gate a second time automatically. Moreover, a rejection isn't silent. It comes back as a structured event naming the exact rule that was broken, with a one-line hint, so the next attempt is a fix instead of a repeat.

## From One Session to the Fleet

### Following a Single Session

Take one session end to end: Llama-3.1-8B-Instruct on a single MI355X, running under vLLM, bf16, 1024 tokens in and out, at concurrency 64. It finished at 26,682 tokens per second per GPU, up from a 7,677 baseline, a **3.48×** gain, measured as one number for the whole stack, not a sum of individual wins (Table 2).

| Stage                                                                       | Change applied                                                                 | Throughput | This stage (percentage points) | Cumulative   |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------- | ------------------------------ | ------------ |
| **Baseline**                                                                | N/A                                                                            | 7,677      | N/A                            | N/A          |
| **Warm replay**                                                             | Six AITER env flags and a 768-sequence batch cap, inherited                    | 7,765      | +1.15                          | +1.15%       |
| **Framework optimization**                                                  | AITER unified attention, FP8 KV cache and weights, n-gram speculative decoding | 24,713     | +220.76                        | +221.91%     |
| **Kernel optimization**                                                     | One rewritten cache kernel                                                     | 25,238     | +6.85                          | +228.75%     |
| *↻ Second cycle opens. The first sweep confirms the win and budget remains* |                                                                                |            |                                |              |
| **Framework optimization** (2nd cycle)                                      | RoPE (rotary position embedding) and KV-cache fusion, graph-capture cap        | 26,682     | +18.81                         | +247.57%     |
| **Validated end to end**                                                    | Everything above, measured together                                            | 26,682     |                                | **+247.57%** |

**Table 2:** One session, stage by stage.

Table 2 highlights three important takeaways:

- **Framework changes drove most of the gain in this case.** A new attention backend, FP8 KV cache and weights, and speculative decoding account for most of the improvement here, and the rewritten kernel contributed a smaller additional gain.
- **A second optimization cycle found another win.** The first Sweep confirmed the accumulated improvement, but remaining budget and roofline headroom justified another cycle. That cycle added RoPE/KV-cache fusion and a graph-capture cap, raising throughput to 26,682 tokens per second per GPU.
- **Warm replay enabled later improvements.** Its direct measured contribution was modest, but five of its six inherited flags remained in the final configuration. The large framework gain also depended on an attention backend that warm replay had already enabled.

The run closed with about 5 of its 24 hours unused because it had converged: it ran out of promising ideas to test, not time.

How much kernel optimization contributes depends on the workload. Kernel optimization contributed 29.8 percentage points on Qwen3-14B-FP8, 27.7 on DeepSeek-V4-Pro, and 20.4 on gpt-oss-120b. On DeepSeek-V4-Pro that was more than half of the total validated gain. On isolated kernels, GEAK and Forge both outpaced a Claude Code baseline (Table 3).

| Metric         | GEAK   | Forge  | Claude Code |
| -------------- | ------ | ------ | ----------- |
| Mean speedup   | 2.45×  | 2.97×  | 1.95×       |
| Median speedup | 2.22×  | 3.09×  | 1.73×       |
| Peak speedup   | 23.88× | 24.06× | 14.38×      |

**Table 3:** Isolated kernel speedup.

### Widening to the Fleet

One session shows the loop can close. Whether it holds up across many different workloads, unsupervised, with no per-model babysitting, is the harder test. Table 4 covers sixteen workloads across three serving frameworks, three precisions, dense and MoE models, text and image generation, ranging from under a billion parameters to 862 billion. The serving stack column lists the framework, the precision, and the tensor parallelism (TP) degree.

| Model                      | Type       | Size | Serving stack        | Gain    |
| -------------------------- | ---------- | ---- | -------------------- | ------- |
| DeepSeek-V4-Flash-0731     | text       | 304B | SGLang · bf16 · TP 4 | +631.5% |
| GLM-5.2-MXFP4              | text       | 743B | vLLM · MXFP4 · TP 8  | +390.3% |
| Llama-3.1-8B-Instruct      | text       | 8B   | vLLM · bf16 · TP 1   | +247.6% |
| gpt-oss-120b               | text       | 120B | vLLM · MXFP4 · TP 2  | +172.1% |
| Qwen3-0.6B                 | text       | 752M | vLLM · bf16 · TP 1   | +102.7% |
| Qwen3-8B                   | text       | 8B   | vLLM · bf16 · TP 1   | +97.0%  |
| Qwen3-14B-FP8              | text       | 15B  | vLLM · FP8 · TP 1    | +88.3%  |
| FLUX.1-schnell             | multimodal | 12B  | xDiT · bf16 · TP 1   | +83.0%  |
| gemma-4-26B-A4B-it         | multimodal | 27B  | vLLM · FP8 · TP 2    | +62.3%  |
| Mixtral-8x7B-Instruct-v0.1 | text       | 47B  | SGLang · FP8 · TP 8  | +62.1%  |
| Qwen-Image                 | multimodal | 29B  | xDiT · bf16 · TP 1   | +55.6%  |
| DeepSeek-V4-Pro            | text       | 862B | vLLM · MXFP4 · TP 8  | +53.2%  |
| Z-Image-Turbo              | multimodal | 10B  | xDiT · bf16 · TP 1   | +47.2%  |
| Qwen3.5-397B-A17B-MXFP4    | multimodal | 222B | vLLM · MXFP4 · TP 4  | +46.9%  |
| Nucleus-Image              | multimodal | 17B  | xDiT · bf16 · TP 1   | +41.5%  |
| ERNIE-Image-Turbo          | multimodal | 8B   | xDiT · bf16 · TP 1   | +35.2%  |

**Table 4:** Results from extensive unattended evaluation.

Gains ranged from 7.31× down to 1.35×, with a median of 1.73×. What matters as much as the size of those numbers is that none of them were produced by a person. Every run profiled its own workload, searched the configuration space on its own, applied and re-measured whatever it found, and reported what held up, with no human in the loop from start to finish. Tuning that used to take a scarce specialist weeks now finishes on its own, which turns pointing the harness at a new model into a scheduling decision instead of a staffing one.

## Summary

In this blog you explored how Hyperloom optimizes inference on AMD Instinct GPUs end to end, tuning a workload from the serving stack down to individual GPU kernels with no person in the loop. You saw how the multi-agent harness keeps a long run on goal, lets each session start from what earlier sessions proved, and bounds what an agent is allowed to change. You also followed one session stage by stage, and saw what the same harness produced in extensive unattended evaluation, where speedups ranged from 1.35× to 7.31× with a median of 1.73×.

Hyperloom is open source under the MIT license, and the [documentation](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html) is on ROCm Docs. The next steps below outline where the project is heading, and we will write about that work as it lands. If you hit issues or want to collaborate, we'd like to hear from you.

## Next Steps

- **A lighter harness.** Part of the structure compensates for model weaknesses that better models will outgrow. The rest checks what reasoning cannot settle, like whether the inference actually accelerated. The goal is to get lighter without getting weaker.
- **A trained loop.** Every session re-discovers a search strategy that the accumulated recipe history already describes. That history is training data for a model that could start where the harness currently has to lead it.
- **Findings that flow upstream.** Hyperloom already reads open pull requests and backports merged ones. The natural completion of that path runs the other way, where a fix validated on one workload becomes a patch the whole ecosystem inherits.

## Additional Resources

### Code

- Hyperloom: [https://github.com/AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom)
- GEAK: [https://github.com/AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK)
- KernelForge: [https://github.com/AMD-AGI/Hyperloom/tree/main/src/kernelforge](https://github.com/AMD-AGI/Hyperloom/tree/main/src/kernelforge)

### Documentation

- Hyperloom documentation on ROCm Docs: [https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
