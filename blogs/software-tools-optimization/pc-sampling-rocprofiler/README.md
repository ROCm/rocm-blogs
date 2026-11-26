---
blogpost: true
blog_title: "Beyond Counters: Instruction-Level GPU Hotspot Profiling with PC Sampling in ROCprofiler-SDK"
date: "26 Nov 2026"
author: "Vladimir Indic"
thumbnail: 'pc-sampling-blog-thumbnail.png'
tags: "Profiling"
category: "Software tools & optimizations"
target_audience: "GPU performance engineers and HIP/ROCm kernel developers optimizing workloads on AMD Instinct accelerators; performance/tools engineers already familiar with basic GPU profiling (kernel-trace, PMC counters) who want instruction-level hotspot and stall-reason visibility; and ROCm users evaluating rocprofv3/ROCprofiler-SDK's PC Sampling capability for the first time."
key_value_propositions: "Shows how to use PC Sampling (host-trap and stochastic) in ROCprofiler-SDK to localize GPU kernel hotspots down to the source line and understand why waves stall there - without recompiling or modifying the application. A worked GEMM example walks through the full loop end-to-end: find the hotspot, diagnose the stall reason (e.g., WAITCNT), apply an LDS-tiling fix, and re-run both methods to confirm the diagnosis moved as expected - with every number backed by documented test conditions and reproducible data."
language: English
myst:
    html_meta:
        "author": "Vladimir Indic"
        "description lang=en": "A hands-on guide to PC Sampling in ROCprofiler-SDK: find GPU kernel hotspots, diagnose stalls, and validate a real GEMM optimization."
        "keywords": "PC Sampling, ROCprofiler-SDK, rocprofv3, GPU profiling, host-trap sampling, stochastic sampling, stall reason analysis, instruction-level profiling, AMD Instinct MI300A, ROCm, GEMM optimization, LDS tiling, AMD CDNA"
        "vertical": "AI, HPC, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Vladimir Indic"
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

# Beyond Counters: Instruction-Level GPU Hotspot Profiling with PC Sampling in ROCprofiler-SDK

Optimizing GPU workloads often starts with understanding what resources are being used — compute units, memory bandwidth, cache behavior, stalls, occupancy, and other hardware counters. While these metrics are essential, they do not always answer the most important optimization question: where in my code is the performance cost coming from?

This is where PC Sampling becomes especially valuable.

PC Sampling helps bridge the gap between high-level performance counters and the actual instructions executing on the GPU. By periodically capturing the program counter of running waves and aggregating thousands of samples, PC Sampling builds an instruction-level view of execution hotspots. Instead of only knowing that a workload is memory-bound, stalled, or inefficient, developers can identify the specific regions of GPU code where time is being spent, where waves are stalling, and which instructions may deserve closer inspection.

This makes PC Sampling a powerful tool for developers working on performance-critical GPU applications, libraries, and frameworks. It can help answer questions such as:

- Which GPU instructions are consuming the most execution time?
- Where are the dominant hotspots inside a kernel?
- Where are stalls concentrated — and around which instruction types or code regions?
- Which parts of a precompiled binary should be investigated first?
- Which optimizations are worth pursuing — without requiring source changes or recompilation?

Another key benefit is that PC Sampling is designed to work with already-precompiled binaries. Users do not need to modify the application or instrument kernels manually. In many cases, they can point the profiler at an existing workload and collect meaningful hotspot information directly, making PC Sampling a practical and low-friction profiling capability for real-world applications.

ROCprofiler-SDK, part of AMD ROCm™ software (version 6.4 and later), provides two complementary PC Sampling methods that target the same optimization problem from different directions:

- **Software-based PC Sampling (host-trap)** — A software-driven method where a background host thread periodically traps running waves to capture program counter information. This method is time-driven from the host and is available on a broad set of AMD CDNA GPUs, starting with AMD Instinct™ MI200 series accelerators.
- **Hardware-based PC Sampling (stochastic)** — A hardware-assisted method where dedicated GPU hardware samples waves directly on a cycle-based schedule. In addition to program counter information, this method can provide richer per-sample execution state, such as instruction type, stall reason, and arbiter state. It is available on AMD CDNA™ 3, AMD CDNA 4, and AMD CDNA 5 architectures (CDNA 3+ GPUs), including the AMD Instinct MI300 series, MI325 series, and MI350 series accelerators, as well as the AMD Instinct MI455X.

This post's scope is the AMD CDNA 3 and AMD CDNA 4 architectures (gfx942/gfx950). The same PC sampling support extends to AMD CDNA 5 (including the AMD Instinct MI455X), but the examples we examine in this post do not cover it.

This blog post walks through both methods using the rocprofv3 command-line interface, with a worked GEMM example that shows how to move from hotspot identification to stall-reason diagnosis and back to optimized code.

## Key Takeaways

- PC sampling gives instruction- and source-line-level visibility into GPU hotspots on precompiled binaries — no recompilation or manual instrumentation required.
- Two complementary methods: host-trap (software-based) finds *where* time is spent; stochastic (hardware-based) explains *why* waves stall there (`WAITCNT`, barrier waits, issue arbitration, etc.).
- In the worked GEMM example, PC sampling helped narrow the naive kernel down to a fetch-bound pattern on uncoalesced global loads — guiding the investigation toward an LDS-tiling fix, without reading disassembly by hand.
- On this post's test system (single AMD Instinct MI300A, ROCm `rocm-10.1.0`), the LDS-tiled rewrite measured ~3.0× faster, with the hot line's WAITCNT stall count down ~12.2× — both numbers are backed by the reproducible test conditions and data described in this post, not quoted as general hardware claims.
- PC sampling complements kernel-trace, performance monitoring counters (PMC), and Advanced Thread Trace (ATT) rather than replacing them — pick the method that answers the question you actually have.

## What Is PC Sampling?

PC sampling is a statistical profiling technique that works by:

1. **Periodically sampling** the program counter (PC) of executing waves on the GPU
2. **Recording** which instruction each wave was executing at the sample point
3. **Correlating** these samples with code objects and source locations
4. **Aggregating** thousands of samples to build a statistical hotspot profile

Unlike instrumentation-based profiling that modifies your kernel code, PC sampling is **non-intrusive**, making it well suited for production workloads and long-running applications.

### Key Benefits

- **Non-intrusive**: no kernel modification or recompile required.
- **No kernel serialization required**: kernels keep running concurrently, so the sampled execution behaves like the real run, not a serialized debug trace.
- **Application-level triage**: profile a long, complex application end-to-end and let the hotspot distribution direct your attention to the kernels (and the lines inside them) that actually cost the most — no need to instrument or analyze each kernel in isolation.
- **Instruction-level**: hotspots resolve to single instructions.
- **Statistical**: thousands of cheap samples instead of one expensive trace.
- **Production-friendly**: same binary, same kernel, same launch parameters.

### Instruction-Level vs. Source-Level Attribution

Because PC sampling works on any precompiled binary, the sampled PC is *always* decoded down to the assembly instruction — the `Instruction` column in the CSV (e.g. `s_waitcnt vmcnt(0)`, `v_fma_f32 v5, v3, v4, v5`). No special build is needed for this.

Mapping that PC back to a **source line** is what requires the binary to be built with debug symbols. With `-g`, the `Instruction_Comment` column is populated with `file.cpp:line`; without it, the column is empty and you are left reading assembly. For the rest of this post we focus on hotspots and stalls attributed back to source lines, so all examples assume the application was built with `-g`:

```bash
hipcc -g your_kernel.cpp -o your_app
```

Samples that did not resolve to a source line (e.g. those landing inside HIP runtime headers shipped without debug info) are reported as a dropped count and excluded from the aggregations below.

## Method 1: Software-Based PC Sampling (Host-Trap)

### How Host-Trap Sampling Works

Host-trap PC sampling is driven by a kernel-space thread initiated by the amdgpu driver. This thread runs on the host CPU and periodically fires a trap signal that is broadcast to every CU, targeting a specific wave slot; if a wave is resident in that slot, it is instructed to trap — i.e., to pause, capture its state, and write that state to a buffer in GPU device (global) memory. Each CU/wave-slot handles the trap independently.

```text
Time ─────────────────────────────────────────────────>
       ↓          ↓          ↓          ↓
      [Sample]  [Sample]  [Sample]  [Sample]

Every N microseconds, the host fires a trap that is broadcast to all
CUs at a specific wave slot. For each CU where that slot is occupied
  1. The resident wave pauses at its next safe point.
  2. The wave captures its state (PC, exec_mask, dispatch ID, wave
     coordinates) into a memory buffer.
  3. The wave resumes execution.
```

The interval is specified in **microseconds**. Because the trap rides on host-side scheduling and on the wave's response latency, the actual sample cadence is not strictly periodic — treat the requested interval as a *target* sample rate, not a guarantee. The state captured at the trap point is also subject to a 1–2 instruction "sampling skid": the reported PC can be one or two instructions past the costly one. When reading host-trap output, consider the neighborhood of a hotspot, not just the single line. See the upstream docs section [Host-trap PC sampling and arbitrary sampling skid](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html#host-trap-pc-sampling-and-arbitrary-sampling-skid) for the authoritative description.

### Configuring Host-Trap Sampling

Host-trap is configured from the command line with `--pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval <microseconds>`, gated behind the `--pc-sampling-beta-enabled` flag. Before configuring, always run the capability check (`rocprofv3-avail info --pc-sampling`) to confirm the GPU advertises host-trap support and to see the min/max interval the driver currently accepts.

### Sample Data

Each host-trap sample carries the PC (encoded as a code-object ID plus offset), the exec_mask of active SIMD lanes, the sample timestamp, the originating kernel dispatch ID, an async correlation ID that links back to the API call, and the wave's coordinates within the workgroup. What you actually see in CSV is a tighter projection of those fields — shown next.

### Example: Profiling a Kernel with Host-Trap

```bash
# One-time capability check
rocprofv3-avail info --pc-sampling
# Host-trap at 1 ms (1000 us) target interval
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method host_trap \
          --pc-sampling-unit time \
          --pc-sampling-interval 1000 \
          --output-format csv \
          -- ./your_application
```

Run `rocprofv3-avail info --pc-sampling` once per machine to confirm the GPU + ROCm + driver combination supports PC sampling; if the output is empty, PC sampling is unavailable on this configuration. On a gfx942 (AMD Instinct MI300 series) the output for a single agent looks like:

```text
GPU                    :       0
Name                   :       gfx942
configs                :
   Method              :        host_trap
   Unit                :        time
   Min_Interval        :        1
   Max_Interval        :        18446744073709551615
   Flags               :        none

   Method              :        stochastic
   Unit                :        cycle
   Min_Interval        :        256
   Max_Interval        :        1048576
   Flags               :        interval pow2
```

(Multi-GPU systems print one such block per agent.) Each `configs` entry is one method the driver advertises for that agent, with the interval window it will currently accept and any constraints in `Flags` — `pow2` here means stochastic intervals must be powers of two. Pick a `--pc-sampling-method`, `--pc-sampling-unit`, and `--pc-sampling-interval` consistent with one of the listed entries.

The host-trap `rocprofv3 --pc-sampling-beta-enabled …` invocation shown above produces (at minimum) two files in the working directory, each prefixed by the process ID:

- `<pid>_pc_sampling_host_trap.csv` — one row per sample.
- `<pid>_agent_info.csv` — GPU agent metadata for correlation.

The CSV has six columns:

```text
"Sample_Timestamp","Exec_Mask","Dispatch_Id","Instruction","Instruction_Comment","Correlation_Id"
54155306462675,65535,1,"s_waitcnt lgkmcnt(0)","/opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:275",1
54155306462715,65535,1,"s_waitcnt vmcnt(0)","/opt/rocm-6.4.0/share/hip/samples/2_Cookbook/0_MatrixTranspose/MatrixTranspose.cpp:44",1
```

- `Sample_Timestamp` — GPU timestamp at the moment of sampling.
- `Exec_Mask` — active SIMD lanes for the sampled wave. The blog scopes to CDNA 3 / CDNA 4 wave64, so the full mask is `0xFFFFFFFFFFFFFFFF` (all 64 lanes); use `popcount` to read occupancy.
- `Dispatch_Id` — which kernel dispatch this sample belongs to.
- `Instruction` — the disassembled instruction at the sampled PC.
- `Instruction_Comment` — the source-line mapping if the binary was built with debug symbols; empty otherwise.
- `Correlation_Id` — joins back to the launching API call.

JSON output (`--output-format json`) adds richer per-sample fields useful for advanced post-processing (full `hw_id` breakdown indicating where a wave was running at the moment of sampling, workgroup coordinates, etc.); CSV is the working format for hotspot triage. Advanced JSON-driven analysis is out of scope for this introductory blog.

### Aggregating Samples by Source Line

The CSV's `Instruction_Comment` column already encodes `path/to/file.cpp:line`. A short pandas snippet is enough to turn the per-sample rows into a Top-N table of the hottest source lines:

```python
import pandas as pd

df = pd.read_csv("<pid>_pc_sampling_host_trap.csv")

# Drop samples without a source mapping (binary not built with -g, or
# samples that landed inside the HIP runtime headers).
total = len(df)
df = df[df["Instruction_Comment"].str.strip().astype(bool)].copy()
print(f"Dropped {total - len(df)} / {total} samples without source mapping")

# Split "path/to/file.cpp:123" -> file + line.
src = df["Instruction_Comment"].str.rsplit(":", n=1, expand=True)
df["Source_File"] = src[0].str.rsplit("/", n=1).str[-1]
df["Source_Line"] = pd.to_numeric(src[1], errors="coerce").astype("Int64")

# Top-5 hot source lines.
hot = (df.groupby(["Source_File", "Source_Line"])
         .size()
         .sort_values(ascending=False)
         .head(5)
         .rename("Samples"))
hot_pct = (hot / len(df) * 100).round(1).rename("Pct")
print(pd.concat([hot, hot_pct], axis=1))
```

Expected output shape:

| Source_File | Source_Line | Samples | Pct  |
|-------------|------------:|--------:|-----:|
| example.cpp |          75 |    7421 | 72.3 |
| example.cpp |          73 |    1305 | 12.7 |
| example.cpp |          77 |     612 |  6.0 |
| …           |           … |       … |    … |

This tells you *where* the kernel spent its time. It does not tell you *why* those lines are hot — for that, switch to stochastic sampling (Method 2).

### When to Use Software-Based (Host-Trap) Sampling

**Best fit:**

- Time-driven sampling that does not depend on wave issue activity.
- Long-running applications where coarse resolution is acceptable.
- Very low sampling resolution (intervals in the milliseconds/seconds/minutes or higher) for overhead-sensitive workloads.
- Initial hotspot identification on a GPU that does not support stochastic sampling (e.g., AMD Instinct MI200 series accelerators).

**Limitations:**

- No microarchitectural state: you see *what* PC the wave was at, not *why* it stalled.
- Sample cadence is approximate; the requested interval is a target, not a guarantee.
- Lower fidelity due to the 1–2 instruction sampling skid: reported PCs can sit one or two instructions past the costly one. Read the neighborhood of a hotspot, not just the line. See [Host-trap PC sampling and arbitrary sampling skid](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html#host-trap-pc-sampling-and-arbitrary-sampling-skid).

## Method 2: Hardware-Based PC Sampling (Stochastic)

### How Stochastic Sampling Works

Hardware sampling logic lives on each Compute Unit (CU). Every ~N GPU cycles, the hardware picks one wave currently resident on the CU, snapshots it, and traps that wave to read the snapshot state and write it to a buffer in GPU device (global) memory. There is no host involvement; each CU does this independently.

```text
  CU
 ┌──────────────────────────────────────────────────┐
 │  wave 0  wave 1  wave 2  ...  wave k             │
 │     │       │       │           │                │
 │     ▼       ▼       ▼           ▼                │
 │  ┌────────────────────────────────────┐          │
 │  │   issue arbiter / pipelines        │          │
 │  └────────────────────────────────────┘          │
 │              │                                   │
 │       every ~N cycles                            │
 │              ▼                                   │
 │         [Sample] ── pick one resident wave       │
 └──────────────────────────────────────────────────┘
```

For the sampled wave the hardware captures:

1. The PC and exec_mask.
2. Whether the sampled wave issued an instruction that cycle (`Wave_Issued_Instruction`); if it did, the instruction type (`Instruction_Type`); if it did not, the reason it could not issue (`Stall_Reason`) — all three defined in the CSV schema that follows. All three fields are scoped to the sampled wave, not to any other wave on the CU.
3. The number of waves resident on the sampled CU at the sampling cycle (`Wave_Count`).
4. A full per-pipeline arbiter snapshot for the sampled cycle — which pipelines had a ready candidate, which one won arbitration, and which lost. CSV omits this for simplicity; the JSON output exposes it as `arb_state_issue_*` / `arb_state_stall_*` fields.

Because the per-sample selection happens in hardware and does not round-trip through a host kernel thread, hardware-based sampling can sustain higher sampling rates than the software-based method.

The sampling interval is specified in **GPU clock cycles** and must be a power of two in `[2^8, 2^20]` (256 to 1,048,576 cycles). 2^20 is the hardware ceiling; smaller intervals sample more aggressively, which both increases the volume of captured data and perturbs kernel execution more (each sampled wave still has to save its state into the buffer, and that cost is paid inside the kernel). Move to the next larger power of two if the generated data becomes overwhelming or the perturbation noticeably slows execution; use smaller intervals on small workloads where data volume is not a concern but you need enough samples to be statistically meaningful. As a rough rule of thumb, halving the interval (e.g., 2^20 → 2^19) approximately doubles the sample count for the same kernel run time.

### Configuring Stochastic Sampling

Stochastic sampling is configured from the command line with `--pc-sampling-method stochastic --pc-sampling-unit cycles --pc-sampling-interval <power-of-two cycle count>`, again gated behind `--pc-sampling-beta-enabled`. A reasonable starting point on gfx942 is `--pc-sampling-interval 1048576` (2^20 cycles); lower it (next smaller power of two) when a short workload is not producing enough samples to be useful.

### What a Stochastic Sample Carries

A stochastic sample carries everything a host-trap sample does plus four extra fields: whether the wave actually issued an instruction that cycle, what type of instruction it was, the stall reason if it did not issue, and the number of waves resident on the CU at the sampling cycle.

The CSV reflects exactly those additions:

```text
"Sample_Timestamp","Exec_Mask","Dispatch_Id","Instruction","Instruction_Comment","Correlation_Id","Wave_Issued_Instruction","Instruction_Type","Stall_Reason","Wave_Count"
390705261841337,18446744073709551615,24,"s_waitcnt vmcnt(0)","vector-ops.cpp:133",24,0,"ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST","ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",4
390705261924637,18446744073709551615,29,"v_max_i32_e32 v1, v2, v0","vector-ops.cpp:77",29,1,"ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU","ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",6
```

Reading the first row: the wave did *not* issue an instruction (`Wave_Issued_Instruction = 0`); the slot held an `s_waitcnt vmcnt(0)`; the reason it could not issue was `WAITCNT` — a memory dependency not yet resolved. The second row issued normally: a VALU `v_max_i32_e32`.

Stall reasons reported by the hardware on CDNA 3 / CDNA 4 (gfx942/gfx950) are `NO_INSTRUCTION_AVAILABLE`, `ALU_DEPENDENCY`, `WAITCNT`, `INTERNAL_INSTRUCTION`, `BARRIER_WAIT`, `ARBITER_NOT_WIN`, `ARBITER_WIN_EX_STALL`, `SLEEP_WAIT`, and `OTHER_WAIT` — see the definitions below. Other information like per-pipeline arbiter state (`arb_state_issue_*` / `arb_state_stall_*`) is present in the JSON output but is not surfaced in CSV; reach for JSON when you need it.

- **NO_INSTRUCTION_AVAILABLE** — the sampled wave's instruction was not in the instruction-buffer (e.g., at the target of a branch, I$ miss, etc.).
- **ALU_DEPENDENCY** — the sampled wave's instruction could not be issued due to a hardware-internal dependency.
- **WAITCNT** — the sampled wave was at a memory-dependency wait (`waitcnt`).
- **INTERNAL_INSTRUCTION** — the sampled wave was issuing an internal instruction (e.g., a no-op).
- **BARRIER_WAIT** — the sampled wave was at a barrier, waiting for other waves in the workgroup to reach the barrier.
- **ARBITER_NOT_WIN** — the sampled wave was not selected to issue.
- **ARBITER_WIN_EX_STALL** — the sampled wave issued, but was stalled due to execution pipe backpressure or resource stall.
- **SLEEP_WAIT** — the sampled wave was in the sleep state.
- **OTHER_WAIT** — catch-all bucket for stalls that do not match any of the categories above.

### Example: Analyzing Stall Patterns

```bash
rocprofv3-avail info --pc-sampling

rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method stochastic \
          --pc-sampling-unit cycles \
          --pc-sampling-interval 1048576 \
          --output-format csv \
          -- ./your_application
```

A small pandas snippet extends the host-trap aggregation with two extra pieces: it splits each hot source line into issued vs. not-issued samples, and for each hot line it breaks down *why* the waves did not issue. (The raw `Stall_Reason` strings are long enums prefixed `ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*`; the snippet strips that prefix to keep the output readable.)

```python
import pandas as pd

df = pd.read_csv("<pid>_pc_sampling_stochastic.csv")

# Drop samples without source mapping (see host-trap section).
total = len(df)
df = df[df["Instruction_Comment"].str.strip().astype(bool)].copy()
print(f"Dropped {total - len(df)} / {total} samples without source mapping")

src = df["Instruction_Comment"].str.rsplit(":", n=1, expand=True)
df["Source_File"] = src[0].str.rsplit("/", n=1).str[-1]
df["Source_Line"] = pd.to_numeric(src[1], errors="coerce").astype("Int64")

# Step 1: top-5 hot source lines, with issued vs. not-issued split.
# Samples_Pct uses the source-mapped total (len(df) after the filter above)
# as its denominator; samples without source mapping cannot be attributed
# to a line, so excluding them keeps the percentages honest.
by_line = df.groupby(["Source_File", "Source_Line"])
hot = (by_line.size().sort_values(ascending=False).head(5)
              .rename("Samples"))
issued = by_line["Wave_Issued_Instruction"].agg(
    Issued=lambda s: int((s == 1).sum()),
    Not_Issued=lambda s: int((s == 0).sum()),
)
top5 = (hot.to_frame().join(issued).assign(
            Samples_Pct=lambda d: (d["Samples"] / len(df) * 100).round(1),
            Stall_Pct=lambda d:   (d["Not_Issued"] / d["Samples"] * 100).round(1),
        )[["Samples", "Samples_Pct", "Issued", "Not_Issued", "Stall_Pct"]])
print("Top-5 hot source lines:")
print(top5)

# Step 2: for each top-5 line, break down WHY waves did not issue.
# Each line is normalized to its own 100% (per-line denominator), so the
# percentages compare stall-reason mix within a line, not across lines.
# Strip the long enum prefix for readability.
PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
print("\nStall reasons per hot line (non-issue samples only, each line = 100%):")
for (src_file, src_line) in top5.index:
    stalled = df[(df["Source_File"] == src_file)
                 & (df["Source_Line"] == src_line)
                 & (df["Wave_Issued_Instruction"] == 0)]
    if stalled.empty:
        continue
    reasons = (stalled["Stall_Reason"].str.replace(PREFIX, "", regex=False)
                       .value_counts()
                       .rename("Samples"))
    print(f"\n  {src_file}:{src_line}")
    print(reasons.to_string().rstrip())
```

Expected output shape:

Top-5 hot source lines:

| Source_File | Source_Line | Samples | Samples_Pct | Issued | Not_Issued | Stall_Pct |
|-------------|------------:|--------:|------------:|-------:|-----------:|----------:|
| example.cpp |          75 |   29840 |        58.2 |   9120 |      20720 |      69.4 |
| …           |           … |       … |           … |      … |          … |         … |

*`Samples_Pct`: percentage of source-mapped samples attributed to this line (samples without source mapping are excluded from the denominator); `Stall_Pct`: fraction of this line's samples where the wave did not issue.*

Stall reasons per hot line (non-issue samples only, each line = 100%):

`example.cpp:75`

| Stall_Reason   | Samples |
|----------------|--------:|
| WAITCNT        |   18203 |
| ALU_DEPENDENCY |    1842 |
| OTHER_WAIT     |     675 |
| …              |       … |

See the GEMM walk-through below for a worked example of how to read these tables in practice.

### When to Use Hardware-Based (Stochastic) Sampling

**Best fit:**

- Stall-reason analysis (`WAITCNT`, `ALU_DEPENDENCY`, `BARRIER_WAIT`, `ARBITER_NOT_WIN`, `ARBITER_WIN_EX_STALL`, `OTHER_WAIT`, …).
- Instruction-mix and pipeline-pressure investigation.
- Memory-bottleneck triage (look for `WAITCNT`-dominated stall mixes).
- CDNA 3 / CDNA 4 / CDNA 5 GPUs (AMD Instinct MI300 series, MI325 series, MI350 series, and MI455X accelerators).
- High fidelity: no sampling skid — the captured PC is the wave's actual PC at the sampled cycle.

**Limitations:**

- Requires CDNA 3, CDNA 4, or CDNA 5. Not supported on AMD Instinct MI200 series accelerators or older. Not productized on RDNA.
- Very short kernels (single-digit microseconds) collect few or no stochastic samples — the cycle-based interval simply doesn't fire often enough relative to the kernel lifetime. For workloads like this, consider [Advanced Thread Trace (ATT)](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html) instead, which records every executed instruction on a limited subset of CUs.
- Richer per-sample state (arbiter / pipeline bits, memory counters) is JSON-only; CSV gives you the four core extra columns.
- Output data size grows quickly on large workloads — long-running kernels at aggressive intervals can produce multi-gigabyte CSVs. Raise the interval, profile a narrower window, or split the run across multiple invocations.

## Real-World Use Case: Diagnosing a GEMM Kernel

Let's apply both methods end-to-end to a naive global-memory GEMM and see what PC sampling tells us. The kernel is a textbook `C = A × B` with one SIMD-thread per output element:

```cpp
__global__ void
matrix_multiply(float* A, float* B, float* Out, int m, int n, int k)
{
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int gid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (gid_x < n && gid_y < m)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)                      // line 74
        {
            sum += A[gid_y * k + i] * B[i * n + gid_x];  // line 76
        }
        Out[gid_y * n + gid_x] = sum;                    // line 78
    }
}
```

### Test Environment

All performance and sample-count numbers in this section come from measured runs of the `mm.cpp` / `mm-lds.cpp` examples below, on the following setup:

- **Hardware**: a system with 4× AMD Instinct MI300A accelerators; this example targets a single MI300A (`hipSetDevice(0)`, `gfx942`).
- **Software**: ROCm `rocm-10.1.0`, OS `Ubuntu 24.04.4 LTS`.
- **Build**: `hipcc -g -O3 mm.cpp -o mm` (and the equivalent for `mm-lds.cpp`).

Each binary performs one untimed warmup launch followed by a single `std::chrono`-timed launch (`hipDeviceSynchronize()` on both ends) — no averaging across multiple dispatches.

Run with M = N = K = 8192, this gives a kernel-only wall time of **357.1 ms** per dispatch for the naive kernel. That is the baseline number to beat.

### Step 1: Where Does Time Go? (Host-Trap)

```bash
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method host_trap \
          --pc-sampling-unit time \
          --pc-sampling-interval 1000 \
          --output-format csv \
          -- ./mm
```

Because the binary performs an untimed warmup launch before the timed one, the raw CSV contains two `Dispatch_Id` values. We keep only the later (timed) dispatch before aggregating — extending the [Aggregating samples by source line](#aggregating-samples-by-source-line) snippet with one line prepended (`df = df[df["Dispatch_Id"] == df["Dispatch_Id"].max()]`) is enough. On the resulting, warmup-filtered CSV (86,239 total samples, all source-mapped):

| Source location | Samples | Pct |
| --- | ---: | ---: |
| `mm.cpp:76` | 83,091 | 96.3 |
| `mm.cpp:74` | 3,128 | 3.6 |
| `mm.cpp:80` | 14 | 0.0 |
| `mm.cpp:68` | 5 | 0.0 |

96% of all samples land on the inner-loop body `sum += A[...] * B[...]` (line 76), and the remaining ~4% sit on the loop header right above it (line 74). That is a single hotspot, but host-trap doesn't tell us *why* the wave was there — only that it spent essentially all its time on those two lines.

### Step 2: Why Is It Stuck There? (Stochastic)

Same binary, switch to stochastic to get the issue/no-issue split and the stall reasons:

```bash
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method stochastic \
          --pc-sampling-unit cycles \
          --pc-sampling-interval 1048576 \
          --output-format csv \
          -- ./mm
```

Running the stochastic aggregation snippet — extended with the same `Dispatch_Id` warmup filter as Step 1 — on the resulting CSV (167,891 total samples after filtering):

Top-5 hot source lines:

| Source location | Samples | Samples_Pct | Issued | Not_Issued | Stall_Pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mm.cpp:76` | 162,533 | 96.8 | 5,091 | 157,442 | 96.9 |
| `mm.cpp:74` | 5,346 | 3.2 | 5,111 | 235 | 4.4 |
| `mm.cpp:68` | 7 | 0.0 | 3 | 4 | 57.1 |
| `mm.cpp:78` | 4 | 0.0 | 1 | 3 | 75.0 |
| `mm.cpp:71` | 1 | 0.0 | 0 | 1 | 100.0 |

*`Samples_Pct`: percentage of source-mapped samples attributed to this line (samples without source mapping are excluded from the denominator); `Stall_Pct`: fraction of this line's samples where the wave did not issue.*

Stall reasons per hot line (non-issue samples only, each line = 100%):

| Line | Top stall reasons (% of this line's non-issue samples) |
| --- | --- |
| `mm.cpp:76` | WAITCNT 93.1, NO_INSTRUCTION_AVAILABLE 4.6, ALU_DEPENDENCY 1.3, ARBITER_WIN_EX_STALL 0.8, ARBITER_NOT_WIN 0.2 |
| `mm.cpp:74` | ARBITER_NOT_WIN 84.3, ARBITER_WIN_EX_STALL 15.7 |

The table above is from the stochastic run. The `Issued`, `Not_Issued`, and `Stall_Pct` columns are derived from `Wave_Issued_Instruction` and `Stall_Reason` — fields that only stochastic sampling provides; a host-trap CSV has no equivalent and cannot produce this breakdown. Those two stochastic-only columns tell the whole story:

- On the hot line (`mm.cpp:76`), waves issued an instruction in only **3.1%** of samples — they were stalled 96.9% of the time.
- Of those stalls, **93.1% are `WAITCNT`** — the wave is parked on `s_waitcnt vmcnt(...)` waiting for an outstanding vector-memory load to come back.

That is the classic signature of a memory-bound kernel: the math is cheap, the loads are not, and the wave spends almost all of its life waiting on global memory.

### Step 3: Read the Code with the Diagnosis in Hand

Now the kernel itself becomes legible. Each iteration of the inner loop issues two global loads — `A[gid_y * k + i]` and `B[i * n + gid_x]` — and a single FMA (fused multiply-add: `sum = sum + a * b` issued as a single instruction). Across a workgroup the `B` access is the killer: `gid_x = blockIdx.x * blockDim.x + threadIdx.x` varies along a row, but the load is `B[i * n + gid_x]` with `n = 8192`, so two adjacent rows of the workgroup land 8,192 elements (32 KB) apart. There is no reuse of `A` or `B` across threads, and there is no staging into LDS — every FMA pays the full global-memory round trip.

PC sampling pinpointed the *line* and the *reason*; the diagnosis is "fetch-bound on uncoalesced global loads of `B`, no data reuse." A natural optimization to try is a tiled variant that stages `A` and `B` tiles into LDS so each loaded value is reused by an entire row/column of the tile. The next section re-runs both passes against that variant and checks whether the WAITCNT share on the hot line drops — exactly the kind of before/after comparison PC sampling is built for.

### Step 4: Validating the Optimization with an LDS-Tiled Variant

The tiled kernel loads a `TileSize × TileSize` (here, 16×16) block of `A` and `B` into LDS once per outer iteration, synchronizes, and then has each SIMD-thread do `TileSize` FMAs against the staged data. Every value loaded from global memory is now reused by 16 SIMD-threads instead of 1.

```cpp
__global__ void
matrix_multiply_tile(float* A, float* B, float* Out, int m, int n, int k)
{
    __shared__ float subTileM[TileSize][TileSize];
    __shared__ float subTileN[TileSize][TileSize];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TileSize + ty;
    int col = bx * TileSize + tx;

    float sum = 0;
    for (int i = 0; i < ((k - 1) / TileSize + 1); i++)
    {
        int curr_l = row * k + i * TileSize + tx;
        int curr_r = (i * TileSize + ty) * n + col;

        if (i * TileSize + tx < k && row < m)
            subTileM[ty][tx] = A[curr_l];
        else
            subTileM[ty][tx] = 0.0;

        if (i * TileSize + ty < k && col < n)
            subTileN[ty][tx] = B[curr_r];
        else
            subTileN[ty][tx] = 0.0;

        __syncthreads();                                   // line 103

        for (int j = 0; j < TileSize; j++)
        {
            if (j + TileSize * i < k)                      // line 107
                sum += subTileM[ty][j] * subTileN[j][tx];  // line 109
        }

        __syncthreads();                                   // line 113
    }

    if (row < m && col < n)
        Out[row * n + col] = sum;
}
```

Built with `hipcc -g -O3 mm-lds.cpp -o mm-lds` and timed the same way:

| Variant | Wall time per launch |
| --- | ---: |
| `mm` (naive) | 357.1 ms |
| `mm-lds` (16×16 LDS tile) | 118.0 ms |

A ~3× speedup is a good sign that the diagnosis was right, but PC sampling tells us *why* — and what is left on the table.

**Host-trap on `mm-lds`** (26,628 source-mapped samples after the same warmup-dispatch filter, same `--pc-sampling-interval 1000`):

A first corroborating signal before we even look at the table: host-trap fires one sample per wave per fixed wall-clock interval, so the total sample count is roughly proportional to the kernel's wall time. Going from 86,239 samples (naive) to 26,628 (tiled) at the same 1000 µs interval — a ~3.2× drop — tracks the ~3.0× wall-time speedup measured above. The act of *sampling itself* confirms the kernel got faster, before any per-line interpretation.

| Source location | Samples | Pct |
| --- | ---: | ---: |
| `mm-lds.cpp:109` | 11,745 | 44.1 |
| `mm-lds.cpp:107` | 2,734 | 10.3 |
| `mm-lds.cpp:103` | 924 | 3.5 |
| `mm-lds.cpp:85` | 582 | 2.2 |
| `mm-lds.cpp:113` | 525 | 2.0 |

The single 96% hotspot from the naive kernel is gone. Time is now split across the compute (`sum += ... * ...` at :109), the guard around it (`if (j + TileSize * i < k)` at :107), the first `__syncthreads()` where the staged tile loads get drained (:103), the `A`-tile load guard (:85), and the second `__syncthreads()` (:113).

**Stochastic on `mm-lds`** (45,308 source-mapped samples after the warmup-dispatch filter, `--pc-sampling-interval 1048576`):

The stochastic sample count drops in the same ballpark: 167,891 → 45,308, a ~3.7× reduction. The mechanism is different from host-trap — stochastic fires one sample per fixed number of *active wave-cycles*, not per wall-clock interval — so this is corroborating evidence on a different axis: the kernel is also spending ~3.7× fewer wave-cycles in flight, not just less wall time. Two independent measurements agreeing on the same direction is a good sanity check that the optimization is real and not a timing artifact.

Top-5 hot source lines:

| Source location | Samples | Samples_Pct | Issued | Not_Issued | Stall_Pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mm-lds.cpp:109` | 20,897 | 46.1 | 3,099 | 17,798 | 85.2 |
| `mm-lds.cpp:107` | 4,186 | 9.2 | 2,987 | 1,199 | 28.6 |
| `mm-lds.cpp:103` | 1,569 | 3.5 | 53 | 1,516 | 96.6 |
| `mm-lds.cpp:85` | 1,106 | 2.4 | 516 | 590 | 53.3 |
| `mm-lds.cpp:113` | 868 | 1.9 | 63 | 805 | 92.7 |

*`Samples_Pct` is share of all source-mapped samples; `Stall_Pct` is `Not_Issued / Samples` for this line. The top-5 here cover ~63% of source-mapped samples; the remaining ~37% are spread across a longer tail than in the naive kernel — expected, since time that used to be concentrated on one global-memory wait is now spread across compute, a guard, and two synchronization points.*

Stall reasons per hot line (non-issue samples only, each line = 100%):

| Line | Top stall reasons (% of this line's non-issue samples) |
| --- | --- |
| `mm-lds.cpp:109` | WAITCNT 67.5, ARBITER_NOT_WIN 14.3, NO_INSTRUCTION_AVAILABLE 10.1, ARBITER_WIN_EX_STALL 8.1 |
| `mm-lds.cpp:107` | ALU_DEPENDENCY 77.5, ARBITER_NOT_WIN 22.5 |
| `mm-lds.cpp:103` | WAITCNT 51.6, BARRIER_WAIT 47.3, ARBITER_NOT_WIN 1.1 |
| `mm-lds.cpp:85` | ALU_DEPENDENCY 56.6, NO_INSTRUCTION_AVAILABLE 32.5, ARBITER_NOT_WIN 10.9 |
| `mm-lds.cpp:113` | BARRIER_WAIT 99.8 |

Three things to read out of this:

- **WAITCNT pressure on the hot compute line collapsed — but the share-of-stalls comparison understates it.** This comparison is scoped to these two specific hot lines, not a kernel-wide WAITCNT count. Reading the raw percentages, WAITCNT's share of non-issue cycles dropped from 93.1% on `mm.cpp:76` to 67.5% on `mm-lds.cpp:109`, which sounds like a modest improvement. The denominators are not the same, though: 93.1% of 157,442 non-issue samples (`mm.cpp:76`) is 146,507 WAITCNT samples, while 67.5% of 17,798 non-issue samples (`mm-lds.cpp:109`) is 12,008 — about **12.2× fewer WAITCNT samples in absolute terms**, on top of the total stochastic sample count itself being ~3.7× smaller. So the share dropping by ~26 points is the small part of the story; the absolute drop is the real one. The loads that feed the FMA now come from LDS, not global memory, so the wave spends a fraction of its previous life parked on `s_waitcnt`. The remaining ~32% of non-issue cycles on `:109` are spread across arbiter contention (`ARBITER_NOT_WIN` + `ARBITER_WIN_EX_STALL` ≈ 22%) and `NO_INSTRUCTION_AVAILABLE` (~10%) — classic signatures of a kernel that is closer to compute/issue-bound. The lesson generalizes: when comparing stall percentages between two runs, multiply by the per-line non-issue count before drawing conclusions — a smaller percentage of a much larger pile can still be the bigger problem.
- **The remaining global-memory wait moved to the first synchronization point** (`mm-lds.cpp:103`, the `__syncthreads()` right after the tile loads): 96.6% stall rate, split almost evenly between `WAITCNT` (51.6%, the drain of the just-issued `A`/`B` loads) and `BARRIER_WAIT` (47.3%, the barrier itself). That is exactly where you want the memory wait to land — the cost of the global load is paid once per tile, then amortized across 16 FMAs, and it shows up right where the tile is staged rather than spread across the inner loop. The second `__syncthreads()` (`:113`) shows 99.8% `BARRIER_WAIT` with essentially no `WAITCNT`, confirming the next-tier bottleneck there is the synchronization point itself, not more memory traffic.
- **The inner-loop guard (`:107`) and the `A`-tile guard (`:85`) are issue-bound, not memory-bound.** Both are pure control-flow lines — `if (j + TileSize * i < k)` and `if (i * TileSize + tx < k && row < m)` — with no load of their own, and neither shows meaningful `WAITCNT`. Instead they're dominated by three distinct issue-side reasons: `ALU_DEPENDENCY` (the instruction could not be issued due to a hardware-internal dependency), `NO_INSTRUCTION_AVAILABLE` (the instruction wasn't in the instruction buffer yet — both lines are conditional (`if`) guards, which can compile down to branch instructions, so this reads as an instruction-fetch stall around a branch or its target, not a memory wait), and `ARBITER_NOT_WIN` (the wave was ready to issue but lost that cycle's arbitration to another wave on the CU). That is the same "closer to compute/issue-bound" signature as the tail of `:109`'s stalls — once the global-memory wait is confined to the staging barrier, the rest of the loop is contending for instruction fetch and issue slots, not memory bandwidth. This is also a good illustration of why reading a single WAITCNT percentage out of context is risky: the same kernel can have one line where WAITCNT is the dominant signal and several others, a few source lines away, where it barely registers.

This is the loop PC sampling is meant to support: host-trap finds the hotspot, stochastic explains it, you apply the fix, and you re-run both to confirm the diagnosis moved (or shrank) the way you expected.

## Choosing the Right Profiling Method

ROCprofiler-SDK offers several complementary profiling methods that answer different questions about GPU execution. No single method covers the full diagnostic surface — the most effective workflow is to start broad and then zoom in. The table below maps each method to what it tells you, what it cannot tell you, and when to reach for it.

| Profiling method | What it tells you | What it does NOT tell you | When to reach for it |
| --- | --- | --- | --- |
| **Kernel-trace** (`--kernel-trace`) | Timeline of all kernel dispatches: start time, duration, overlap, order — GPU activity initiated from the CPU side | Why a kernel was slow; what happened inside it at the source level | First pass — tells you which kernels to investigate and how they overlap |
| **PMC counters** (`--pmc`) | Aggregate GPU activity for a whole kernel dispatch (cache miss rates, occupancy, memory bandwidth utilization) | Where inside the kernel the cost is; no source-line attribution | After kernel-trace, to characterize a kernel's bottleneck class (e.g., "high L2-miss rate suggests a memory-access pattern problem") |
| **PC sampling** (host-trap + stochastic) | Statistical hotspot distribution across source lines for all executing waves; stall reasons at each hotspot | Exact cycle-accurate instruction timeline; per-wave execution order; rare events too infrequent to appear in a statistical sample | Can be used directly for fast end-to-end application triage, or after PMC when you want to localize a known bottleneck to a source line and understand *why* |
| **ATT** (`--att`) | Cycle-accurate record of every instruction executed by every wave on a subset of CUs | Chip-wide coverage — by design it only instruments a handful of CUs, not every CU running the kernel | Last resort — when PC sampling has exhausted all stall-reason leads and you need cycle-accurate wave-level timing or need to catch rare, non-representative events |

### How the Methods Relate

**Kernel-trace** monitors GPU activity as initiated from the CPU: it records when each dispatch was submitted, when it started executing on the GPU, and when it completed. This gives you a timeline — which kernels overlapped, how long each ran, and which one to focus on. It does not tell you anything about what happened *inside* the kernel.

**PMC counters** measure aggregate GPU or kernel-level activity at the kernel boundary: metrics like L2 cache miss rate, occupancy, or memory bandwidth utilization. This tells you the *class* of bottleneck (e.g., "this kernel is memory-bound") but provides no source attribution — you learn that something is wrong at the whole-kernel level, not which line of code is responsible.

**PC sampling** gives you a statistical approximation of where the GPU application or a specific kernel spends its execution cycles, and — with stochastic sampling — *why* waves stall at each hotspot. It works across all kernels in a run without requiring you to pick a target upfront, so it can serve as a first-pass triage tool that surfaces the top-N hotspots and guides you toward kernels worth investigating further. It does not provide timeline information, nor exact counts of specific microarchitectural events (e.g., precise L2 cache miss counts).

**ATT** records every executed instruction on a small subset of CUs, providing a cycle-accurate wave-level timeline. This covers what PC sampling cannot: the precise order of wave execution, the interleaving of waves on a CU, and rare events that a statistical sampler would miss. Because ATT is scoped to a few CUs by design, it complements rather than replaces PC sampling — it is the tool you reach for when stall-reason analysis is insufficient and you need to understand the cycle-by-cycle behavior of individual waves.

### A Typical Investigation Path

For most optimization workflows, a practical sequence looks like this:

1. **Kernel-trace** — get the dispatch timeline; identify the slow kernel.
2. **PMC counters** — characterize the bottleneck class for the target kernel.
3. **PC sampling** — localize to a source line and understand the stall reason. (Can also be used as step 1 when you want fast statistical triage of the whole application without deciding upfront which kernel to focus on.)
4. **ATT** — reach for this only when PC sampling stall-reason analysis is exhausted and you need cycle-accurate wave-level detail or need to catch rare execution events.

For most optimization loops, kernel-trace + PC sampling (or PC sampling standalone for fast end-to-end triage) covers the full cycle: find the slow kernel, find the hotspot, understand the stall, apply the fix, re-run to confirm. PMC and ATT are the methods you reach for when you need to go deeper in a specific direction.

## Best Practices and Tips

### 1. Choosing Sampling Intervals

**Host-trap (time-based, microseconds; range `[1 µs, arbitrarily large]`):**

- Long-running apps / continuous monitoring: 10 ms – 1 s, and freely larger (multiple seconds, minutes, or more) when you only need very coarse coverage of an overnight or all-day run.
- Long kernels: 1 – 10 ms.
- Medium kernels: 100 µs – 1 ms.
- Short kernels: 1 – 100 µs (and even sub-microsecond intervals are accepted at the API level, though very short kernels may still produce too few samples to be useful — see Method 2).

**Stochastic (cycle-based, power-of-two; range `[2^8, 2^20]`):**

- Default starting point: 2^20 = 1,048,576 cycles (the hardware ceiling). Values must stay powers of two, so adjust by halving (more samples, finer resolution) or doubling (fewer samples, coarser resolution) — never by an arbitrary amount.
- **Practical lower bound: 2^16 = 65,536 cycles.** 2^8 = 256 is the minimum the hardware will accept, but we discourage intervals below 65,536 cycles in practice — the data volume and per-sample perturbation grow quickly and rarely buy useful additional resolution.
- **Very short kernels (under ~5 µs) may not produce enough samples even at the smallest interval.** The cycle-based clock simply does not tick often enough relative to the kernel's lifetime; consider Advanced Thread Trace (ATT) for kernels in that range instead.

### 2. Sample Size Considerations

- Minimum: 1,000 samples for basic hotspots
- Recommended: 10,000+ samples for statistical significance
- Detailed analysis: 100,000+ samples

### 3. Combining with Other Profiling Methods

PC sampling complements other rocprofv3 profiling methods. The examples below sketch a typical multi-phase investigation that mixes them — PMC counters for aggregate metrics, host-trap for hotspot identification, stochastic for stall-reason deep-dive, and kernel-trace for timeline context.

```bash
# Phase 1: PMC counters for aggregate metrics
rocprofv3 --pmc SQ_WAVES MemUnitBusy -- ./my_app

# Phase 2: Software-based (host-trap) for hotspot identification
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method host_trap \
          --pc-sampling-unit time \
          --pc-sampling-interval 1000 \
          --output-format csv \
          -- ./my_app

# Phase 3: Hardware-based (stochastic) for microarchitectural deep-dive
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-method stochastic \
          --pc-sampling-unit cycles \
          --pc-sampling-interval 1048576 \
          --output-format csv \
          -- ./my_app

# Phase 4: Kernel-dispatch trace for timeline analysis
rocprofv3 --kernel-trace -- ./my_app
```

`--kernel-trace` can be combined with either PC sampling or PMC in a single invocation, which is useful when you want a timeline alongside the hotspot or counter data. PMC and PC sampling, however, cannot run together — the implementation does not allow both to be configured on the same agent at the same time, so phases 2/3 and the PMC pass in phase 1 must be separate runs.

### 4. Multi-Process Considerations

A GPU agent can host at most one active PC sampling configuration at a time, across all processes. If a second process tries to configure a *different* method or interval, the configuration request will be rejected; if it requests the same one already in use, both processes receive samples for their own kernels. In practice, `rocprofv3` surfaces this as an error at startup — re-run with parameters that match the active configuration (or wait for the other process to finish).

### 5. Interpreting `Exec_Mask`

The execution mask shows which SIMD lanes were active:

```python
def analyze_divergence_cdna_wave64(samples):
    """Per-sample lane occupancy for CDNA 3 / CDNA 4 wave64 kernels."""
    active_lanes = samples["Exec_Mask"].apply(lambda x: bin(int(x)).count("1"))
    print(f"Mean active lanes per sample: {active_lanes.mean():.1f} / 64")
    print(f"Std: {active_lanes.std():.1f}")
```

The blog scopes to CDNA 3 / CDNA 4 wave64 throughout, so the divisor is 64. A mean well below 64 points at branch divergence; how "well below" matters depends on the kernel — judge against an expected baseline, not a fixed cutoff.

## Troubleshooting Common Issues

### `rocprofv3-avail info --pc-sampling` Returns Nothing

The driver does not advertise PC sampling support on any GPU on this machine. The usual causes:

- The GPU does not support PC sampling at all (e.g., MI100 or older, discrete RDNA parts).
- The amdgpu kernel driver predates PC sampling support.
- ROCm is older than 6.4.

Do not fall back to grepping `rocminfo` or `dmesg` — they will not tell you whether the rocprofv3 stack can actually configure PC sampling. The `rocprofv3-avail info --pc-sampling` check is the only authoritative answer.

### Zero Samples Collected

The most common cause is that a kernel is too short for the interval. If a kernel completes in fewer cycles (or less time) than your `--pc-sampling-interval`, the sampling mechanism (the hardware sampler for stochastic, the kernel-space background thread for host-trap) may never fire. Lower the interval, or increase the workload size.

### Unsupported Configuration

If rocprofv3 rejects your invocation as an unsupported configuration even though `rocprofv3-avail info --pc-sampling` listed *something* for the target GPU, the argument combination you passed does not match any of the advertised entries. Re-read the `rocprofv3-avail` output and make sure `--pc-sampling-method`, `--pc-sampling-unit`, and `--pc-sampling-interval` line up exactly with one of the listed configs (including the `pow2` constraint for stochastic intervals). Common slip-ups:

- **Wrong unit.** `--pc-sampling-unit time` expects microseconds; `--pc-sampling-unit cycles` expects GPU cycles. Mixing them silently produces nonsensical sampling rates.
- **Forgot `--pc-sampling-beta-enabled`.** Without the beta flag rocprofv3 explicitly rejects the other `--pc-sampling-*` arguments.

### Huge Output Files or Visible Application Slowdown

PC sampling writes one record per sample, so an aggressive interval on a long-running workload can fill the disk with multi-gigabyte CSVs (or JSON several times larger) and noticeably slow the application down — each sampled wave still pays the cost of saving its state to the buffer. If you see either symptom:

- **Raise the sampling interval.** For stochastic, double it (next power of two). For host-trap, multiply by 2–10×. A coarser interval gives up resolution but cuts data volume and per-sample perturbation proportionally.
- **Shorten the run.** PC sampling on a long-running training job produces enormous outputs; profile a narrow window (a few hundred iterations) rather than the full run.
- **Prefer CSV over JSON for triage.** CSV is compact and is the right working format for hotspot identification; reach for JSON only when you actually need the richer per-sample fields it exposes.

## Summary

In this post, you learned how PC Sampling in ROCprofiler-SDK closes the gap between high-level GPU counters and the actual instructions running on the hardware. Instead of stopping at "this kernel is memory-bound," you can now point `rocprofv3` at an existing, precompiled binary and get instruction- and source-line-level answers to *where* a kernel spends its time and *why* waves stall there — no recompilation or manual instrumentation required.

You saw the two methods work together rather than compete: software-based host-trap sampling for broad, low-friction hotspot identification across a wide range of CDNA GPUs, and hardware-based stochastic sampling for stall-reason detail plus a per-sample instruction-type breakdown that host-trap can't provide, available on CDNA 3 / CDNA 4 / CDNA 5. The worked GEMM example put that combination into practice end-to-end. Host-trap flagged a single fetch-bound hotspot, and stochastic sampling traced it to a memory-dependency stall on uncoalesced global loads. An LDS-tiling rewrite turned that diagnosis into a measured ~3.0× speedup. Re-running both methods on the new kernel confirmed that the memory-dependency stalls on the hot line had dropped by roughly 12.2× — a full loop from symptom to fix to verification, backed by reproducible numbers rather than general claims.

You also saw where PC Sampling fits alongside kernel-trace, PMC counters, and Advanced Thread Trace: each answers a different question, and PC Sampling is often a fast, direct path from "something is slow" to "this line, for this reason."

Under the hood, both methods are driven by the ROCprofiler-SDK C API (`rocprofiler_configure_pc_sampling_service`, `rocprofiler_query_pc_sampling_agent_configurations`), which is what `rocprofv3` and other third-party tools build on top of. See the [SDK reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/pc_sampling.html) if you want to integrate PC sampling directly into your own tooling.

Ready to put this to work? Grab a kernel you already suspect is underperforming, build it with `-g`, and run it through both host-trap and stochastic sampling with the commands in this post. Let the hotspot and stall-reason tables point you at your next optimization — and feel free to share what you find.

## Additional Resources

- **Using PC sampling with rocprofv3**: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html>
- **CDNA 3 / CDNA 4 PC sampling deep dive**: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/cdna3-cdna4-pc-sampling.html>
- **ROCprofiler-SDK API reference**: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/pc_sampling.html>
- **Advanced Thread Trace (ATT)**: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html>

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD ROCm, AMD Instinct, AMD CDNA, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
