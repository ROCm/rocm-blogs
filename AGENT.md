# ROCm Blog Authoring Instructions

## Project

This blog describes our experience porting BLASST to AMD Instinct
MI355X GPUs.

The article should explain:

1. What BLASST is and why sparse attention is useful.
2. How we implemented BLASST on AMD GPUs.
3. Problems encountered while porting and optimizing the implementation.
4. How those problems were diagnosed.
5. The optimizations/fixes that resolved them.
6. Kernel-level and end-to-end performance results.
7. Accuracy/sparsity tradeoffs and practical lessons.

The intended audience is technical readers of ROCm Blogs, including
GPU developers, ML systems developers, and performance engineers.

 
working (top) dir: rocm-blogs/blogs/artificial-intelligence/blasst-amd
final article: <top_dir>/README.md 
article organization notes: <top_dir>/working/blog_matrix.xlsx


---

## Sources of truth

Technical claims must come from one of:

- benchmark results
- profiler results
- source code / PRs
- working notes
- published BLASST material
- cited external sources
- information explicitly supplied by the authors
data lives in <top_dir>/data

Do not invent benchmark numbers, hardware behavior, architectural
explanations, or causes of performance changes.

When evidence is missing, insert a TODO rather than guessing.

Example:

TODO: Verify whether this speedup is primarily due to reduced HBM
traffic or reduced matrix computation.

---

## Measurements vs. interpretation

Clearly distinguish measured behavior from our interpretation.

Good:

"Kernel latency decreased by 9% after the layout change. Profiling
suggests that the improvement comes from..."

Bad:

"The layout change improves performance because..."

unless that causal relationship has been established.

---

## Performance claims

For important performance results, preserve enough information to
identify the experiment:

- GPU
- model
- sequence length
- attention configuration
- sparsity
- baseline
- optimized implementation
- software/framework version where relevant

Do not compare results produced under materially different
configurations without explicitly stating the difference.

---

## Article style

Write concise technical prose suitable for ROCm Blogs.

Prefer:

Problem -> observation -> investigation -> solution -> result

over a chronological diary of development.

Explain GPU behavior concretely where useful.

Avoid marketing language and unsupported superlatives.

Do not unnecessarily remove technical detail that has already been
reviewed by an author.

Preserve existing technical content unless specifically asked to
rewrite or remove it.

---

## Figures

Figures intended for publication belong in the article's images
directory.

Where possible, preserve the raw data and/or script used to generate
performance plots in the working area.

Every performance figure should clearly specify:

- metric
- baseline
- hardware
- relevant workload configuration

Do not fabricate or approximate missing data.
Raw data: <top_dir>/data
Figures: <top_dir>/images
Relevant scripts: <top_dir>/scripts

---

## Working material

Material under data/ working/ scripts/ is research and authoring material, not
necessarily publication-ready.

It may contain:

- raw CSV files
- profiler output
- experiment summaries
- notes from individual PRs
- temporary figures
- TODOs
- colleague contributions

Use this material as evidence for README.md but do not copy working
notes verbatim unless instructed.

---

## Collaboration

Do not substantially rewrite sections explicitly marked as owned by
another contributor unless asked to do so.

Preserve TODO and OWNER markers.

When merging material written by different contributors, normalize
style and terminology without changing technical meaning.

---

## BLASST-specific terminology

BLASST is a dynamic block-sparse attention technique based on
softmax thresholding.

When discussing BLASST behavior, distinguish among:

- target sparsity
- observed sparsity
- threshold/calibration
- skipped attention blocks
- accuracy impact
- kernel speedup
- end-to-end model speedup

These are not interchangeable.

---

## Current platform

Primary target:

AMD Instinct MI350X and MI355X

Software components may include:

- ROCm
- Triton
- AITER
- ATOM
- Gluon
- other explicitly identified components

Do not assume that an optimization or architectural observation
generalizes to NVIDIA GPUs or to other AMD GPU generations unless
there is evidence supporting that claim.

---

## Primary objective

Help the authors produce a technically accurate ROCm Blog article.

Optimize for:

correctness > reproducibility > clarity > brevity

When uncertain, preserve the source material and flag the uncertainty
rather than silently resolving it.