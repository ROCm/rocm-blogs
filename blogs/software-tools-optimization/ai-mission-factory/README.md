---
blogpost: true
blog_title: "AI Mission Factory: Building an AMD GPU Inference Control Plane and Debugging a VRAM Telemetry Bug"
date: "28 Oct 2026"
author: "Siddharth Chandra"
thumbnail: ''
tags: "AI/ML, Memory, Profiling, Optimization, Serving"
category: "Software tools & optimizations"
target_audience: "ROCm developers, GPU infrastructure engineers, MLOps practitioners, AI inference developers, and system administrators working with AMD Instinct GPUs, AMD SMI, vLLM, and GPU monitoring pipelines."
key_value_propositions: "Learn how to build reliable GPU telemetry for AMD Instinct inference systems using AMD SMI and ROCm. Through a real integration issue encountered while developing AI Mission Factory, this blog demonstrates how incorrect VRAM unit conversion can produce misleading memory readings, how to fix the conversion in Python, and how to write regression tests that prevent similar errors. It also introduces practical lessons for integrating GPU monitoring with inference orchestration and observability tools."
language: English
myst:
    html_meta:
        "author": "Siddharth Chandra"
        "description lang=en": "Fix misleading GPU memory dashboards: learn AMD SMI VRAM units, correct Python conversions, and test telemetry in AI Mission Factory."
        "keywords": "AI Mission Factory, AMD Instinct MI300X, ROCm, AMD SMI, VRAM, GPU Monitoring, GPU Telemetry, Python, vLLM, AI Inference, Prometheus, Grafana, MLOps"
        "vertical": "AI, Developers, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Deploying AI at Scale, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Siddharth Chandra"
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

# AI Mission Factory: Building an AMD GPU Inference Control Plane and Debugging a VRAM Telemetry Bug

ROCm Blogs follow a consistent magazine article approach where there is no explicit introduction per se,
but rather each blog starts with a brief, wide-scoped introductory text, without a section title,
before moving into the blog's first section.
The introductory text should include a concise description of your blog: briefly describe for the
reader how they will benefit from the blog, detailing its main deliverables. Please use an active-voice,
call-to-action approach.

## Body

This is where you unleash your creativity. Please follow these general guidelines:

- use actionable, hands-on, conversational approach, guiding your reader through the blog and its content, maintaining engagement. Use active voice, call-to-action (CTA) text (e.g. "Interested in learning more?", "Run this function by using", "Try implementing this yourself")

- keep your writing structured, engaging, and actionable. Divide the blog's content into logical sections.

- Make sure you provide the required background and prerequisites for your blog. Outline any foundational knowledge and tools the reader will likely require.

- When describing a process use step-by-step guide, employ numbered steps or subheadings to guide the reader through the process.

- Integrate examples and use cases: provide real-world applications and scenarios. Reflect on common pitfalls and possible troubleshooting approaches, addressing potential mistakes and solutions.

Leeway into figures, equations, etc.

## Sample markdown

This section covers some markdown techniques commonly used in a blogs.

This is a table.

|      | SPX (MI300X) | CPX (MI300X) |
| ---- | :----------: | :----------: |
| NPS1 |      ✔       |      ✔       |
| NPS4 |              |      ✔       |

Below is a code snippet from the console. You can also use bash, C++, python and other languages.

```console
echo "c 226:128 rwm" > /sys/fs/cgroup/devices/devices.deny #Deny access to device 226:128 in docker (renderD128)

echo "c 226:128 rwm" > /sys/fs/cgroup/devices/devices.allow #Allow access to device 226:128 in docker (renderD128)
```

```{note}
This is how to add a note. See the [myst markdown admonition guide](https://mystmd.org/guide/admonitions) for more details.
```

## Summary

ROCm Blogs follow a consistent magazine-article approach where each blog ends with a "Summary" section.
Please provide a brief summary of your blog, reiterating the main takeaways and deliverables, as well
as what the reader learned from it.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. 
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, [insert all other AMD trademarks used in the material here per AMD Trademarks] and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. [Insert any third party trademark attribution here per AMD's Third Party Trademark List.]
© [Insert year written*] Advanced Micro Devices, Inc. All rights reserved
