---
blogpost: true
blog_title: "Prompt to Playable: World Models, LLM NPCs, and RL on Radeon"
date: "03 Nov 2026"
author: "Mehdi Saeedi"
thumbnail: ''
tags: "Reinforcement Learning, AI/ML, GenAI, LLM"
category: "Applications & models"
target_audience: "Game and simulation developers, AI/ML engineers, and RL researchers who want to run generative world building and AI-driven characters locally on AMD Radeon GPUs. Intermediate level: comfortable with Python, PyTorch, and Docker. No prior game-engine, Gaussian splatting, or ROCm experience needed."
key_value_propositions: "Shows an end-to-end pipeline, running locally on one AMD Radeon GPU with ROCm and permissively licensed open models, that turns a text prompt into a playable 3D level: a camera-controlled video world model, Gaussian splat reconstruction with collision and props, LLM-driven NPCs with voice through vLLM, and a reinforcement-learning agent trained in the same world. All code and Dockerfiles are included to reproduce it."
language: English
myst:
    html_meta:
        "author": "Mehdi Saeedi"
        "description lang=en": "Turn a text prompt into a playable 3D world on one AMD Radeon GPU with ROCm: a video world model, Gaussian splats, LLM-driven NPCs and an RL agent."
        "keywords": "Text to 3D, Video Generation, LLM, NPCs, Game AI"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, AI Inference, Content Creation / Rendering"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Mehdi Saeedi"
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

# Prompt to Playable: World Models, LLM NPCs, and RL on Radeon

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
