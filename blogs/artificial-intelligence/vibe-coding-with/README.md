---
blogpost: true
blog_title: "Vibe Coding Pac-Man Inspired Game Generation with DeepSeek-R1 and AMD Instinct MI300X"
date: 17 Jul 2025
author: 'Charles Yang, Mahdi Ghodsi, George Wang'
thumbnail: 'pac.jpg'
tags: AI/ML, GenAI, LLM
category: Applications & models
target_audience: AI Developers
key_value_propositions: Hands on guide to show developers that you can create fun games using AMD Instinct GPUs
language: English
myst:
    html_meta:
        "author": "Charles Yang, Mahdi Ghodsi, George Wang"
        "description lang=en": "Learn LLM-powered game dev using DeepSeek-R1 on AMD MI300X GPUs with iterative prompting, procedural generation, and VS Code AI tools"
        "keywords": "vibe coding, game generation"
        "vertical": "Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Charles Yang, Mahdi Ghodsi, George Wang"
---

# Vibe Coding Pac-Man Inspired Game with DeepSeek-R1 and AMD Instinct MI300X

AI systems have been constrained by their narrow capabilities and limited contextual understanding. Modern large language models (LLMs), such as GPT-4, Claude, DeepSeek, and CodeLlama, are different from previous approaches to AI. LLMs leverage vast datasets and incorporate natural language and code repositories. This enables them to understand natural language syntax, semantics, and programming logic in multiple programming languages (Python, JavaScript, C++, etc.)

In this blog, we are going to discuss how to create simple and advanced games, like Pac-Man shown below in figure 1, using LLMs served by SGLang on AMD Instinct™ MI300X GPUs. We'll explore techniques, best practices, and reveal key tips and tricks to achieve your game development goals. This approach is particularly empowering for developers aiming to build larger, more complex projects, even without extensive professional experience or background knowledge. By harnessing an LLM-powered AI code assistant for robust program scaffolding and AMD computing to speed up development cycles you are able to enhance production capabilities and gain significant competitive edge.

```{figure} ./images/figure.1.gif
:align: center
:alt: Pac-Man Game Demonstration
Figure 1. Pac-Man Inspired Game Demonstration
```

## Overview

Utilizing AMD MI300X GPUs with ROCm ™ software is ideal for running large-scale language models like DeepSeek R1, when inference workloads are exceptionally demanding—processing up to hundreds of billions of tokens daily in production environments. With 192 GB of unified HBM3 memory per GPU, MI300X can host the entire full-size DeepSeekR1 671B model on a single node, whereas competing products require multiple nodes to achieve the same. AMD ROCm  also supports multiple serving frameworks like vLLM, SGLang, Ollama, and HuggingFace TGI for Deepseek-R1 providing developers multiple options to choose from. Combined with high-performance frameworks like SGLang, the MI300X excels at long-context tasks such as Vibe-coding, which demand sustained attention across tens of thousands of tokens. With support for FP8 and BF16 computation and full compatibility with the ROCm software stack, it delivers excellent efficiency and scalability. When a single GPU is available, we can use the much smaller DeepSeek distilled R1 Qwen 32B model instead.  Official SGLang Docker images with ROCm support are regularly published on Docker Hub, allowing users to easily run the latest versions.

Looking ahead, the upcoming MI355 is expected to push these capabilities even further, offering greater performance and memory bandwidth—reinforcing next-generation LLM deployment.  

## How to Set up Development Environment

Before diving into the details of how we generated a version of the Pac-Man inspired game, we will first briefly describe our development environment setup. We used the [MS AI Toolkit](https://code.visualstudio.com/docs/intelligentapps/overview) extension in VS Code (see figure 2) as our code generation tool. This tool requires you to add an LLM for code generation. For the LLM endpoint, we leveraged SGLang running on AMD MI300x GPUs to serve DeepSeekR1. SGLang is one of the leading LLM serving frameworks for optimizing LLM inference. In the LLM code assistant example, we run the full-size DeepSeek-R1 671B model on 8 GPUs. When a single GPU is available, we use the much smaller DeepSeek distilled R1 Qwen 32B model instead. To learn more about setting up your code development assistant using the VS Code Toolkit and SGLang, check out [this tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/deepseekr1_sglang.html%5D/). You can also learn more about DeepSeek-R1 in this blog.

```{figure} ./images/figure2.png
:align: center
:alt: VS Code AI Code Generator Extension
Figure 2. IDE Workspace
```

## How to Create Games

### Simple Game Generation – Snake Game

Using LLMs for game source code generation, such as a classic [Snake game](https://snakegame.org/), as seen in figure 3, involves a structured, iterative process that blends precise prompting with debugging and refinement. Here’s a step-by-step approach:

1. Initial Prompting:
   Craft a detailed prompt by specifying the programming language, framework (e.g., Python + Pygame), and core features: snake movement, food spawning, collision detection, score tracking, and game-over logic. Example:
   “Generate a Python Snake game using Pygame. Include keyboard-controlled snake movement, food spawning, collision detection with walls/self, score tracking, and a game-over screen.”
2. Code Generation & Testing:
   The LLM will produce a code draft. Copy this into an IDE, install dependencies (e.g., Pygame), and run it. Initial versions often have bugs—like incorrect event handling or rendering issues. For instance, the snake might not grow when eating food due to a flawed segment-appending logic.
3. Iterative Debugging:
   Feed errors back to the LLM for fixes. Example:
   “The snake doesn’t grow after eating food. Here’s the current eat food function…”
   The LLM may adjust the snake’s body-tracking list or collision-checking logic.
4. Feature Expansion:
   Once you get the basic version of the game working, request enhancements: difficulty levels, sound effects, or a high-score system. Break these into smaller prompts:
   “Add a high-score system that saves to a file.”
5. Code Explanation & Optimization:
   Ask the LLM to explain complex sections (e.g., the game loop structure) for clarity. Request optimizations for performance or readability, like replacing nested loops with vector math.

```{figure} ./images/figure3.png
:align: center
:alt: ‘Snake’ game generated by Deepseek-R1 Distilled Qwen 32B model
Figure 3. ‘Snake’ game generated by Deepseek-R1 Distilled Qwen 32B model
```

### Advanced Game Generation – Pac-Man Inspired Game

Creating a simple game like *Snake* only requires a single prompt. In contrast, building a more complex game like *Pac-Man* demands a more structured and iterative approach.

Pac-Man introduces significantly more complexity than Snake, due to its maze structure, character behaviors, and layered mechanics. As a result, we found that a single prompt is insufficient to generate the entire game reliably.

This is where **Procedural Content Generation (PCG)** becomes valuable. While PCG is traditionally used to create dynamic or re-playable game environments, we can leverage its principles here in a novel way: as a *prompting strategy* to guide the LLM in *vibe coding* different parts of the game. Specifically, PCG-style thinking helps us break down and abstract the game into manageable components that the model can handle more consistently.

Here are the steps to create Pac-Man.

#### Abstract key elements of target game

To begin, we analyzed the original *Pac-Man* (often referred to as the **vanilla Pac-Man** or **Pac-Labyrinth**) shown in figure 4 and identified the 10 core elements required for recreating the game:

1. Maze: walls, paths, dots, Super-Pellets, base
2. Pac-Man: a yellow pie-shaped character
3. Four ghosts in different colors
4. Middle wraparound tunnel
5. Super-pellet mechanics
6. Character animations
7. Scoreboard
8. Player lives (typically 4)
9. Win/Lose conditions
10. Sound effects

```{figure} ./images/figure6.png
:align: center
:alt: Classic Pac-Man 1980
Figure 4. Classic Pac-Man 1980
```

#### Generate the Maze for basis

```{figure} ./images/figure7.png
:align: center
:alt: Figure 5: Maze Representation
Figure 5. Maze Representation
```

Maze generation, shown above in figure 5,  via vibe coding is one of the most challenging tasks in this game. DeepSeek-R1 struggles to consistently produce a maze that resembles the original Pac-Man layout. It often generates imperfect mazes with disconnected paths, inconsistent path lengths, or structures that significantly differ from the original design.

```{figure} ./images/figure8.png
:align: center
:alt: Maze Creation Challenges
Figure 6. Maze Creation Challenges
```

In our experiments, we found that explicitly referencing the original maze—**Pac-Labyrinth maze**—helps the model create layouts much closer to the intended design. Figure 6 highlights the evolution of our maze creation and the challenges that we faced. Additionally, specifying key details such as: *“Make the maze approximately the same size (28x31, using an ASCII-based map representation with '#' for walls) and follow a similar color scheme to the original Pac-Man”* results in noticeably more consistent outputs.

That said, leaving some elements, like the tunnel or the ghost base, in a simpler form that can be iteratively refined later often makes the overall game creation process smoother.

To establish a stable representation of the maze, we use a symbolic method to denote its various components, as shown in Fig. 5. The symbols are defined as follows: # represents walls, empty indicates open channels, stands for dots (which can be eaten to increase the score), and O represents Super-Pellets, which grant Pac-Man temporary powers to attack enemies.

#### Prompt Evolution

Below we will go over the prompting evolution while creating the Pac-Man inspired game that is highlighted in figure 7.

```{figure} ./images/figure9.gif
:align: center
:alt: 9 Prompt Evolutions
Figure 7. Nine Prompt Evolutions
```

* prompt\_1

Create a simple version of Pac-Man game for me using Pygame. For the maze, follow a simplified version of Pac-Labyrinth maze design. Make the maze simple by making it enclosed with no tunnel. Make the maze around the same size (28x31 ASCII-based map representation such as "#" to represent walls) and color schema as the original Pac-Man. Implement the winning, losing conditions as well as wall collision detection. Include 4 red squared shaped ghosts with eyes constantly moving in one direction but turning if there is an opening. Pac-Man's objective is to eat the dots, which disappear as it touches them.

* prompt\_2

Make the ghosts in different colors, rounded or square-like silhouette with a wavy, flowing hem at the bottom, evoking a ghostly sheet. This design emphasizes their ethereal, floating movement. Add *Super Pellets*, which behaves the same way as in the original Pac-Man game. A scoreboard at the top shows how many dots the player has eaten.

* prompt\_3

Implement wrap around logic for the middle tunnel so if any character leaves from the right tunnel enters from the left side, and vice versa.

* prompt\_4

The result is good. Let us continue to improve the Pac-Man’s appearance. Pac-Man is a bright yellow, circular, pie-shaped character with a distinctive open-and-close mouth animation facing its moving direction, resembling a pizza missing a slice as he appears to constantly chomp.

* prompt\_5

The result is good. Let us continue to improve Pac-Man’s appearance.  Pac-Man’s mouth animation is too simple and weird. We want to replicate the Pac-Man’s animation as close as possible to the one in the arcade game: Pac-Man (1980 Namco).

* prompt\_6

The result is good. Let us continue to improve the characters’ appearance. We want to make the ghosts look as close as the one in the arcade game: Pac-Man (1980 Namco). Plus, make the ghosts have a rounded, square-like silhouette with a wavy, flowing hem at the bottom, evoking a ghostly sheet.

* prompt\_7

When Pac-Man eat Super-pellets, the ghosts should not stand still, but also in moving status. Add 5 seconds time limit for the effect of Super Pellets.

* prompt\_8

Pac-Man has total 4 lives, show icon in bottom left represent this. If any collision with the ghosts, will cause Pac-Man to lose one life. When the Pac-Man is moving up and down, the mouth open direction is wrong inversed. Pls correct it.

* prompt\_9

Plus, add sound effect for the game, it’s better as close to the one in arcade game: Pac-Man (1980 Namco) as possible.

### Problems and Challenges

```{figure} ./images/figure.8.png
:align: center
:alt: Game Creation Challenges
Figure 8. Game Creation Challenges
```

Figure 8 shows 4 failures and challenges during the game code generation process with references to each failure below.

* A) Visible typos that are easy to find and correct via the LLM itself. If not, developers could run the code using the Pygame runtime, where sufficient error messages aid debugging.
* B) Game logic errors, such as ghosts moving unexpectedly fast or turning in the exact opposite direction of the player’s input. Similar to the above issue, this can be easily corrected by the LLM or developers.
* C) Ghosts staying in the base and refusing to hunt; we need prompts to make ghost movement more flexible.
* D) Pac-Man’s path was misaligned with the channels, causing it to run into walls. For this issue, the initial prompt is critical as it forms the game’s foundation. We must carefully choose the number of vertical/horizontal grids and set symbols for dots and Super-Pellets.

There was also an error in sound effect generation, where background music resembled noise and caused user discomfort. Switching to different sound-generation tools and algorithms resolved this. We used NumPy for algorithms and the wave library for audio file generation.

In total, it took 9 prompts to create the whole Pac-Man inspired game.

## Prompting Tips and Tricks

Below are helpful prompting tips to help you generate the whole game source code.

* Mention libraries or frameworks to use. For example, use Python and Pygame.
* Develop the game through multiple prompts. Start with a simple version, then incrementally add features or fix bugs with follow-up prompts.
* Procedurally generate. Following the principle of PCG, for some complex games, we need to generate the program with a dedicated order, from the initial scene to a sketch of the player and the enemy. Then, we can use more prompts to add detailed descriptions of these components. And more rules to the gaming mechanism.
* Leverage code-based software tools like NumPy and/or Wave as much as possible to generate your game components like map, sprites, motion, sound effects, etc. Avoid using graphical tools with user interfaces, which is difficult for existing LLMs to manipulation. For example, to generate the sound effects/music, we can use some software with graphical user interface to build and tune them with some buttons, However, the better way is to use NumPy to blend some sin/cos functions to generate these sounds.
* During iterative game creation, including details about code that worked well can help the model generate better code and improve performance in subsequent responses.
* Provide error logs and the code that caused the error to help the model identify cause and effect, allowing it to correct its mistakes based on evidence.
* Explicitly mentioning names and versions can guide the model to implement what you have in mind, more accurately. For example, Pac-Man 1980 Namco, Pac-Labyrinth maze design.
* Reading the model reasoning response can help debug and find the source of any errors and bugs.
* Do not disconnect the server during procedural content generation. Once the majority of the game has been developed, a single disconnection can disrupt the LLM’s reasoning path, making it difficult to resume the process from where it left off. Ensure a stable and continuous connection until the generation is fully complete.

## Summary

This blog explores how developers can use AMD MI300X GPUs and ROCm software with the DeepSeek-R1 language model to generate interactive games, including a Pac-Man-inspired title, through iterative prompting and procedural content generation (PCG). By combining LLM-powered coding tools with frameworks like SGLang and Python/Pygame, users can build complex, classic-style games—even without extensive programming experience. The process highlights both the strengths and limitations of LLMs in game development, underscoring the importance of structured prompts, effective debugging, and tool-assisted refinement.

Now it’s your turn: leverage the techniques and insights from this blog to reimagine your favorite retro or original game. Whether you start simple or aim for something more advanced, the tools and strategies discussed here can help bring your creative vision to life.

## Additional Resources

* <https://code.visualstudio.com/docs/setup/setup-overview>
* <https://github.com/greyblue9/pacman-python>
* <https://github.com/JohnSismanoglou/Pac-Man_Game/tree/main>
* [https://github.com/microsoft/vscode-ai-toolkit/blob/main/doc/playground.md#remote-inference-in-playground](https://github.com/microsoft/vscode-ai-toolkit/blob/main/doc/playground.md)
* <https://www.youtube.com/watch?v=dScq4P5gn4A>
* <https://www.youtube.com/watch?v=nNYXAzqMHiI>
* <https://www.youtube.com/watch?v=wUIQOEdMoEA>
* [Game Generation via Large Language Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10645597)
* [Game-based Platforms for Artificial Intelligence Research](https://chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https%3A/www.researchgate.net/profile/Chengpeng-Hu-3/publication/370295354_Game-based_Platforms_for_Artificial_Intelligence_Research/links/650b9f0c82f01628f0346412/Game-based-Platforms-for-Artificial-Intelligence-Research.pdf)
* [Procedural Content Generation via Generative Artificial Intelligence](https://arxiv.org/abs/2407.09013) <https://arxiv.org/abs/2407.09013>
* [ChatPCG: Large Language Model-Driven Reward Design for Procedural Content Generation](https://ieeexplore.ieee.org/abstract/document/10645619)
* [Games for Artificial Intelligence Research: A Review and Perspectives](https://ieeexplore.ieee.org/abstract/document/10552162)
* [MarioGPT: Open-Ended Text2Level Generation through Large Language Models](https://arxiv.org/pdf/2302.05981)
* [Evolving code with a large language model](https://link.springer.com/article/10.1007/s10710-024-09494-2)
* [Practical PCG Through Large Language Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10333197)
* <https://arxiv.org/abs/2403.10249>
* <https://arxiv.org/pdf/2410.01791>
* [MoGraphGPT: Creating Interactive Scenes Using Modular LLM and Graphical Control](https://arxiv.org/pdf/2502.04983)
* <https://dl.acm.org/doi/abs/10.1145/3649921.3659853>
* [Games for Artificial Intelligence Research: A Review and Perspectives](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10680313)
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

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
