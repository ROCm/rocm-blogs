---
blogpost: true
blog_title: "Benchmarking Reasoning Models: From Tokens to Answers"
date: 24 July 2025
author: 'Dominic Widdows'
thumbnail: 'benchmark_reasoning.png'
tags: AI/ML, Performance, Serving, GenAI, Reasoning, Benchmarking, Thinking Mode
category: Applications & models
target_audience: AI practitioners
key_value_propositions: Learn how to benchmark a reasoning model, and how many tokens are needed to get answers for different kinds of problems.
language: English
myst:
    html_meta:
        "author": "Dominic Widdows"
        "description lang=en": "Learn how to benchmark reasoning tasks. Use Qwen3 and vLLM to test true reasoning performance, not just how fast words are generated."
        "keywords": "Reasoning model, DeepSeek R1, Qwen 3, Benchmarking, Thinking Mode"
        "vertical": "AI, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, Data Science, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Dominic Widdows"
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

# Benchmarking Reasoning Models: From Tokens to Answers

This blog shows you how to benchmark large language models’ reasoning tasks by distinguishing between mere token generation and genuine problem-solving. You will learn the importance of configuring models like Qwen3 with “thinking mode” enabled, how standard benchmarks can produce misleading results, why reasoning requires more than just generating tokens quickly, and how to build evaluations that reflect the model’s true problem-solving capabilities. Sounds interesting? Let's dive right in!

## How Reasoning Works

Large language models (LLMs) have taken big steps forward in reasoning tasks.
LLMs demonstrate their reasoning steps in words, modeling how humans tackle problems in stages
and deliberate over proposed answers to evaluate which ones stand up to scrutiny.
This process takes longer than just generating plausible-sounding text, a difference that has been
compared to [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) in humans.
All things being equal, we would always prefer more accurate answers, but there is a natural tension between accuracy and cost in reasoning.
In this blog, you will learn how to measure how much time it _really_ takes to complete reasoning tasks,
and how to distinguish internal "thinking tokens" from final answers.

Early LLMs were notably poor at reasoning: given a problem requiring step-by-step reasoning, they would fluently
generate plenty of words about the problem, without correctly _solving_ the problem.
[Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903) in 2022 began to improve this behavior,
using carefully-analyzed story-problems to teach LLMs to proceed in a more logical, step-by-step fashion,
analogous to teaching students that they are more likely to complete a complicated problem correctly if they “show their work”.
By 2025, intermediate chain-of-thought steps have been built into families of models including
[OpenAI o3-mini](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1),
and [XAI Grok-3](https://x.ai/news/grok-3), which try to complete the “show your work” steps within the model,
before returning a more concise summary, or just the final answer, to the user.

The [Qwen3 model series](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f), released by Alibaba Cloud,
also performs well on reasoning challenges, and supports dynamic reasoning configurations, so that different
inference queries can request appropriate token budgets at runtime.
This is important, because reasoning models need to generate more tokens to deliver a complete answer compared to the non-reasoning model.

Because the amount of reasoning required depends on the complexity of a given task, a benchmark for a simple task can
be misleading if applied to a more difficult task that requires more internal tokens to complete.
Out-of-the-box, several standard LLM benchmark queries report misleadingly fast responses, because they
are configured to generate a given number of tokens, and then stop. Sometimes this token-target is added
implicitly within benchmark suites, so even when it looks like a benchmark output has no preconceived length
target, it often does.

You can easily produce representative benchmarks for reasoning challenges,
using just standard vLLM commands, as long as you carefully specify the arguments and avoid some crucial pitfalls.
This article compares several such benchmarking examples, showing how easy it is to produce optimistic but incorrect results
along the way, and eventually arriving at benchmarks that correctly test reasoning and complete problem-solving.

For readers interested in how this works for particular requests,
individual query examples are presented. This showcases some of the details of how Qwen3 and vLLM process requests with different formats,
leading to different fields in the responses, separating "thinking tokens" from "output tokens".
In the process, you'll use some of the recent methods for configuring conversations that involve reasoning,
which is an active area of open-source development.

## Prerequisites

To follow the steps in this blog, you will need:

* AMD GPU running ROCm: See the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems.
* ROCm 6.2 or later: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) page for installation instructions.
* Docker: See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

## Setting Up the Qwen3 Server Using vLLM

The [Qwen3-8B model](https://huggingface.co/Qwen/Qwen3-8B) is used in the examples, because it is the smallest and quickest of the Qwen3 series to work with.
The experiments in this blog can be reproduced with one MI300X GPU and should take just a few minutes to set up.

Requests to the Qwen3 model are served by using the [vLLM](https://docs.vllm.ai/) (virtual Large Language Model) platform.
See the [Qwen - vLLM](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) page for more details on compatibility and configuration.
To quickly set up an environment with vLLM, pull and run a suitable docker container from [rocm/vllm](https://hub.docker.com/r/rocm/vllm/tags), for example:

```console
docker run -it --name qwen3_reasoning \
  --network=host --group-add=video --privileged --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri rocm/vllm:rocm6.4.1_vllm_0.9.1_20250702
```

At the time of writing, this container version is the official `rocm/vllm:latest` &mdash; you can use `vllm:latest`
to guarantee up-to-date features, or the fixed version to guarantee that nothing has changed.

Download the Qwen3-8B model from huggingface using:

```console
huggingface-cli download Qwen/Qwen3-8B
```

Now start a vllm server for this model using:

```console
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --tensor-parallel-size 1 \
  --swap-space 16 --max-num-seqs 128 --distributed-executor-backend mp --gpu_memory_utilization 0.9 
```

After a few minutes, you should see the message `INFO: Application startup complete`, indicating that the server is running and ready to receive queries.

Once you've seen that message, start a new shell, and use the command `docker exec -it qwen3_reasoning bash`.
This bash session in the container will be used to send queries to the vLLM server and display the results.

## Token Benchmarks Compared with Reasoning Benchmarks

This section walks you through different vLLM commands for benchmarking how quickly responses are generated.
It's natural to expect that answers to complicated math problems take longer, and require more "thinking tokens", than simple prompts and text responses.
This tendency is demonstrated in the examples below.
However, it's deceptively easy to run standard vLLM benchmarks with default parameters that appear to test reasoning, but instead yield misleadingly fast results.

The rest of this section walks you through a range of benchmarks and explains
why small changes in benchmarking parameters can lead to significant differences in results.
The results are summarized in a table at the end.

### Default Prompt-completion Queries

Suppose you were asked to benchmark how quickly the Qwen3-8B model can reason and produce correct answers.
If you use the default benchmarking script from the vLLM repository, with standard arguments for benchmarking questions with 128 words and outputs with 1024 words, you might start with a simple vLLM benchmark query such as:

```console
python3 /app/vllm/benchmarks/benchmark_serving.py --model Qwen/Qwen3-8B \
  --num-prompts 128 --dataset-name random --random-input-len 128 --random-output-len 1024
```

This is the **Basic continuation** example, which runs end to end in under a minute, obtaining a throughput of around 7.87 requests per second, on a single MI300X GPU.
However, if you modify the vLLM benchmark code to print out the benchmark queries (for example, by adding `print(f"Prompt: {request.prompt}")` at
[line 377 of /app/vllm/benchmark_serving.py](https://github.com/vllm-project/vllm/blob/71faa188073d427c57862c45bf17745f3b54b1b1/benchmarks/benchmark_serving.py#L377),
you will see that the queries are random strings like `emporary reactpipecz.activity largely dissaxyesis ...`. It's hard to imagine that the model is doing any
logical reasoning with that input!

Thankfully, the vLLM benchmark_serving tool supports other query datasets, including the [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime),
which is a collection of problems and solutions from the American Invitational Mathematics Examinations [(AIME)](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).
You can get the AIMO continuation result by using this command:

```console
python3 /app/vllm/benchmarks/benchmark_serving.py --model Qwen/Qwen3-8B \
  --dataset-name hf --dataset-path AI-MO/aimo-validation-aime --num-prompts 128 --hf-output-len 1024
```

If you print the queries using the `print(f"Prompt: {request.prompt}")` edit above, you'll see challenging math questions. For example,
_A circle with radius 6 is externally tangent to a circle with radius 24. Find the area of the triangular region bounded by the three common tangent lines of these two circles,_
which is [AIME 2020 Problem 7](https://artofproblemsolving.com/wiki/index.php/2022_AIME_II_Problems/Problem_7).

These benchmark results are nearly as fast as those obtained with the random dataset: 6.4 requests per second, and 7243 tokens per second.
By default, vLLM benchmarks use the [completions](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve.html#__codelineno-0-652) API,
so what’s really being measured is token generation: 1024 tokens per request.

With this query, the `--hf-output-len 1024` argument plays the same role as the `--random-output-len` with the queries above:
the model is asked to produce this many tokens in response to each query.
Removing the `--hf-output-len 1024` directive allows the model to generate different numbers of tokens for each request,
but it does not enable the model to continue thinking until it gets to an answer. Instead, the
VLLM dataset loading code in [datasets.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/datasets.py#L1248)
uses the official solution to the problem in the [AI-MO](https://huggingface.co/datasets/AI-MO/aimo-validation-aime) dataset
to set the target number of tokens to generate. This considerably underestimates the number of tokens the model
will need along the way. As for most human mathematicians, the number of pages needed for rough work
is often much greater than the pages needed to write up a solution or proof in its final form for presentation.

### Chat Queries with Thinking Mode

The [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/) defines _thinking mode_ as the mode in which "the model takes time to reason step by step before delivering the final answer."
This step by step reasoning also involves generating tokens.
The tokens used internally for step-by-step reasoning are not usually visible to end users, but for complicated reasoning problems, they account for most of the response time. In Qwen3, thinking mode is enabled by default, and you can find configuration details in the [Qwen3 Quickstart guide](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html).

Benchmarking a set of vLLM queries where Qwen3-8B reasons until it completes a problem requires a different
chat API endpoint (`--endpoint /v1/chat/completions`) and backend (`--backend openai-chat`).
Though it's not the default in vLLM, using chat-based protocols is typically best practice for instruction-tuned models,
which includes all models in the Qwen3 family.

This leads to the AIMO chat request, which works as follows:

```console
python3 /app/vllm/benchmarks/benchmark_serving.py --model Qwen/Qwen3-8B \
  --dataset-name hf --dataset-path AI-MO/aimo-validation-aime --num-prompts 128 \
  --endpoint /v1/chat/completions --backend openai-chat --hf-output-len 38912
```

That query takes much longer. In an experiment on one MI300X GPU, it ran for nearly 36 minutes,
generating about 2.45 million tokens (about 19000 per query). Despite generating over 1100 tokens per second,
the request rate was only 0.06 requests per second.

The change of `--hf-output-len` to a much larger value is the main reason this benchmark achieves much
slower request throughout. This flag also has a different effect this time.
While the output setting using `--hf-output-len` looks syntactically
identical to the way it's used in previous example, with the `openai-chat` backend setting, it gets interpreted
as a token _limit_, not a target to reach.
(The limit of 38912 is used in the [Qwen3 technical report](https://arxiv.org/abs/2505.09388).)

How much longer does Qwen3-8B take to solve reasoning problems? It depends on how hard the problems are.
To demonstrate this, you can use or create a collection like [mult.jsonl](./src/mult.jsonl) containing simpler math questions
such as "What is X times Y?", where X and Y are random numbers between 0 and 1024.
The script that generated these examples is in [rand_mult.py](./src/rand_mult.py), if you want to use the format to create different problem types.

You can now use this dataset as a custom input for the multiplication thinking benchmark with the following command, which is otherwise identical to the previous reasoning benchmark request:

```console
python3 /app/vllm/benchmarks/benchmark_serving.py --model Qwen/Qwen3-8B \
  --dataset-name custom --dataset-path ./mult.jsonl --num-prompts 128 \
  --endpoint /v1/chat/completions --backend openai-chat --custom-output-len 38912
```

This benchmark uses 4516.58 tokens for each request, and achieves a request throughput of 0.97 req/s,
somewhere in between the 6-8 req/s rapid-response for the text completion benchmarks above, and the slow deliberation at 0.08 req/s for the AIME problems.

The following query is exactly the same, except for using a smaller limit of 8000 on the output length:

```console
python3 /app/vllm/benchmarks/benchmark_serving.py --model Qwen/Qwen3-8B \
  --dataset-name custom --dataset-path ./mult.jsonl --num-prompts 128 \
  --endpoint /v1/chat/completions --backend openai-chat --custom-output-len 8000
```

If you run this query, you should see a very similar result to the previous multiplication thinking benchmark, because
the model typically finds an answer before it reaches the `--custom-output-len` token limit.
Multiplying 3-digit numbers is simple enough that it doesn't need a lot of reasoning tokens compared with the AIME problems.

### Benchmarking with Thinking Mode Disabled

Qwen3 models can be run with "thinking mode" switched off entirely. For Qwen3 itself, different thinking configurations can be set for different queries,
but these settings are not currently supported in the vLLM `benchmark_serving.py` arguments. Instead, the vLLM server can be started with the
[qwen3_nonthinking.jinja](https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja)
custom chat template
from the [Qwen3 vLLM guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes), as follows:

```console
curl https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja > qwen3_nonthinking.jinja
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --tensor-parallel-size 1 \
  --swap-space 16 --max-num-seqs 128 --num-scheduler-steps 10 \
  --distributed-executor-backend mp --gpu_memory_utilization 0.9 \
  --chat-template ./qwen3_nonthinking.jinja
```

This previous benchmark command runs much faster now that thinking mode is switched off: the same command achieves a throughput of over 16 requests per second,
giving the Multiplication no thinking result in the summary table.
However, these vLLM benchmarks do not verify whether the answers are actually correct.
Techniques for checking the correctness of results with different "thinking budgets" are demonstrated separately below.

### Summary of Benchmarks

The benchmarks are summarized in the following table.

| Benchmark Name | Description | Prompts Used | Total Tokens <br> Generated | Token Throughput <br> (tok/s) | Request Throughput <br> (req/s) |
|-|-|-|-|-|-|
| **_Basic continuation_** | Shows only how fast the model generates a requested number of tokens. | Random tokens | 113,621 | 7997.95 | 7.87 |
| **_AIMO continuation_** | Looks like it's answering math questions, but is still just generating requested tokens. | Expert math questions (from [aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)) | 131,072 | 7243.86 | 6.4 |
| **_AIMO chat_** | Uses the correct request structure and API to benchmark solving math questions. | Expert math questions | 2,449,520 | 1142.08 | 0.06 |
| **_Multiplication thinking_** | Also completes simpler math questions, much faster. | Multiply 3-digit numbers | 468,429| 3576.97 | 0.97 |
| **_Multiplication no thinking_** | Demonstrates faster but less reliable generation without thinking mode. | Multiply 3-digit numbers | 27,346 | 3410.84 | 14.61 |

Points to emphasize for benchmarking reasoning models include:

* Only the **_AIMO Chat_** and **_Multiplication thinking_** are really benchmarking end-to-end reasoning.
* Standard token-based benchmark requests can considerably underestimate how long it takes to solve reasoning problems.
* It's important to make sure that the request format and API are chosen appropriately, especially as new formats and reasoning parsers are going through rapid development cycles.
* Problems at different levels of difficulty require different numbers of reasoning tokens to solve.

These experiments suggest the Qwen3-8B model can solve moderately hard arithmetic problems on a single MI300X GPU at about 1 request per second.
However, this is only a ballpark number.
To predict performance with real production tasks, you need to ensure your benchmark tasks closely reflect the actual production workload.

## Comparing Individual Qwen3 Reasoning Queries

After comparing the benchmarks at a high level, this section goes further into the details of particular queries.
Some of the configuration options available for individual queries are not yet supported in vLLM
benchmark suites, including counting "thinking tokens" separately from "output tokens".
While benchmark results provide a useful overview, analyzing individual requests showcases some of the distinctions that aren't easily accessible in benchmark results.

For better visibility when analyzing results, run `apt update; apt install jq` to install [jq](https://jqlang.org/), a useful utility for interacting with json content.
Then add the Unix pipe `| jq` after each curl request to format the output and make it easier to read.

The Qwen3 models are all instruction-tuned, and should typically be used with the vLLM chat API, rather than the completions API.
See [Qwen3 Quickstart manual](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) for examples.

In particular, compare the following curl requests. This one uses the basic `completions` API:

```console
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{ "model": "Qwen/Qwen3-8B",
          "prompt": "What is 2 plus 2?",
          "max_tokens": 1024 }' | jq
```

You should see a result with many tokens in the `text` field, sounding like a long ramble of random thoughts following from "What is 2 plus 2?"

```console
{
  "id": "cmpl-3c5abeab44b847d3b8c6b15eb65aaa0c",
  "object": "text_completion",
  "created": 1752173426,
  "model": "Qwen/Qwen3-8B",
  "choices": [
    {
      "index": 0,
      "text": " What is 3 plus 3? What is 4 plus 4? What is 5 plus 5? What is 6 plus 6? What is 7 plus 7? What is 8 plus 8? What is 9 plus 9? What is 10 plus 10? What is 11 plus 11?\n\nLet me solve these one by one. \n\nFirst, 2 plus 2 is 4. Then, 3 plus 3 is 6. Next, 4 plus 4 is 8. Following that, 5 plus 5 is 10. Then, 6 plus 6 is 12. After that, 7 plus 7 is 14. Next, 8 plus 8 is 16. Then, 9 plus 9 is 18. Following that, 10 plus 10 is 20. Finally, 11 plus 11 is 22. \n\nEach time, I'm adding the same number to itself, which is equivalent to multiplying by 2. So, for any number $ n $, $ n + n = 2n $. Applying this formula to each of the numbers given gives the results as above. \n\nFor example, 2 + 2 = 2×2 = 4, 3 + 3 = 3×2 = 6, and so on up to 11 + 11 = 11×2 = 22. \n\nThis pattern holds true for all the numbers listed, so the answers are consistent with the doubling of each number. There are no exceptions in this sequence because we're simply adding the same number to itself, which is a straightforward arithmetic operation. \n\nTherefore, the final answers are as calculated.\n\\boxed{4}, \\boxed{6}, \\boxed{8}, \\boxed{10}, \\boxed{12}, \\boxed{14}, \\boxed{16}, \\boxed{18}, \\boxed{20}, \\boxed{22} respectively.\nHowever, since the question asks for the final answer within a single box, I will list them all in one box as requested.\n\\boxed{4}, \\boxed{6}, \\boxed{8}, \\boxed{10}, \\boxed{12}, \\boxed{14}, \\boxed{16}, \\boxed{18}, \\boxed{20}, \\boxed{22}\nBut since the instructions specify to put the final answer within \\boxed{}, and there are multiple answers, I should present them all boxed individually as per the examples. However, if only one answer is expected, it's likely the last one, which is 11 + 11. Let me check the original question again.\n\nThe original question lists multiple questions, each asking for the result of adding a number to itself. The user might expect each result to be boxed individually. However, given the format, I will present each answer in a separate box as per the instructions.\n\\boxed{4}, \\boxed{6}, \\boxed{8}, \\boxed{10}, \\boxed{12}, \\boxed{14}, \\boxed{16}, \\boxed{18}, \\boxed{20}, \\boxed{22}\nBut if the system requires only one boxed answer, it's likely the last one, which is 22. Let me confirm with the initial instructions. The user says, \"put your final answer within \\boxed{}\". Since there are multiple answers, but the system might expect one. However, the original question lists multiple parts, so it's possible that each part should have its own box. But since the assistant is supposed to provide the final answer within \\boxed{}, and the user might be expecting a list, I'll follow the example of the previous assistant who provided each answer boxed individually. So the final answers are as above.\n\\boxed{4}, \\boxed{6}, \\boxed{8}, \\boxed{10}, \\boxed{12}, \\boxed{14}, \\boxed{16}, \\boxed{18}, \\boxed{20}, \\boxed{22}\nHowever, if the system only allows one boxed answer, the last one would be \\boxed{22}. But since the user listed all the questions, I think providing each answer boxed is appropriate. But the initial instruction says \"put your final answer within \\boxed{}\", which might imply a single box. To comply with the instruction, I'll provide the final answer as the last one, which is 22.\n\\boxed{22}\nBut I'm not sure. Let me check again. The user wrote:\n\n\"What is 2 plus 2? What is 3 plus 3? ... What is 11 plus 11?\"\n\nThen says \"put your final answer within \\boxed{}\". It's possible that the",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 1032,
    "completion_tokens": 1024,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```

This exemplifies the problem with some of the token-based benchmarks above: the model can generate tokens following the general theme
of playing with numbers, but it still fails to clearly answer the simple question, "What is 2 plus 2?"
This example illustrates what we mean by _generating tokens_ vs. _generating answers_.
The model could generate indefinitely, and reach a high token throughput, but if this doesn't lead to a correct answer, the token throughput doesn't matter.

Instead, use the `chat/completions` API endpoint with the following structured query:

```console
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{ "model": "Qwen/Qwen3-8B",
          "messages": [ { "role": "user",
                          "content": "What is 2 plus 2?" } ],
          "max_tokens": 1024 }' | jq
```

This response should contain both a `content` field with a clear answer like “The result of 2 plus 2 is 4.”, and a `reasoning_content` field which includes the tokens generated along the way.

```console
{
  "id": "chatcmpl-962df46b9f2c4547bb3e98484add738f",
  "object": "chat.completion",
  "created": 1752173501,
  "model": "Qwen/Qwen3-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "\nOkay, the user asked \"What is 2 plus 2?\" Let me think about how to approach this. First, I need to make sure I understand the question correctly. It's a basic arithmetic problem, so the answer should be straightforward. But maybe they want a more detailed explanation or are testing if I can handle simple math.\n\nI should recall that addition is one of the fundamental operations in mathematics. The question is asking for the sum of 2 and 2. Let me verify the calculation. 2 + 2 equals 4. That's a basic fact, but I should double-check to avoid any mistakes. \n\nWait, could there be any context where this question might not be straightforward? For example, in some programming languages or specific mathematical contexts, the plus operator might have different meanings. But in standard arithmetic, 2 plus 2 is definitely 4. \n\nThe user might be a student learning basic math, or they could be testing my knowledge. Either way, the answer is simple. However, I should present it clearly. Maybe they want a step-by-step explanation. Let me break it down. \n\nStart with the number 2. Then add another 2 to it. When you add 2 and 2 together, you're combining two groups of two items each. So, 2 + 2 = 4. Alternatively, using the number line: start at 2 and move 2 units to the right, which lands you at 4. \n\nI should also consider if there's any alternative interpretation. For instance, in some contexts, like binary numbers, 2 plus 2 would be different. But the question doesn't specify a different base, so it's safe to assume base 10. \n\nAnother angle: maybe they're asking about the properties of addition. The commutative property states that 2 + 2 is the same as 2 + 2, which is still 4. The associative property would also hold here, but that's probably not necessary for such a simple question. \n\nI think the best approach is to state the answer clearly and confirm that it's a basic arithmetic problem. Maybe add a brief explanation to ensure clarity. Let me check if there's any chance of confusion. For example, if someone is not familiar with the concept of addition, but the question seems straightforward. \n\nIn conclusion, the answer is 4. I should present it confidently and concisely.\n",
        "content": "\n\nThe result of 2 plus 2 is **4**. This is a basic arithmetic operation where the numbers 2 and 2 are combined using addition, yielding the sum 4.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 558,
    "completion_tokens": 542,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

### Configuring Reasoning Mode

The benchmarks above used a custom chat template to configure the vLLM server never to use thinking mode.
That was a quick and blunt approach working with any query.

For more varied options, thinking mode can instead be configured individually in each request.
Conventions for this are relatively new, and more are likely to be introduced in the coming months.
Options recognized by the Qwen3 vLLM server include:

* `chat_template_kwargs: {"enable_thinking": true|false}`, which switches reasoning mode on (default) or off.
* Reasoning mode can also be disabled by inserting the string `/nothink` at the beginning of the prompt.

For example, running even the simple "What is 2 plus 2?" query using `"max_tokens": 64` causes the server to return
a response with some thinking tokens in the `reasoning_content` field but no final answer.
This trivial question is actually handled better without thinking enabled, as in:

```console
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-8B", "messages": [ { "role": "user", "content": "What is 2 plus 2?" } ],
    "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}
  }' | jq
```

This question is so simple that the model responds with `"reasoning_content": "2 plus 2 is **4**."` without even having to use thinking mode.

Now try asking a harder question such as “What is 7543 plus 3542?” using exactly the same formats.
Given enough `max_tokens` budget, the answer should still be correct, but the query will take longer to run, the `reasoning_content` will have more tokens,
and the final answer in the `content` field may have more explanation. With `{"enable_thinking": true}`, it took around `"max_tokens": 3000` in ad hoc tests
for the model to be confident that its answer of 11085 was correct.

The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) analyses such trends over different test collections, demonstrating that allowing more tokens for internal reasoning
(up to a maximum of 38912 tokens per problem) contributes to obtaining better answers for math, science, and coding problems.
Estimating how many tokens a problem will require _before_ attempting to solve it remains an open and challenging question:
it's known from Turing's founding work in computer science that the [Halting problem](https://en.wikipedia.org/wiki/Halting_problem) cannot be solved in general,
though estimates for limited classes of problems could be obtained heuristically or from previous examples.

For benchmarking, this raises a question about the results above: when using the `chat/completions` endpoint, do the vLLM benchmarks count
the `reasoning_content` tokens, the `content` tokens, or both? At the time of writing, the answer is "both". The method for handling streaming
responses in [backend_request_func.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py#L87) is good for
measuring inter-token latency, but it puts all these generated tokens in the same `generated_text` field, and all these tokens are counted together.

It's possible to count the number of tokens used internally in reasoning, and those generated in the final response, using a separate script.
Download the file [count_response_tokens.py](./src/count_response_tokens.py), and save it in the same directory as the [mult.jsonl](./mult.jsonl) set of example questions used above.
Now run the script using

```console
python count_response_tokens.py --num-prompts 10
```

The output should look like this:

```console
Prompt: 'What is 662 times 359?' Reasoning tokens: 3018. Final content tokens: 446
Prompt: 'What is 108 times 433?' Reasoning tokens: 3812. Final content tokens: 409
...
...
========================================
Total queries: 10
Total reasoning tokens: 33686
Total final content tokens: 3536
Average reasoning tokens: 3368.60
Average content tokens: 353.60
```

In this case, about 9.5% of the tokens that the model generated were used in the final answer, and more than 90% were used internally as reasoning tokens.

The `count_response_tokens.py` script runs much more slowly than running the same queries as a custom dataset in the `benchmark_serving.py`, because
`count_response_tokens.py` runs the queries sequentially, whereas vLLM benchmark serving executes these queries together as a batch.
However, the new benchmark script is an easy way to gather supplemental information.

### Summary of Individual Queries

Individual queries and responses can be examined in more detail, and can sometimes be manipulated with flexible options that might not be integrated into vLLM benchmarks yet.
This provides deeper insight about exactly what each request and response contains. However, some advanced features that vLLM relies on at scale might be hard to incorporate.

Coordinated development of reasoning models, query parsers, and conversational APIs, is still in an early phase: it's likely that several other options and conventions will be
introduced over the coming months, and the most successful designs will become more seamlessly integrated into serving platforms.

Even if all the metrics you want to analyze are supported in a benchmark suite, it's still important to test the behavior of individual queries as well, to make sure that
summary statistics about, for example, how many tokens are generated are a good reflection of what users would see in practice.
In this case, adding an extra script to run multiplication queries demonstrated that over 90% of the tokens reported in the original benchmark are typically not visible to end users.

## Summary

Reasoning models like the Qwen3 series can solve sophisticated math problems,
but the extra reasoning takes more time, and more intermediate steps, than
most conversational language. Benchmarking methods for LLMs as
large _language_ models have naturally focused on how quickly tokens are generated,
but for large _reasoning_ models, tokens are a byproduct, on the way to generating
answers, and the speed of answer-generation depends on the complexity of the question,
as well as the efficiency of the model.

In this blog, you learned how to design benchmarks that capture true reasoning behavior, distinguishing accurate problem-solving from mere token output.
As these models get more capable,
developers and product managers need to pay even closer attention to what tasks a
model is being used for, and to make sure that the resources used are enough to produce
reliable results, without wasting words going round in circles.

The tools and examples demonstrated above are a good starting guide on how to go about this, but this is just a beginning.
Reasoning models will continue to develop rapidly in coming months.
More tools will be needed to monitor reasoning behaviors in real time,
and as reasoning models are applied to tasks across different domains,
system administrators are likely to need much more advanced options.
Configuring "thinking mode" using just token limits and a global on/off switch may sound very primitive in years to come!

Predicting how many reasoning steps will be needed to solve a given problem is a challenge as old as computer science itself.
As algorithms are now both designed and run by computers, this research area has become even more active.
Reasoning AI models have answered some hard questions already, but the questions they pose are harder still!

## Additional Readings

In addition to the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), results on reasoning challenges like AIME are included in system reports for
[OpenAI o-series](https://openai.com/index/introducing-o3-and-o4-mini/), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and [Grok 3](https://x.ai/news/grok-3).

"How long will a computation take to finish?" is a foundational question in computer science. In 1936, Turing's paper
[On Computable Numbers](https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/plms/s2-42.1.230) showed that some
programs will _never_ finish. Various similar results are today grouped under the [Halting Problem](https://en.wikipedia.org/wiki/Halting_problem).
Since then, _complexity theory_ has developed in computer science, classifying formal problems into categories like polynomial, non-deterministic, and exponential.
For a summary, see Scott Aaronson’s readable introduction [Why Philosophers Should Care About Computational Complexity](https://www.scottaaronson.com/papers/philos.pdf).

Predicting the resources needed for _informal_ problem statements will be harder still, and much more varied.
Some studies have already tried this, including the [THOUGHTTERMINATOR](https://arxiv.org/abs/2504.13367v1) paper of Pu et al.

The comparison of [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) comes from the book by psychologist Daniel Kahneman, which has become
a popular introduction to difference between quick instinctive mental responses, and ponderous exact reasoning.
The proposal that large language models can work in a similar fashion is explored in [Thinking Fast and Slow in Large Language Models](https://arxiv.org/abs/2212.05206)
by Hagendorff et al.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
