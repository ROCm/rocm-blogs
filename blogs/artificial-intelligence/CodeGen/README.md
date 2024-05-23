---
blogpost: true
date: 16 Apr 2024
author: Phillip Dang
tags: PyTorch, AI/ML, GenAI
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Program Synthesis with CodeGen"
    "keywords": "PyTorch, PyTorch Lightning, train models, Tuning, Generative AI"
    "property=og:locale": "en_US"
---

# Program Synthesis with CodeGen

CodeGen is a family of standard transformer-based auto-regressive language models for program synthesis, which as [defined by the authors](https://arxiv.org/pdf/2203.13474.pdf) as a method for generating computer programs that solve specified problems, using input-output examples or natural language descriptions.

The specific CodeGen model that we'll be testing is fine-tuned on a set of data which consists of 71.7B tokens of Python programming language. For a deeper dive into the inner workings of CodeGen, we recommend that users take a look at [this paper](https://arxiv.org/pdf/2203.13474.pdf) from Salesforce.

In this blog, we run several inferences with CodeGen and demonstrate how it works out-of-the-box with AMD GPUs and ROCm.

## Prerequisites

* Software:
  * [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
  * [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
  * Linux OS

For a list of supported GPUs and OS, please refer to [this page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). For convenience and stability, we recommend pulling and running the ROCm/PyTorch Docker container in your Linux system with the following code:

```sh
docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
           --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --name=olmo rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1 /bin/bash
```

* Hardware:

Make sure the system recognizes your AMD GPU:

``` python
! rocm-smi --showproductname
```

```bash
================= ROCm System Management Interface ================
========================= Product Info ============================
GPU[0] : Card series: Instinct MI210
GPU[0] : Card model: 0x0c34
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] : Card SKU: D67301
===================================================================
===================== End of ROCm SMI Log =========================
```

Let's check if we have the right version of ROCm installed.

```python
!apt show rocm-libs -a
```

```bash
Package: rocm-libs
Version: 5.7.0.50700-63~22.04
Priority: optional
Section: devel
Maintainer: ROCm Libs Support <rocm-libs.support@amd.com>
Installed-Size: 13.3 kB
Depends: hipblas (= 1.1.0.50700-63~22.04), hipblaslt (= 0.3.0.50700-63~22.04), hipfft (= 1.0.12.50700-63~22.04), hipsolver (= 1.8.1.50700-63~22.04), hipsparse (= 2.3.8.50700-63~22.04), miopen-hip (= 2.20.0.50700-63~22.04), rccl (= 2.17.1.50700-63~22.04), rocalution (= 2.1.11.50700-63~22.04), rocblas (= 3.1.0.50700-63~22.04), rocfft (= 1.0.23.50700-63~22.04), rocrand (= 2.10.17.50700-63~22.04), rocsolver (= 3.23.0.50700-63~22.04), rocsparse (= 2.5.4.50700-63~22.04), rocm-core (= 5.7.0.50700-63~22.04), hipblas-dev (= 1.1.0.50700-63~22.04), hipblaslt-dev (= 0.3.0.50700-63~22.04), hipcub-dev (= 2.13.1.50700-63~22.04), hipfft-dev (= 1.0.12.50700-63~22.04), hipsolver-dev (= 1.8.1.50700-63~22.04), hipsparse-dev (= 2.3.8.50700-63~22.04), miopen-hip-dev (= 2.20.0.50700-63~22.04), rccl-dev (= 2.17.1.50700-63~22.04), rocalution-dev (= 2.1.11.50700-63~22.04), rocblas-dev (= 3.1.0.50700-63~22.04), rocfft-dev (= 1.0.23.50700-63~22.04), rocprim-dev (= 2.13.1.50700-63~22.04), rocrand-dev (= 2.10.17.50700-63~22.04), rocsolver-dev (= 3.23.0.50700-63~22.04), rocsparse-dev (= 2.5.4.50700-63~22.04), rocthrust-dev (= 2.18.0.50700-63~22.04), rocwmma-dev (= 1.2.0.50700-63~22.04)
Homepage: https://github.com/RadeonOpenCompute/ROCm
Download-Size: 1012 B
APT-Manual-Installed: yes
APT-Sources: http://repo.radeon.com/rocm/apt/5.7 jammy/main amd64 Packages
Description: Radeon Open Compute (ROCm) Runtime software stack
```

Make sure PyTorch also recognizes the GPU:

``` python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

``` cpp
number of GPUs: 1
['AMD Radeon Graphics']
```

Let's start testing CodeGen.

## Libraries

Before you begin, make sure you have all the necessary libraries installed:

``` python
!pip install transformers
```

Next import the modules you'll be working with for this blog:

```python
import torch
import time 
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## Loading the model

Let's load the model and its tokenizer. CodeGen has several variants at different sizes from 350M to 16.1B parameters. In this blog, we'll be running inferences on the 350M parameters variant of the model.

```python
torch.set_default_device("cuda")
start_time = time.time()
checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(f"Loaded in {time.time() - start_time: .2f} seconds")
print(model)
```

```python
Loaded in  6.89 seconds
CodeGenForCausalLM(
  (transformer): CodeGenModel(
    (wte): Embedding(51200, 1024)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-19): 20 x CodeGenBlock(
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): CodeGenAttention(
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
          (qkv_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (mlp): CodeGenMLP(
          (fc_in): Linear(in_features=1024, out_features=4096, bias=True)
          (fc_out): Linear(in_features=4096, out_features=1024, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1024, out_features=51200, bias=True)
)
```

## Running inference

Let's create a function that takes in some input prompt and generates the output. We'll also estimate the following 2 inference metrics:

1. Latency: The total time it takes for the model to generate the output
2. Throughput: The number of output tokens per second

```python
def run_inference(raw_input):
    start_time = time.time()
    inputs = tokenizer(raw_inputs, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs,max_length=1000)
    latency = time.time() - start_time
    throughput = len(outputs[0]) / latency
    print(f"Latency:    {latency: .2f} seconds")
    print(f"Throughput: {throughput: .2f} tokens/s")
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
```

With this, we're ready to run inference and have some fun with CodeGen! We'll be testing the model's ability to generate code.

### Generate code

Let's give CodeGen a medium difficulty [Leetcode question](https://leetcode.com/problems/3sum/description/) and see how it does.

```python
raw_inputs = ''' 
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
'''
text = run_inference(raw_inputs)
```

Output:

```python
Latency:  14.45 seconds
Throughput:  36.12 tokens/s
 
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i!= j, i!= k, and j!= k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation:
-1 and -1 are triplets.
-1 and 0 are not triplets.
-1 and 1 are not triplets.
-4 and -1 are not triplets.
-4 and -1 are triplets.
-4 and 0 are not triplets.
-4 and 1 are triplets.
-1 and 2 are not triplets.

Example 2:

Input: nums = []
Output: []

Example 3:

Input: nums = [0]
Output: []

Constraints:

1 <= nums.length <= 104
-104 <= nums[i] <= 104

"""

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                if nums[i] + nums[l] + nums[r] == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    l += 1
        return res
<|endoftext|>
```

While the answer is correct and accepted by Leetcode, we see that the example generated by the model `1 and -1 are triplets` does not make much sense.

Let's try [another problem](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/), this time with a bit of a twist:

```python
raw_inputs = ''' 
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.
'''
text = run_inference(raw_inputs)
```

Output:

```python
Latency:  13.03 seconds
Throughput:  41.05 tokens/s
 
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.

Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
Example 2:

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 3 is 6. Therefore index1 = 1, index2 = 3.
Example 3:

Input: numbers = [2,3,4], target = 18
Output: [1,3]
Explanation: The sum of 2 and 3 is 6. Therefore index1 = 1, index2 = 3.
Example 4:

Input: numbers = [2,3,4], target = 0
Output: [1,2]
Explanation: The sum of 2 and 0 is 0. Therefore index1 = 1, index2 = 2.
Example 5:

Input: numbers = [2,3,4], target = 10
Output: [1,3]
Explanation: The sum of 2 and 3 is 6. Therefore index1 = 1, index2 = 3.

Constraints:

1 <= numbers.length <= 10^4
-10^9 <= numbers[i] <= 10^9
-10^9 <= target <= 10^9

"""

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                if numbers[i] + numbers[j] == target:
                    return [i, j]
        return []
<|endoftext|>
```

This time the answer is incorrect since it missed one of the requirements in the question, which is `Return the indices of the two numbers, index1 and index2, added by one`.

Next, let's ask the model to generate a commonly used loss function in training ML models. This time we do not constrain it as much as the two previous examples.

```python
raw_inputs = ''' 
Implement the cross entropy loss function
'''
text = run_inference(raw_inputs)
```

Output:

```python
Latency:  32.24 seconds
Throughput:  31.02 tokens/s
 
Implement the cross entropy loss function

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        n = inputs.size(0)
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1)
        loss = torch.mean(loss)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        n = inputs.size(0)
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1)
        loss = torch.mean(loss)
        return loss

class DiceLoss2(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss2, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        n = inputs.size(0)
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1)
        loss = torch.mean(loss)
        return loss
```

Here we see that even though the model already finishes generating the `CrossEntropyLoss` class, it goes on and on to generate unnecessary code until it reaches the `max_length=1000`.

Finally, let's ask the model to implement binary search.

```python
raw_inputs = ''' 
Implement binary search
'''
text = run_inference(raw_inputs)
```

Output:

```python
Latency:     4.60 seconds
Throughput:  30.65 tokens/s
 
Implement binary search
"""

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

arr = [1,2,3,4,5,6,7,8,9,10]
target = 10
print(binary_search(arr, target))<|endoftext|>
```

This time, we see the model is able to perfectly implement binary search!

From the examples above, we see that CodeGen works quite well barring some strange behaviors like not knowing when to stop or missing some minor details in the responses. This could be due to our using the smallest variant with 300M parameters, which is quite small for a language model. Readers are encouraged to explore larger variants and test out the quality of the generated responses.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
