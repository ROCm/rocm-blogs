---
blogpost: true
date: 29 May 2024
author: Phillip Dang
tags: PyTorch, AI/ML, Tuning
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Unveiling Performance Insights with PyTorch Profiler on an AMD GPU"
    "author": "Phillip Dang"
    "keywords": "PyTorch, Profiler, AMD, GPU, MI300, MI210, ROCm"
    "property=og:locale": "en_US"
---

# Unveiling performance insights with PyTorch Profiler on an AMD GPU

In the realm of machine learning, optimizing performance is often as crucial as refining model architectures. In this blog, we delve into the PyTorch Profiler, a handy tool designed to help peek under the hood of our PyTorch model and shed light on bottlenecks and inefficiencies. This blog will walk through the basics of how the PyTorch Profiler works and how to leverage it to make your models more efficient in an AMD GPU + ROCm system.

## What is PyTorch Profiler?

PyTorch Profiler is a performance analysis tool that enables developers to examine various aspects of model training and inference in PyTorch. It allows users to collect and analyze detailed profiling information, including GPU/CPU utilization, memory usage, and execution time for different operations within the model. By leveraging the PyTorch Profiler, developers can gain valuable insights into the runtime behavior of their models and identify potential optimization opportunities.

Using the PyTorch Profiler is straightforward and can be done in a few simple steps:

* Instrument Your Code: To start profiling your PyTorch code, you need to instrument it with profiling annotations. These annotations specify the regions of code or operations to profile. The PyTorch Profiler provides context managers and decorators for easy instrumentation.

* Configure Profiler Settings: Configure the profiler settings according to your profiling requirements. You can specify parameters such as the level of detail, profiling mode (e.g., CPU, GPU), and output format.

* Run Profiling: Once your code is instrumented and profiler settings are configured, run your PyTorch code as usual. The profiler will collect performance data during execution.

* Analyze Profiling Results: After execution, analyze the profiling results using the visualization tools provided by PyTorch Profiler. Explore timelines, flame graphs, and memory usage graphs to identify performance bottlenecks and optimization opportunities.

* Iterate and Optimize: Use the insights gained from profiling to iteratively optimize your code. Make targeted optimizations based on the profiling data, and re-run the profiler to assess the impact of your changes.

## Prerequisites

To follow along with this blog, you must have the following software:

* [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Linux OS

For a list of supported GPUs and OS, please refer to [this page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). For convenience and stability, we recommend you directly pull and run the rocm/pytorch Docker in your Linux system with the following code:

```sh
docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
           --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --name=olmo rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1 /bin/bash
```

To check your hardware and make sure that the system recognizes your GPU, run:

``` bash
! rocm-smi --showproductname
```

Your output should look like this:

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

Next, make sure PyTorch detects your GPU:

```python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

Your output should look like this:

```bash
number of GPUs: 1
['AMD Radeon Graphics']
```

## Instrument Your Code

### Libraries

Import the required libraries and modules we'll be using.

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity
```

### Model

Let's first create a very simple convolutional neural network model which we'll profile.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Data

Next, let's download a simple dataset.

```python
# Load CIFAR-10 dataset 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
```

### Training loop

Let's create a simple training loop with forward and backward passes, which we will profile. For the purpose of this blog, we'll profile the model's forward and backward pass for 200 batches instead of going through the entire dataset.

```python
# Function to train the model
def train(model, trainloader, criterion, optimizer, device, epochs=1):
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # exit after 200 batches 
            if i == 200:
                break
```

Also, let's write a utility function that sets up the optimizer and criterion, instantiates the model, and runs the actual profiling.

```python
# utility function for running the profiler 
def run_profiler(trainloader, model, profile_memory=False):
    device = 'cuda'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    with profile(activities=activities, record_shapes=True, profile_memory=profile_memory) as prof:
        with record_function("training"):
            train(model, trainloader, criterion, optimizer, device, epochs=1)

    if profile_memory == False:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
         print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

Profiling is as simple as wrapping the training loop with the profiler context manager.

## Run Profiling

With the model training loop and profiling utility function implemented, we're ready to use the PyTorch Profiler to profile the execution time and the memory consumption.

### Execution time profiling

Let's first look at the execution time of the training loop.

```python
model = SimpleCNN()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
run_profiler(trainloader, model)
```

The output looks like:

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               training        23.76%     360.249ms        71.31%        1.081s        1.081s       0.000us         0.00%      68.837ms      68.837ms             1  
autograd::engine::evaluate_function: ConvolutionBack...         0.15%       2.271ms         3.63%      55.037ms     136.908us       0.000us         0.00%      34.770ms      86.493us           402  
                             aten::convolution_backward         2.34%      35.480ms         3.34%      50.615ms     125.908us      18.366ms        16.60%      34.770ms      86.493us           402  
                                   ConvolutionBackward0         0.14%       2.151ms         3.46%      52.431ms     130.425us       0.000us         0.00%      34.486ms      85.786us           402  
    autograd::engine::evaluate_function: AddmmBackward0         0.33%       4.960ms         7.98%     120.946ms     300.861us       0.000us         0.00%      16.764ms      41.701us           402  
                                            aten::copy_         0.44%       6.674ms         2.08%      31.585ms      77.037us      15.762ms        14.25%      16.408ms      40.020us           410  
                                         aten::_to_copy         0.14%       2.079ms         2.31%      34.972ms      86.995us       0.000us         0.00%      16.306ms      40.562us           402  
                                              aten::sum         0.78%      11.818ms         0.93%      14.160ms      17.612us      14.723ms        13.31%      16.162ms      20.102us           804  
                                               aten::to         0.13%       2.031ms         2.36%      35.852ms      35.674us       0.000us         0.00%      15.783ms      15.704us          1005  
                                       CopyHostToDevice         0.00%       0.000us         0.00%       0.000us       0.000us      15.739ms        14.23%      15.739ms      39.152us           402  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.516s
Self CUDA time total: 110.639ms
```

Note the difference between self cpu time and cpu time. According to the [tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), "operators can call other operators, self cpu time excludes time spent in children operator calls, while total cpu time includes it. You can choose to sort by other metrics such as the self cpu time by passing sort_by="self_cpu_time_total" into the table call.

Let's now reduce our Convolution Neural Network (CNN) to a much simpler linear layer and run the profiler again. We expect to see a big reduction in CUDA time total.

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

model = SimpleNet()
run_profiler(trainloader, model)
```

Here's the output:

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               training        23.91%     192.128ms        84.59%     679.785ms     679.785ms       0.000us         0.00%      39.361ms      39.361ms             1  
                                           aten::linear         0.10%     768.000us         1.57%      12.605ms      62.711us       0.000us         0.00%      16.955ms      84.353us           201  
                                            aten::addmm         0.99%       7.943ms         1.28%      10.247ms      50.980us      16.955ms        37.52%      16.955ms      84.353us           201  
Cijk_Alik_Bljk_SB_MT64x64x32_MI32x32x2x1_SE_1LDSB0_A...         0.00%       0.000us         0.00%       0.000us       0.000us      15.556ms        34.42%      15.556ms      77.393us           201  
                                            aten::copy_         0.25%       2.028ms         3.07%      24.636ms      60.980us      14.614ms        32.34%      14.614ms      36.173us           404  
                                       CopyHostToDevice         0.00%       0.000us         0.00%       0.000us       0.000us      14.608ms        32.32%      14.608ms      36.338us           402  
                                         aten::_to_copy         0.27%       2.130ms         3.50%      28.122ms      69.955us       0.000us         0.00%      14.554ms      36.204us           402  
                                               aten::to         0.31%       2.460ms         3.61%      28.972ms      28.771us       0.000us         0.00%      13.586ms      13.492us          1007  
                                Optimizer.step#SGD.step         2.09%      16.809ms         2.94%      23.664ms     117.731us       0.000us         0.00%       5.557ms      27.647us           201  
    autograd::engine::evaluate_function: AddmmBackward0         0.28%       2.236ms         1.64%      13.185ms      65.597us       0.000us         0.00%       3.691ms      18.363us           201  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 803.604ms
Self CUDA time total: 45.193ms
```

As expected, there's a big reduction in total CUDA time (from 110.639ms to 45.193ms).

### Memory consumption profiling

We can also profile the amount of memory used by the model’s tensors that was allocated or released during the execution of the model’s operators.

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
model = SimpleCNN()
run_profiler(trainloader, model, profile_memory=True)
```

The output table looks like:

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        22.44%     224.849ms        22.74%     227.911ms       1.134ms       0.000us         0.00%       0.000us       0.000us      75.42 Mb      75.42 Mb           0 b           0 b           201  
                                            aten::empty         0.22%       2.204ms         0.22%       2.204ms       2.731us       0.000us         0.00%       0.000us       0.000us     390.64 Kb     390.64 Kb       3.79 Mb       3.79 Mb           807  
                                    aten::scalar_tensor         0.00%       9.000us         0.00%       9.000us       9.000us       0.000us         0.00%       0.000us       0.000us           8 b           8 b           0 b           0 b             1  
                                          aten::random_         0.00%      25.000us         0.00%      25.000us      12.500us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                                             aten::item         0.00%       9.000us         0.00%      13.000us       6.500us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                              aten::_local_scalar_dense         0.00%       4.000us         0.00%       4.000us       2.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                                          aten::resize_         0.00%       6.000us         0.00%       6.000us       0.002us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2615  
                                     aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                                      aten::resolve_neg         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                                               aten::to         0.22%       2.206ms         3.73%      37.335ms      37.149us       0.000us         0.00%      14.821ms      14.747us           0 b           0 b      75.47 Mb       2.63 Mb          1005  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.002s
Self CUDA time total: 109.871ms
```

If we're not happy with the memory consumption of the DataLoader, we can address the memory bottleneck by trying various strategies. These may include reducing the batch size, simplifying the model architecture, or using mixed precision training. Let's reduce the batch size from 32 to 4 and run the profiler again:

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
model = SimpleCNN()
run_profiler(trainloader, model, profile_memory=True)
```

The new output looks like:

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        13.45%     127.135ms        13.74%     129.910ms     646.318us       0.000us         0.00%       0.000us       0.000us       9.43 Mb       9.43 Mb           0 b           0 b           201  
                                            aten::empty         0.23%       2.193ms         0.23%       2.193ms       2.717us       0.000us         0.00%       0.000us       0.000us     390.64 Kb     390.64 Kb       3.87 Mb       3.87 Mb           807  
                                    aten::scalar_tensor         0.00%       9.000us         0.00%       9.000us       9.000us       0.000us         0.00%       0.000us       0.000us           8 b           8 b           0 b           0 b             1  
                                          aten::random_         0.00%      22.000us         0.00%      22.000us      11.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                                             aten::item         0.00%       6.000us         0.00%      10.000us       5.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                              aten::_local_scalar_dense         0.00%       4.000us         0.00%       4.000us       2.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             2  
                                          aten::resize_         0.00%       7.000us         0.00%       7.000us       0.003us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2615  
                                     aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                                      aten::resolve_neg         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                                               aten::to         0.21%       2.013ms         2.86%      27.042ms      26.907us       0.000us         0.00%       5.850ms       5.821us           0 b           0 b       9.52 Mb     481.50 Kb          1005  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 945.407ms
Self CUDA time total: 83.583ms
```

Here we significantly reduce the CPU memory required to load the data from 75.42 Mb to 9.43 Mb.

In this blog, we demonstrated that by analyzing the memory profiling and execution time, we can effectively improve the efficiency of our model training process. We encourage readers to experiment with different optimization strategies.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
