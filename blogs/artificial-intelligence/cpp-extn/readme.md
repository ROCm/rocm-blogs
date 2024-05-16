---
blogpost: true
date: 16 Apr, 2024
author: Vara Lakshmi Bayanagari
tags: C++, PyTorch, AI/ML
category: Applications & models
language: English

myst:
  html_meta:
    "description lang=en": "PyTorch C++ Extension on AMD hardware"
    "keywords": "Custom cpp extension, AMD GPU, Pytorch, C++, Programming Languages, Custom C++ Extension"
    "property=og:locale": "en_US"
---

# PyTorch C++ Extension on AMD GPU

This blog demonstrates how to use the PyTorch C++ extension with an example and discusses its advantages over regular PyTorch modules. The experiments were carried out on AMD GPUs and ROCm 5.7.0 software. For more information about supported GPUs and operating systems, see [System Requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

## Introduction

PyTorch has become the go-to development framework for both ML practitioners and enthusiasts due to its ease of use and wide availability of models. PyTorch also allows you to easily customize models by creating a derived class of ```torch.nn.Module```, which reduces the need for repetitive code related to differentiability. Simply put, PyTorch provides extensive support.

But what if you want to speed up your custom model? PyTorch provides C++ extensions to accelerate your workload. There are advantages to these extensions:

- They provide a fast C++ test bench for out-of-source operations (the ones not available in PyTorch) and easily integrate into PyTorch modules.
- They compile models quickly, both on CPU and GPU, with only one additional build file to compile the C++ module.

The [Custom C++ and CUDA Extensions tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html#) by Peter Goldsborough at PyTorch explains how PyTorch C++ extensions decrease the compilation time on a model. PyTorch is built on a C++ backend, enabling fast computing operations. However, the way in which the PyTorch C++ extension is built is different from that of PyTorch itself. You can include PyTorch's library (```torch.h```) in your C++ file to fully utilize PyTorch's ```tensor```and ```Variable``` interfaces while utilizing native C++ libraries such as ```iostream```. The code snippet below is an example of using the C++ extension taken from [the PyTorch tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html#):

```cpp
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
```

The *d_sigmoid* function computes the derivative of the sigmoid function and is used in backward pass implementations. You can see that the implementation is an extension of PyTorch written in C++. For example, the data type of the return value of the ```d_sigmoid``` function, as well as the function parameter ```z``` is ```torch::Tensor```. This is possible because of the ```torch/extension.h``` header, which includes the famous ```ATen``` tensor computation library. Let's now see how C++ extensions can be used to speed up a program by looking at a complete example.

## Implementation

In this section we'll test a generic MLP network with one hidden layer in both native PyTorch and PyTorch C++. The source code is inspired by [Peter's example](https://pytorch.org/tutorials/advanced/cpp_extension.html#) of LLTM (Long Long Term Model) model and we establish a similar flow for our MLP model.

Let's now implement *mlp_forward* and *mlp_backward* functions in C++. PyTorch has ```torch.autograd.Function``` that implements backward passes under the hood. PyTorch C++ extension requires us to define the backward pass in C++ and later bind them to PyTorch's `autograd` function.

As shown below, *mlp_forward* function carries out the same computations as the one in the [MLP](./src/mlp_train.py#L8) Python class and the *mlp_backward* function implements the derivatives of the output with respect to the input. If you're interested in understanding the mathematical derivations, view the backward pass equations defined in *Back Propagation* section of [Prof. Tony Jebara's ML slides](https://www.cs.columbia.edu/~jebara/4771/notes/class4x.pdf). He represents an MLP network with two hidden layers and details the differential equations for back propagation. For simplicity, we consider only one hidden layer in our example. Be aware that writing custom differential equations in C++ is a challenging task and requires expertise in the field.

```cpp
#include <torch/extension.h>
#include <vector>
#include <iostream>


torch::Tensor mlp_forward(  
    torch::Tensor input,  
    torch::Tensor hidden_weights,  
    torch::Tensor hidden_bias,  
    torch::Tensor output_weights,  
    torch::Tensor output_bias) {  
  // Compute the input/hidden layer  
  auto hidden = torch::addmm(hidden_bias, input, hidden_weights.t());  
  hidden = torch::relu(hidden);  
  
  // Compute the output layer  
  auto output = torch::addmm(output_bias, hidden, output_weights.t());   
  
  // Return the output  
  return output;  
    
}  
  
std::vector<torch::Tensor> mlp_backward(  
    torch::Tensor input,  
    torch::Tensor hidden_weights,  
    torch::Tensor hidden_bias,  
    torch::Tensor output_weights,  
    torch::Tensor output_bias,
    torch::Tensor grad_output) {  
  
  // Compute the input/hidden layer
  auto hidden = torch::addmm(hidden_bias, input, hidden_weights.t());
  hidden = torch::relu(hidden);  
  // Compute the output layer  
  auto output = torch::addmm(output_bias, hidden, output_weights.t());  
  // Compute the gradients for output layer
  auto grad_output_weights = torch::mm(grad_output.t(), hidden);
  auto grad_output_bias = torch::sum(grad_output, /*dim=*/0).unsqueeze(0); 
  // Compute the gradients for input/hidden layer using chain rule
  auto grad_hidden = torch::mm(grad_output, output_weights);
  // grad_hidden = grad_hidden
  auto grad_hidden_weights = torch::mm(grad_hidden.t(), input);
  auto grad_hidden_bias = torch::sum(grad_hidden, /*dim=*/0).unsqueeze(0);
  // Compute the gradients for input
  auto grad_input = torch::mm(grad_hidden , hidden_weights);
    
  // Return the gradients  
  return {grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias};  
}
```

Let's wrap the C++ implementation using ```ATen's``` Python binding function as shown below. ```PYBIND11_MODULE``` maps the keyword *forward* to the pointer of the ```mlp_forward``` function and *backward* to the ```mlp_backward``` function. This binds the C++ implementations to the Python definitions. The macro ```TORCH_EXTENSION_NAME``` will be defined as the name passed in the setup.py file during build time.

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "MLP forward");
  m.def("backward", &mlp_backward, "MLP backward");
}
```

Next, write a `setup.py` file that imports the ```setuptools``` library to help compile the C++ code. To build and install your C++ extension, run the ```python setup.py install``` command. This command creates all the build files relevant to the ```mlp.cpp``` file and provides a module ```mlp_cpp``` that can be imported into the PyTorch modules.  

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mlp_cpp',
    ext_modules=[
        CppExtension('mlp_cpp', ['mlp.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

Now, let's prepare a PyTorch's MLP class powered by C++ functions with the help of ```torch.nn.Module``` and ```torch.autograd.Function```. This enables the use of C++ functions in a manner that is more native to PyTorch. In the following example, the forward function of the *MLP* class points to the forward function of `MLPFunction`, which is directed to the C++'s `mlp_forward` function. This flow of information establishes a pipeline that works seamlessly as a regular PyTorch model.

```python
import math
from torch import nn
from torch.autograd import Function
import torch

import mlp_cpp

torch.manual_seed(42)

class MLPFunction(Function):
    @staticmethod
    def forward(ctx, input, hidden_weights, hidden_bias, output_weights, output_bias):
        output = mlp_cpp.forward(input, hidden_weights, hidden_bias, output_weights, output_bias)
        variables = [input, hidden_weights, hidden_bias, output_weights, output_bias]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias =  mlp_cpp.backward( *ctx.saved_variables, grad_output)
        return grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias


class MLP(nn.Module):
    def __init__(self, input_features=5, hidden_features=15):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.hidden_weights = nn.Parameter(torch.rand(hidden_features,input_features))
        self.hidden_bias = nn.Parameter(torch.rand(1, hidden_features))
        self.output_weights = nn.Parameter(torch.rand(1,hidden_features))
        self.output_bias = nn.Parameter(torch.rand(1, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.001
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return MLPFunction.apply(input, self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias)
```

Now, let's use [trainer.py](.src/trainer.py) to test the speed of the forward and backward computations and compare the native PyTorch implementation with the C++ implementation.

Note: In some cases you may have to run the program multiple times before benchmarking the results to see expected trends in the speed-up.

```bash
python trainer.py py

Forward: 0.102 milliseconds (ms) | Backward 0.223 milliseconds (ms)
```

```bash
python trainer.py cpp

Forward: 0.094 milliseconds (ms) | Backward 0.258 milliseconds (ms)
```

We can see that for 100,000 runs, the average time taken for a forward pass by the native PyTorch model is 0.102 ms, whereas for the C++ model it's only 0.0904 ms (an improvement of ~8%). If the backward pass doesn't follow the same trend, its implementation might not be optimized. As mentioned previously, it's a challenging task and requires expertise to translate mathematical differential equations into C++ code. As the complexity and size of the model increase, we might see a larger difference between both experiments, as noted in Peter's LLTM example. Despite a few implementation challenges, C++ is proving to be faster and also easier to integrate with PyTorch.

The complete code can be found in the [src](./src) folder, which has the following structure:

- [setup.py](./src/setup.py) - Build file that compiles C++ module
- [mlp.cpp](./src/mlp.cpp) - C++ module
- [mlp_cpp_train.py](./src/mlp_cpp_train.py) - Applying C++ extension to a PyTorch model
- [mlp_train.py](./src/mlp_train.py) - Native PyTorch implementation for comparison
- [trainer.py](./src/trainer.py) - Trainer file to test PyTorch vs. PyTorch's C++ extension.

## Conclusion

This blog walks you through an example of using custom PyTorch C++ extensions. We observed that custom C++ extensions improved a model's performance compared to a native PyTorch implementation. These extensions are easy to implement and can easily plug into PyTorch modules with the minimal overhead of pre-compilation.

Moreover, PyTorch's ```Aten``` library provides us with enormous functionalities that can be imported into the C++ module and mimicks PyTorch-like code. Overall, PyTorch C++ extensions are easy to implement and a good option for testing the performance of custom operations, both on CPU and GPU.

## Acknowledgements

We would like to acknowledge and thank Peter Goldsborough for an extremely well written [article.](https://pytorch.org/tutorials/advanced/cpp_extension.html#)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
