---
blogpost: true
blog_title: "Towards Feature Complete Triton Support in JAX-Triton"
date: "08 Jul 2026"
author: "Aleksei Rechinskii, Jehandad Khan"
thumbnail: 'jax-triton-updated.jpg'
tags: "AI/ML, Scientific Computing, HPC, JAX"
category: "Applications & models"
target_audience: "Engineers working with JAX ecosystem"
key_value_propositions: "Introduction of major rework and compatibility update of JAX-Triton, an integration layer to run Triton/Gluon kernels seamlessly in JAX."
language: English
myst:
    html_meta:
        "author": "Aleksei Rechinskii, Jehandad Khan"
        "description lang=en": "Learn what new features were added to JAX-Triton and how that could help you write or reuse more efficient and readable GPU kernels in JAX."
        "keywords": "JAX-Triton, JAX, Triton, Gluon, AI/ML, HPC, ML, AI"
        "vertical": "AI, HPC, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Aleksei Rechinskii, Jehandad Khan"
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

# Towards Feature Complete Triton Support in JAX-Triton

In this blog we'll explore recent improvements to JAX-Triton project and learn about new features available at the AMD fork.

The promise of JAX-Triton has always been simple: let Triton GPU kernels run seamlessly inside JAX. Until recently, however, JAX-Triton fell short of that promise, and the ideal use case - taking an existing Triton kernel from another framework, such as PyTorch, and running it in JAX - worked only for a narrow set of specifically written kernels.

The compatibility update that AMD contributed to JAX-Triton is a major leap forward. It supports most Triton features and moves the project from an experimental prototype toward maturity. The update does not yet fully cover the ideal use case, but today you can run virtually any Triton or Gluon kernel inside JAX, either directly or after minor cosmetic changes. In practice, this means that in many cases you can simply copy-paste or import a kernel from another framework into JAX and run it efficiently as part of a compiled compute graph.

## What Has Changed in JAX-Triton

By version 0.5.1, JAX-Triton has accumulated the following changes and improvements:

- Support for the Gluon dialect
- A special `@jt.kernel` decorator for clear and concise Triton-like launch syntax
- Proper backend object initialization and the ability for a kernel to be aware of initialization options
- Feature-complete kernel launcher argument types
- Simplified and clearer specification of output and input-output parameters

### Gluon Dialect in JAX-Triton

Gluon is a low-level dialect of Triton that grants direct access to a specific platform's hardware features. It lets kernel authors extract the most performance from a GPU without resorting to pure assembly.

JAX-Triton has fully supported Gluon since v0.4.0. This version has already been merged upstream. However, some important updates to the Gluon compilation pipeline are not yet upstream - nor are most of the features described below - and are currently available only through the AMD fork at [ROCm/jax-triton](https://github.com/ROCm/jax-triton).

### `@jt.kernel` Decorator

Originally, the only way to launch a Triton or Gluon kernel in JAX was through JAX-Triton's `triton_call()` service function. While this was in line with how, for example, JAX Pallas-based kernels are launched, this approach is verbose and hard to read. For example:

```py
import jax_triton as jt
import triton

# kernel definition
@triton.jit
def kernel_func(x_ptr, y_ptr, z_ptr, ret_ptr):
	...

...
# kernel launch site
ret1 = jt.triton_call(
	x1,
	y1,
	z1,
	kernel=kernel_func,
	grid=grid,
	out_shape=x1,
)

ret2 = jt.triton_call(
	x2,
	y2,
	z2,
	kernel=kernel_func,
	grid=grid,
	out_shape=x2,
)
```

Every kernel launch required a direct `triton_call()` invocation with several mandatory service arguments: `kernel=` to specify the kernel to launch, `grid=` to set the threading structure, and `out_shape=` to describe the kernel's outputs. In effect, this made the kernel that does the actual work a second-class citizen, hiding it deep in the argument list while pushing the purely mechanical `triton_call()` to the front. Imagine a sequence of such calls throughout a project, each beginning with `triton_call()` - it is hardly obvious which code actually processes the data.

Triton solved this problem long ago by making clever use of Python's special methods, which tuck the kernel launch code out of sight and highlight the kernel where it matters. JAX-Triton now has similar functionality through the `@jt.kernel` decorator, which hides the service calls to `triton_call()` and makes the visible code much cleaner (for conciseness, subsequent code snippets are assumed to inherit the imports defined earlier):

```py
# kernel definition
@jt.kernel
@triton.jit
def kernel_func(x_ptr, y_ptr, z_ptr, ret_ptr):
	...

...
# kernel launch site
ret1 = kernel_func[grid](x1, y1, z1, out_shape=x1)
ret2 = kernel_func[grid](x2, y2, z2, out_shape=x1)
```

This is the same familiar, clear, and concise Triton-native syntax, reminiscent of the HIP or CUDA way of invoking a kernel. It promotes the kernel launch to a first-class citizen of the code.

An additional benefit of `@jt.kernel` is that it can capture any key-value arguments and pass them to `triton_call()`. This is useful for keeping a kernel's inherent properties tied to its definition instead of repeating them at every call site. A typical example is designating kernel output parameters through the new `out_names=` argument, which enables passing non-constexpr arguments as key-value `**kwargs` pairs (more about it later):

```py
import triton.language as tl

@jt.kernel(out_names="ret_ptr")
@triton.jit
def kernel_func(x_ptr, y_ptr, scalar, ret_ptr, p1 : tl.constexpr = 0):
	...

ret = kernel_func[grid](x, y, scalar=z, out_shape=x)
```

This further improves readability of the code and relaxes constraints on the launcher, making it similar to invoking just another Python callable.

### Correct Backend Initialization

A backend in Triton is an object that represents the platform for which code is compiled and on which it runs. At least two backends are available, one for AMD and one for NVIDIA. Each backend offers about two dozen initialization options that let users tailor a kernel's behavior to their specific needs.

Triton handles these options transparently through key-value arguments passed to the kernel launcher. This key-value space is shared between the arguments passed to the kernel and the backend initialization options. During the launch phase, Triton routes each argument to the kernel, to the backend, or to both -  when the argument's name is a backend option that also appears in the kernel's signature. This lets a kernel access anything it needs and, for example, branch on the value of the `num_warps` backend option.

Originally, JAX-Triton supported only five backend initialization options: `num_warps`, `num_stages`, `num_ctas`, `enable_fp_fusion`, and `debug`. Nothing else was possible - for example, `extern_libs`, which specifies the location of external compiled-kernel libraries. Worse, the namespaces for backend options and kernel arguments were separate, so a kernel could not learn its own launch parameters. JAX-Triton v0.5.0 fixes all of this, and it now works the same way as upstream Triton.

### Full Support for Every Triton Argument Type

Since v0.5.0, JAX-Triton is at parity with upstream Triton for argument types. For example, these types are now fully supported: tuples (including deeply nested ones), callables (other kernels), strings, explicit `tl.constexpr` values, `None` objects, and so on. It also supports passing non-constexpr arguments as key-value pairs (which requires `out_names=`, or the dictionary form of `out_shape=`).

This required a full rework of how a kernel is lowered through JAX's Primitive system to the JAX kernel launcher. At the heart of the change was a re-implementation of a kernel specialization pipeline. But what is kernel specialization?

Triton can compile and run different versions of a kernel binary depending on the arguments passed to the launcher. This process - producing or selecting a kernel binary based on its arguments - is called "kernel specialization," and it is an important technique for performance optimization.

There are two canonical use cases. The first is the numeric value `1`, the typical stride of the innermost tensor dimension. The second is the `None` object, whose use is typically guarded by a check such as `if x is not None`. Producing dedicated kernel binaries with these special values baked into the code can enable highly efficient vectorized loads and stores in the first case and eliminate dead code in the second. When necessary, you can explicitly ask Triton to specialize a kernel for a specific scalar argument by wrapping its value in `tl.constexpr()`. For example:

```py
@jt.kernel(out_names="ret_ptr")
@triton.jit
def kernel_func(x_ptr, stride_x, ret_ptr):
	...
	
# run a generic binary where stride_x = 4 is passed as a runtime value
ret1 = kernel_func[grid](x, stride_x=4, out_shape=x)

# run a specialized binary with baked in stride_x = 4
ret2 = kernel_func[grid](x, stride_x=tl.constexpr(4), out_shape=x)  
```

Passing a callable to a kernel launcher statement is another special case for kernel specialization. Modern GPUs support indirect function calls through pointers, but this is generally discouraged for performance reasons: it can easily cause wavefront divergence and adds significant complexity in handling the registers modified by the called function. Inlining all invoked functions instead typically produces a far more efficient binary, and that suits kernels better.

JAX-Triton now fully delegates specialization to the original Triton code, so it is feature-complete with respect to what Triton supports.

As the description above implies, the specialization pipeline is also where Triton applies all of the type annotations attached to a kernel argument in the signature, such as those marking an argument as `constexpr` or giving it a specific data type like `tl.int8`. It is also where Triton enforces the kernel flags set through `triton.jit()`, such as `do_not_specialize=`. One might therefore expect a fairly branch-heavy routine that traverses the kernel's signature and matches its parameters to the `*args, **kwargs` passed to the launcher - for every kernel launch. Such code does exist, but it is ingeniously optimized to do all of the branching just once per kernel and, at runtime (that is, for each launch), to execute only a linear sequence of the operations that the launch strictly requires.

To accomplish this, Triton parses the kernel signature only once, infers all the parameters and their annotations, and then dynamically builds a function tailored to that kernel. The generated function:

1. uses the parameter list to split the incoming `*args` and `**kwargs` into a set of key-value pairs that match the signature's parameters (later returned in a dictionary) and a dictionary of everything else (presumably backend options); and
2. runs an efficient C++ algorithm that computes a specialization description of an argument from its type and value - but only for arguments not explicitly marked as `constexpr`. Which arguments require this step is decided when the function is created, based on signature type annotations, so each launch runs the absolute minimum of code.

An elegant design.

### Improvements in Handling Output-Only and Input-Output Arguments

#### Why Output Argument Specification Matters

In contrast with PyTorch's eager mode, JAX compiles the whole computational graph to binary code with the XLA compiler. Compiled code is not merely faster than interpreted code; the compilation process itself lets the compiler exploit properties of the graph to optimize operations and reduce the surrounding boilerplate. One of the key features that enable many of these optimizations is the static single assignment (SSA) representation of the graph, in which every variable is assigned exactly once. A consequence is that all tensors in JAX are immutable by default, and every operation produces a new tensor (at least within the compiler's representation). Under the hood, XLA arranges operations so that it need not materialize most of the intermediate tensors, and the net effect for a compiled model is highly beneficial.

JAX's tensor immutability requirement has a major implication for operations that are opaque to XLA, such as custom calls, which inject an external operation into the compute graph. One such custom call is - that's right - a Triton kernel call. To produce correct binary code and arrange the correct data transfers between host and devices, XLA needs to know which tensors a custom call produces and which tensors it modifies.

None of this matters in PyTorch: tensors there are mutable and the user moves data between host and devices explicitly. In graph mode (`torch.compile()`) that isn't a concern either: PyTorch statically analyzes the Triton kernel's own IR (TTIR) to automatically discover which tensor arguments are written to. It walks backward from store/atomic operations to the kernel's pointer parameters. This is the automatic equivalent of JAX-Triton's manual `input_output_aliases` / `out_names`, and automating that remains one of the development directions for JAX-Triton.

To clarify the terminology: when we talk about kernel output-only or input-output parameters or arguments, we consider it from the point of view of the code launching the kernel. The launcher must ensure that all required data is copied from the host to the device and passed to the kernel inputs, and that all modified data is passed back to the host (or to the next consumer) correctly. Hence, the term "output-only"/"purely output" doesn't mean that a kernel cannot read from the associated buffer. It merely means that information cannot flow through that argument into the kernel.

#### The Old Convention

JAX-Triton defines kernel outputs through several specialized arguments to the `triton_call()` launcher. Originally, these were:

- `out_shape=`: a sequence of ShapeDtype-like objects that describe the output tensors; JAX-Triton implicitly mapped these tensor buffers to the last non-constexpr parameters of the kernel signature.
- `input_output_aliases=`: a dictionary that describes input-output kernel arguments by mapping the raw tensor indices in `*args` to the raw indices of output tensors in `out_shape=`. JAX and XLA call such tensors "aliased" because, under the hood, XLA reuses the tensor's data buffer even though it generates a new tensor object to represent the modified buffer.
- `zeroed_outputs=`: a sequence of output-tensor indices, or a callable that returns such a sequence, that designates the output tensors XLA must explicitly zero before the kernel starts. This is useful, for example, for kernels that perform sparse updates (the feature is needed because XLA may reuse the underlying memory between calls and, by default, does not spend time zeroing it before reuse).

This approach was good enough for a prototype, and it worked, but such code was puzzling to read and unnecessarily exhausting and error-prone to write or modify. The convention was too restrictive. On top of that, once the update added support for tuples and tensors in key-value arguments, asking users to compute and track the raw indices of the tensors they want to alias no longer seemed reasonable.

#### Improvements

Since v0.5.0, JAX-Triton has completely rewritten the handling of these `triton_call()` arguments. Backward compatibility - with the exception of zeroing aliased tensors - is preserved, and an entirely new way to reference tensors by their paths in the kernel's signature has been added, which makes specifying the kernel's outputs clearer and simpler.

##### Changes to `out_shape=` and Purely Output Parameters

`out_shape=` can now be a dictionary that maps a kernel's output parameter names or indices to a single ShapeDtype-like object or to an ordered, possibly nested, sequence of such objects. The nested structure of the dictionary's values defines the structure of the output argument.

For example, `out_shape={"a": (x,y), "b": z}` declares that the kernel has two output parameters: the parameter named `a` is a tuple of two arrays that borrow their shapes and dtypes from objects `x` and `y`, and the parameter `b` is a single `z`-like array.

Another way to specify purely output parameters is the new `out_names=` parameter of `triton_call()`. It accepts a string or a sequence of strings that correspond to output parameter names in the kernel signature. Think of it as a way to set `out_shape.keys()` without using the dictionary form of `out_shape=`. The most convenient use case is to keep everything related to the kernel itself (as opposed to things needed for launching it) where the kernel is defined - that is, to use it together with the `@jt.kernel(out_names=...)` decorator.

Using `out_names=` or the dictionary form of `out_shape=` does more than make the code easier to read. Telling JAX which kernel parameters serve as outputs is what lets it correctly match non-constexpr arguments inside `**kwargs`. Without that information, the old constraint - "all non-constexpr arguments must be passed positionally in `*args`" - has to remain in force. That constraint stems from a requirement of JAX's Custom Call API, and the requirement still affects every kernel signature, even with `out_names=`: purely output parameters must come last among the non-constexpr parameters, because constexpr parameters are baked into the kernel code and are never passed to the Custom Call API. At least the requirement to always use positional arguments for the launcher can now be dropped, which helps with readability and complex cases.

##### Changes to `input_output_aliases=` Parameter

Specifying an aliased tensor as a mapping between a raw index in `out_shape=` and a raw index in `*args` is a major inconvenience with `input_output_aliases=`.  Additionally, you have to specify a tensor's shape and dtype twice: the input tensor (referenced by the dictionary key) already carries that information implicitly, yet you must also provide a ShapeDtype-like object in `out_shape=` (referenced by the dictionary value).

With these issues in mind, `input_output_aliases=` now also supports a natural way of specifying aliased arguments as a scalar or a sequence of input-array coordinates in the kernel's signature coordinate space. The nested structure of the sequence describes the structure of the corresponding output argument. For example:

- `input_output_aliases="in_out_ptr"` indicates that the tensor bound to the `in_out_ptr` kernel parameter is aliased and its buffer is reused by a tensor returned from `triton_call()`.
- `input_output_aliases=["out_ptr", [(2,0), "out_ptr2"]]` describes three aliased (input-output) buffers:
  - the first is the input array passed to the `out_ptr` kernel parameter,
  - the second is the array at index 0 of the tuple passed to the third positional parameter, and
  - the third is the input array passed to the `out_ptr2` kernel parameter.
  - These three buffers (assuming each argument is a single array) are returned from `triton_call()` as a tuple of two elements: element `[0]` is a new array that wraps the same buffer used for the `out_ptr` kernel parameter, and element `[1]` is a tuple of two new arrays that wrap the other two buffers.

If a referenced input argument is not an array but a possibly nested tuple of arrays, all of its inner arrays are considered aliased and are returned as a flat tuple of arrays by the call.

Note that you can also use tuples instead of lists for the sequences in `input_output_aliases=`. Just remember that if a tuple's first element is not itself an iterable, the tuple is always treated as a coordinate descriptor, which may not be what you want. For example, `input_output_aliases=(0,1)` is always read as a reference to the array at the second position of a tuple passed to the first kernel parameter, whereas `input_output_aliases=[0,1]` references two arrays passed to the first and second kernel parameters, respectively. For clarity, therefore, it's better to use tuples only when describing a single input-array coordinate.

##### Changes to `zeroed_outputs=` Parameter

Like the sequence form of `input_output_aliases=`, `zeroed_outputs=` now accepts a sequence of kernel-signature coordinates for the output arguments to zero before launching the kernel.

However, `zeroed_outputs=` also carries a breaking change: as of v0.5.0, you can no longer trigger the zeroing of aliased (input-output) buffers. The rationale is simple. Input-output arguments are meant to pass information to the kernel and back. Zeroing a buffer before the launch lets no information flow into the kernel and effectively turns an input-output parameter into a purely output one - but not for free, because XLA does not handle this special case and still schedules a host-to-device copy before the launch, copying memory to a destination that will be zeroed right afterward.

### Breaking Changes in v0.5.0

We have already discussed one breaking change: `zeroed_outputs=` no longer triggers the zeroing of input-output arguments.

The other breaking change, likely far less consequential, is that JAX-Triton now lowers Python `float` scalars as `fp32`, following upstream Triton, instead of the old `fp64` convention. This should have no visible effect, since Triton has always produced binaries on that implicit assumption.

### Putting it All Together

Thanks to AMD's revamp of JAX-Triton, you can finally write or easily port genuinely complex, interacting kernels that remain easy to read and understand. Consider the following example, which showcases many of the new features:

```python
import jax
import jax.numpy as jnp
from jax import random
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import NamedTuple
import time

# A helper class to represent a closure passed to a kernel.
class Function(NamedTuple):
  fn: tl.constexpr
  captured: tuple

# let's do some trigonometry
@triton.jit
def func1(x_ptr, y_ptr: tl.const, SIZE: tl.constexpr):
  off = tl.arange(0, SIZE)
  x = tl.load(x_ptr + off)
  y = tl.load(y_ptr + off)
  x1 = libdevice.sin(x)
  x2 = libdevice.cos(x)
  z = x1 * x1 + x2 * x2
  tl.store(x_ptr + off, z)
  y = libdevice.asin(y) + libdevice.acos(y)
  return z, libdevice.floor(2 * y)

# and invoke a function "dynamically" by its name
@triton.jit
def floor_of_func(values, FUNC_NAME: tl.constexpr):
  return libdevice.floor(getattr(libdevice, FUNC_NAME)(values))

# and iterate over a tuple
@triton.jit
def aggregate(Ptrs):
  z = tl.zeros([], tl.float32)
  for i in tl.static_range(len(Ptrs)):
    z += Ptrs[i]
  return z

# one kernel to rule them all
@jt.kernel(out_names="out_ptr")
@triton.jit
def kernel(capture, out_ptr, SIZE: tl.constexpr, FUNC_NAME: tl.constexpr):
  off = tl.arange(0, SIZE)
  t1, t2 = capture.fn(*capture.captured, SIZE=SIZE)
  t3 = floor_of_func(t1, FUNC_NAME=FUNC_NAME)
  t4 = t2 * t3
  result = aggregate((t4, t4 * t4)).to(tl.int32)
  tl.store(out_ptr + off, result)

size = 8
k1, k2 = random.split(random.key(time.perf_counter_ns()), 2)
x = random.uniform(k1, (size,), dtype=jnp.float32)
y = random.uniform(k2, (size,), dtype=jnp.float32)

fn = Function(func1, (x, y))
out, x = kernel[(1,)](
  fn,  # essentially a tuple of (func_name, (2 arrays in a subtuple))
  SIZE=size,
  FUNC_NAME="exp",  # the name of a function to invoke from libdevice
  out_shape=jax.ShapeDtypeStruct(size, dtype=jnp.int32),
  input_output_aliases=("capture", 1, 0),  # a path inside `capture` argument, 0th
  # element of a 1st subtuple, i.e. `x`.
)

np.testing.assert_array_equal(out, jnp.full((size,), 42, dtype=jnp.int32))
assert out.dtype == jnp.int32
np.testing.assert_allclose(x, jnp.full((size,), 1.0, dtype=jnp.float32), rtol=5e-7)
assert x.dtype == jnp.float32
```

## Current State of JAX-Triton

While JAX-Triton has received a set of significant updates, it still does not fully support everything upstream Triton does. Keep the following known limitations in mind when working with JAX-Triton:

1. Because of restrictions of the JAX Custom Call API, a kernel's purely output parameters should come last in the signature (ignoring constexpr parameters). Some relaxations of this rule might exist, but it is simpler to always follow it by reordering the parameters.

2. Be aware that a pattern that is harmless in upstream Triton - preallocating an empty buffer and passing it as an input-output buffer only to write to it - may incur the cost of a redundant host-to-device copy. Use purely output arguments instead.

3. The autotuner and heuristics may have gaps. For example, the `Config.pre_hook` and `Config.post_hook` hooks are not implemented, and custom `do_bench`, `perf_model`, `early_config_prune`, and `top_k` settings in `triton.autotune` may not behave correctly.

4. Triton's Python-runtime features - such as `JITFunction.warmup`, `.preload`, and Interpreter mode - are not supported.

5. JAX-Triton may not work well on systems with several different GPU models.

6. JAX-level transformations such as `jax.grad`, `jax.jvp`, and `jax.vjp` are not supported on kernels.

## How to Install JAX-Triton

To get the latest features, install JAX-Triton from the AMD fork:

```bash
pip install triton==3.6.0  # more on that below
git clone https://github.com/ROCm/jax-triton.git
cd jax-triton
pip install .
```

At the time of writing, compatibility of JAX-Triton with the newly released Triton 3.7.0 wasn't fully verified, so it's recommended to use Triton 3.6.0 instead.

## Acknowledgements

The author is grateful to Google for developing the initial version of the project and for providing high-quality, thoughtful, and helpful reviews of upstream submissions.

## Summary

In this blog we have explored recent contributions to JAX-Triton project that drove it way forward towards feature-complete parity with upstream Triton available in PyTorch ecosystem. With most Triton features supported, JAX-Triton now enables use of virtually any Triton of Gluon kernels in JAX either directly or after minor cosmetic changes.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, ROCm and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved
