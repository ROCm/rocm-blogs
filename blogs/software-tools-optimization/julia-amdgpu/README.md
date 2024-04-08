---
blogpost: true
date: 8 Apr 2024
tags: Julia, GPGPU, HPC, AI/ML
category: Software tools & optimizations
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="">
  <meta name="keywords" content="Julia, AMDGPU, GPGPU, HPC, AI, ROCm">
</head>

# Programming AMD GPUs with Julia

[Julia](https://julialang.org/) is a high-level, general-purpose
dynamic programming language that automatically compiles to efficient
native code via LLVM, and supports multiple platforms.
With LLVM, comes the support for programming GPUs, including AMD GPUs.

[AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) Julia package
is the main entry point for programming AMD GPUs.
It provides the support for both high-level array programming
as well as low-level kernel programming
and integrates with rich Julia ecosystem for a unifying experience.

#### Quickstart

Installing AMDGPU.jl is as easy as adding a regular Julia package.
From the Julia REPL, execute (`]` symbol enters Pkg REPL mode, `Backspace` exits it):

```julia
] add AMDGPU
```

We do also require users to have a working ROCm installation, see the
[requirements](https://amdgpu.juliagpu.org/dev/#Requirements) section.

#### Example: element-wise addition

Once you have a working AMDGPU.jl installation,
you can import the package and start using it:

```julia
julia> using AMDGPU

julia> a = AMDGPU.rand(Float32, 1024); # ';' suppresses output in the REPL

julia> b = AMDGPU.rand(Float32, 1024);

julia> c = a .+ b; # '.' does function broadcasting for '+' operator

julia> sum(c)
2048
```

In the example above we import the package,
allocate `a` and `b` arrays on the GPU and fill them with `1`s,
compute the element-wise sum (notice `.+` which does operator broadcasting)
and finally compute the sum over the entire `c` array.

For the sake of it we can also compare the computation against CPU:

```julia
julia> ch = Array(a) .+ Array(b);

julia> Array(c) ≈ ch # '≈' can be replaced with 'isapprox(Array(c), ch)' to avoid Unicode
true
```

Here we first transfer `a` and `b` from GPU to CPU,
compute element-wise sum, store it in `ch`
and compare against `c` using `≈` operator
(Julia supports [Unicode input](https://docs.julialang.org/en/v1/manual/unicode-input/)).

#### Example: GPU kernel for element-wise addition

Alternatively, we can perform the same computation by writing our custom GPU kernel.

```julia
julia> function vadd!(c, a, b)
           i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
           if i ≤ length(c)
               @inbounds c[i] = a[i] + b[i]
           end
           return
       end
vadd! (generic function with 1 method)
```

Similar to C++/HIP kernels, Julia has support for AMD GPU-specific
[intrinsics](https://amdgpu.juliagpu.org/dev/kernel_programming/)
which can be used within kernels.
Here we compute index `i` of a single workitem in the same way as
with regular HIP kernels.
We then write the element-wise result of `a` and `b` into `c`
(`@inbounds` macro disables bounds-checking, which improves the performance).

> **Note:** since a kernel is a regular GPU function, all kernels should return nothing.

Launching a kernel can be done with a handy `@roc` macro:

```julia
julia> groupsize = 256;

julia> gridsize = cld(n, groupsize);

julia> @roc groupsize=groupsize gridsize=gridsize vadd!(c_d, a_d, b_d);

julia> Array(c) ≈ ch
true
```

> **Note:** all kernel launches are asynchronous, therefore users must
> explicitly synchronize with `AMDGPU.synchronize()` when needed.
> However, during GPU -> CPU transfer, AMDGPU.jl performs synchronization
> under-the-hood.

We can see that it is extremenly easy to do kernel programming with Julia
and kernels are not limited in functionality and being on par with HIP.

#### Integration with Julia ecosystem

AMDGPU.jl integrates ROCm libraries with Julia ecosystem offering unifying
experience, where there's almost no difference between
using arrays backed by AMDGPU.jl or by CPU.

E.g. [rocBLAS](https://github.com/ROCm/rocBLAS) is used for common
BLAS operations, and Julia's operators dispatch to them for efficiency.

```julia
julia> a = AMDGPU.rand(Float32, 1024, 1024);

julia> b = AMDGPU.rand(Float32, 1024, 1024);

julia> c = a * b; # dispatches to rocBLAS for matrix multiplication
```

[Flux.jl](https://fluxml.ai/) or [Lux.jl](https://lux.csail.mit.edu/stable/)
can be used to do machine learning supporting common building blocks:

```julia
julia> using AMDGPU, Flux;

julia> Flux.gpu_backend!("AMDGPU");

julia> model = Conv((3, 3), 3 => 7, relu; bias=false)

julia> x = AMDGPU.rand(Float32, (100, 100, 3, 50)); # random images in WxHxCxB shape.

julia> y = model(x) # dispatches to MIOpen for convolution
```

[Zygote.jl](https://github.com/FluxML/Zygote.jl) can be used to compute gradients
given any Julia function:

```julia
julia> θ = AMDGPU.rand(Float32, 16, 16);

julia> x = AMDGPU.rand(Float32, 16, 16);

julia> loss, grads = Zygote.withgradient(θ) do θ
           sum(θ * x)
       end;
```

And much more!

#### Performance

Provided that you are using efficient constructs, the performance of
Julia GPU code is on par with C++ and sometimes even exceeding it.

<img src="data/amdgpu-performance.png" width="80%">

Performance comparison of a memcopy and 2D diffusion kernel
implemented in Julia with AMDGPU.jl and executed on a MI250x GPU.

For performance inspection, [profiling](https://amdgpu.juliagpu.org/dev/profiling/)
can be used to get a timeline view of the entire program.

And `@device_code @roc ...` macro on a per-kernel basis to dump
different intermediate representations (unoptimized LLVM IR, optimized LLVM IR, assembly).

#### Applications & Libraries

With rich ecosystem integration it is extremely easy to implement applications,
here are just a few of them:

- [Nerf.jl](https://github.com/JuliaNeuralGraphics/Nerf.jl):
    [Instant-NGP](https://nvlabs.github.io/instant-ngp/) implementation in native Julia.
- [Whisper.jl](https://github.com/JuliaNeuralGraphics/Whisper.jl):
    Popular speech-to-text model.
- [Diffusers.jl](https://github.com/JuliaNeuralGraphics/Diffusers.jl):
    Stable Diffusion 1.5.
- [GPU4GEO](https://ptsolvers.github.io/GPU4GEO/stream/):
    Modelling of ice motion using LUMI supercomputer targeting LUMI-G's AMD MI250x GPUs.
