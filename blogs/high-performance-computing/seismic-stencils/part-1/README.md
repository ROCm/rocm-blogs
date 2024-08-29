---
blogpost: true
date: 29 Aug 2024
author: Justin Chang, Ossian O'Reilly
tags: HPC, Memory, Performance, Profiling, Scientific Computing
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Seismic Stencil Codes - Part 1"
    "keywords": "HPC, finite difference, Seismic, PDE, MI250, performance"
    "property=og:locale": "en_US"
---

# Seismic stencil codes - part 1

<span style="font-size:0.7em;">12 Aug, 2024 by {hoverxref}`Justin Chang<justchan>` and {hoverxref}`Ossian O'Reilly<ossiorei>`. </span>

Seismic workloads in the HPC space have a long history of being powered
by high-order finite difference methods on structured grids. This trend
continues to this day. While there is a rich body of work of
optimization techniques and strategies, there is currently no known
\"one size fits all\"-approach that works across the broad spectrum of
physical models. Many tradeoffs impact the overall
performance of propagation algorithms that depend mainly on discretization, and hardware
characteristics. GPU acceleration provides a cost-effective way to
meet the massive demand for processing the evergrowing seismic data volumes.
A distinguishing feature of seismic workloads compared to
other stencil-based areas e.g., climate and weather modeling
is the prevalent use of high-order targeting for better accuracy. In the seismic exploration
space, high-order finite differencing is the de facto standard method
because it offers a favorable trade-off between numerical accuracy and
computational efficiency. High-order stencils are characterized by a
wider or larger stencil radius. Like many stencil patterns, the stencils
in seismic codes tend to be memory or bandwidth-bound. As the
stencil radius increases, the higher the arithmetic intensity
(floating-point instructions/memory load-store instructions) becomes
provided one can achieve a high degree of data reuse. As a result, high-order stencil codes have a tendency to put a significant amount of
pressure on the memory subsystem. While there is no \"one size fits
all\" approach, a few techniques form the basic building
blocks found in most implementations across the vast landscape of
seismic models. In this blog post and upcoming posts, we will explain
and demonstrate how some of these techniques can be implemented on AMD
Instinct accelerators and leveraged for getting the most out of the
hardware\'s performance for accelerating seismic stencil codes.

## Isotropic acoustic wave equation

Seismic stencil codes are mainly concerned with discretizing different
types of wave equations. These wave equation solvers form the
fundamental building blocks of reverse time migration (RTM) and full
waveform inversion (FWI) solvers; designed to image the Earth\'s
subsurface structure for oil and gas exploration. The governing
equations range from solving the acoustic wave equation to the elastic wave
equation in various types of media. Arguably the most well-studied
equation is the isotropic and constant density acoustic wave equation in
second form. This equation is solved for the pressure field $p(x,y,z,t)$
in a media parameterized by the speed of sound of $c(x,y,z)$ that is
spatially varying.

The isotropic acoustic wave equation is:

$$ p_{tt} = L(p,c) + s(x,y,z,t)  , \quad L = c^2\nabla^2 p, s(x,y,z,t) =  \delta(x-x')\delta(y-y')\delta(z-z')g(t)$$

subject to the homogeneous initial condition $p(x,y,z,0) = 0$. In the
above, $\nabla^2 p$ is the Laplacian of the pressure field and $g(t)$ is
a source term. The source term is to perturb the initial
pressure field such that waves are excited from a predetermined location
$(x',y',z')$. Over time, waves originating from the source propagate
throughout the domain.

Theoretically, waves propagate to infinity or continue until vanishing.
Unfortunately, this is not applicable in modeling a particular region
because of the computational resources limitations. Therefore we truncate
the model to a region of interest along with boundary regions.
Absorbing all incoming energy at the boundaries of the computational
grid mimic a real-life infinite media.

### Notes on boundary conditions

Of great importance is the enforcement of boundary conditions. In
practice, the desired boundary condition is of an absorbing type that
mimics that of an unbounded medium. The truncation of the computational
domain introduces an artificial boundary that generates spurious
reflections that can pollute the solution in the interior of the
computational domain. So many absorbing boundary approaches were
developed to deal with this problem with trade-offs between accuracy,
and computational cost; some depend on physical attenuation like
random boundaries and some depend on artificial attenuation like
sponge and perfectly matched layers (PML). For our
purposes, we focus on interior discretizations and set aside boundary
treatments as their computational impacts are usually smaller
compared to the interior computation provided that the volume to surface
the ratio is large.

## Discretization overview

It is common to discretize the acoustic wave equation in two steps,
following a method of lines approach:

1. The time integrator used is explicit in time and discretizes the
   $p_{tt}$ term using 2nd-order leapfrog central discretization whereas
   the right-hand side is evaluated at the current time step, $t_n$.\
2. The spatial part is discretized using high-order finite differences,
   typically 8th order.

With a slight abuse of notation, the resulting temporal discretization
becomes

$$ p^{n+1} - 2 p^n + p^{n-1} = \Delta t s(p^n, t^n)$$

In the above, it is understood that $p^n = p(x,y,z,t_n)$ and that
$t_n = \Delta t n$ is the $n$:th time step.

### Main loop

The following section demonstrates the main structure of an acoustic
wave equation solver.

```c
for (int n = 0; n < nsteps; ++n) {
     // Time step: t = n\Delta t 
     float t = n * dt;
     // Solves  p^{n+1} = -p^{n-1} + 2p^n + \Delta t L(p^n,t^n)
     solve_acoustic_wave_equation(p_next, p_curr, p_prev, c_squared, dt);
     // Treat snapshots
     // Per each n cycles; store compressed or uncompressed snapshots
     // Adds the discretization of the source term: s(x,y,z,t) evaluated at the grid index: si, sj, sk
     apply_source(p_next, si,sj,sk, t);
     // Cycle pointers to advance the pressure field for each time step
     swap(p_prev , p_curr);   
}
```

Note that this implementation stores the pressure field at only three
snapshots in time. In more detail, `p_next` corresponds to $p^{n+1}$,
`p_curr` corresponds to $p^n$, and `p_prev` corresponds to $p^{n-1}$.
The pointer-swapping technique is a simple way to avoid data movement
when advancing the wavefield.

## Space discretization

Most of the complexity that arises in the acoustic-iso seismic wave equation comes from the high
order discretization of the spatial part. For this
equation, it is necessary to discretize the Laplacian operator
$\\nabla\^2$ expressed in Cartesian coordinates, i.e., $\\nabla\^2 =
\\partial\^2\_{xx} + \\partial\^2\_{yy} + \\partial\^2\_{zz}$.

### Multi-pass and one-pass approaches

There are two types of approaches for discretizing the spatial part: the multi-pass approach and the one-pass one. These
approaches lie on opposite sides of the spectrum. In the multi-pass
approach, spatial derivatives are updated in multiple passes by operating in
one direction at a time. For example, the Laplacian operator can be
discretized by passing over the x-direction, then the y-direction,
and finally the z-direction. This is a *three-pass* approach. Here is an
example of this technique, demonstrated in pseudo-code for the Laplacian
operator only:

```c
// split `p_out = \partial_x^2 p_in + \partial_y^2 p_in + \partial_z^2 p_in` into:
// p_out = \partial_x^2 p_in
compute_full_fd_d2<x>(p_out, p_in, ...);
// p_out += \partial_y^2 p_in
compute_full_fd_d2<y>(p_out, p_in, ...);
// p_out += \partial_z^2 p_in
compute_full_fd_d2<z>(p_out, p_in, ...);
```

In this example, it is understood that the function
`compute_full_fd_d2<x>(...)` handles the finite difference
discretization in the x-direction for all grid points. Likewise,
in the y-dim and z-dim, in each step incrementally update output field array.

The disadvantage of this *three-pass* approach is that it requires
multiple trips to main memory, over the same data, to perform the
computation. In this case, `p_in` needs to be read three times and
`p_out` needs to be read two times and written three times. Since
seismic stencils are typically bandwidth-bound, these extra trips to
main memory can become quite costly from a performance standpoint. When
performance is important, it is generally recommended to implement a
*one-pass* approach. However, a major advantage of the multi-pass
approach is its simplified implementation because it processes
one grid direction at a time. This aspect becomes apparent if e.g., the
finite difference discretization changes at the boundary. In this case,
it could be necessary to alter the computation near the boundary. As a result, there would be three
different types of computations per grid direction. However, if all of
the grid directions are processed in a single pass it would be necessary
to do 3\^3 = 27 different computations.

At the opposite end of the spectrum is the *one-pass approach*. As the
name suggests, the one-pass approach uses a single function to compute
the entire Laplacian operator at once.

```c
// Compute the entire Laplacian: p_out = \partial_x^2 p_in + \partial_y^2 p_in + \partial_z^2 p_in
compute_full_fd_laplacian(p_out, p_in, ...);
```

This technique was used to discretize spatial operators in our blog
[here](../../finite-difference/docs/Laplacian_Part1.md). While this approach minimizes the number of trips to main
memory, it typically requires more GPU hardware resources, i.e.,
registers and shared memory as compared to multi-pass approaches. Since
the one-pass approach packs more work into a single kernel, it gives the
compiler more opportunities to optimize the code, and the number of
hand-crafted optimizations that one can perform dramatically increase
as well.

For our purposes, we are going to demonstrate some of the popular
techniques focusing on taking a multi-pass approach. We have made this
choice because it simplifies the forthcoming presentations. However, it
should be noted that these techniques should in most cases be extended
to one-pass approaches to obtain the best possible performance that the
hardware can deliver.

### High order finite difference approximations

Here are some high-order finite difference approximations of second
derivative terms of the Laplacian applied at the grid point
$(x_i, y_j, z_k)$ with uniform grid spacing $h_x, h_y, h_z$ in each grid-direction:

$$
\partial^2_{xx} p  \approx  \sum_{r=-R}^{R} \frac{d_{r}}{h_x^2} p(x_i + r h_x, y_j, z_k,t), \
\partial^2_{yy} p  \approx  \sum_{r=-R}^{R} \frac{d_{r}}{h_y^2} p(x_i, y_j + rh_y, z_k,t), \
\partial^2_{zz} p  \approx  \sum_{r=-R}^{R} \frac{d_{r}}{h_z^2} p(x_i, y_j, z_k + rh_z,t).
$$

In the approximations above, the stencil has a radius $R$, and the width of
the stencil is $2R + 1$ grid points wide. If the stencil is symmetric,
$d_r = d_{-r}$, then it is possible to construct difference
approximations of $2R$ in order of accuracy.For $R=1$, the symmetric second
order accurate coefficients are: $d_{-1} = d_{1} = 1$ and $d_{0} = -2$.
Higher orders are tabulated
[here](https://en.wikipedia.org/wiki/Finite_difference_coefficient). For
seismic applications, $R$ is typically at least $4$ (8th order) or more.

The following example code applies a high-order finite difference
operator to `nx * ny * nz` interior grid points in a given direction.

```c
template <int stride>
__inline__ void compute_d2(float *p_out, const float *p_in, const float *d, const int R, int pos) {
    /* Computes a central high-order finite difference approximation in a given direction
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
     pos: The current grid point to apply the approximation to
     stride: The stride to use for the stencil. This parameter controls the direction in which the stencil is applied.
     */
    float out = 0.0f;
    for (int r = -R; r < R; ++r) 
        out += d[r + R] * p_in[pos + r * stride]; 
    }
    p_out[pos] += out;
}
```

This example code uses floating-point single precision because this
level of precision is arguably the most widely used choice in practice.

When implementing the high-order finite difference method, it is
convenient to pad the arrays to leave room for halo regions. The halo
regions could be occupied by neighbors or boundary values for imposing
boundary conditions.

```c
// Interior grid size
int nx, ny, nz;
nx=ny=nz=10;
// Padded grid size
int mx = nx + 2 * R;
int my = ny + 2 * R;
int mz = nz + 2 * R;
uint64_t m = mx * my * mz;
float *p_out = new float[m];
float *p_in = new float[m];
```

Since the input and output arrays are collapsed into one dimension, the
variable `pos` specifies how to linearly access grid value at a grid point
$(x_i, y_j,z_k)$ via the formula:

```c
pos = i + line * j + slice * k;
```

If the leading dimension of the array is taken along the x-direction,
then

```c
int line = nx + 2 * R;
int slice = line * (ny + 2 * R);
```

The listing below demonstrates the application of the high-order finite
difference method.

```c
const float h = 0.1;
const int R = 1;
const float d[2*R + 1] = {1.0 / (h * h), -2.0 / (h * h), 1.0 / (h * h)};

// Define each direction to apply the finite difference approximation in
const int x = 1;
const int y = line;
const int z = stride;

const int line = mx;
const int slice = mx * my;
const uint64_t n = mx * my * mz;

// zero-initialize the memory
memset(p_out, 0, n * sizeof(float));    
    
// Apply approximation in the x-direction for all interior grid points
for (int k = R; k < nz + R; ++k) {
    for (int j = R; j < ny + R; ++j) {
        for (int i = R; i < nx + R; ++i) {
            const uint64_t pos = i + line * j + slice * k;          
            compute_d2<x>(p_out, p_in, d, R, pos, line, slice);
        }
    }
}
```

By changing the `stride` parameter, the code can easily be changed into
a finite difference approximation of a particular direction.

```c
// Apply approximation in the x-direction
compute_d2<x>(p_out, p_in, d, R, pos, line, slice);
// Apply approximation in the y-direction
compute_d2<y>(p_out, p_in, d, R, pos, line, slice);
// Apply approximation in the z-direction
compute_d2<z>(p_out, p_in, d, R, pos, line, slice);
```

## Baseline kernels

It is now time to take a look at high-order finite difference
approximations on the GPU. As a starting point, let us consider a simple
kernel that applies the finite difference method in the x-direction.
This kernel will be used for all future performance comparisons.

``` cpp
  // Table containing finite difference coefficients
  template <int R> __constant__ float d_dx[2 * R + 1];
  template <int R> __constant__ float d_dy[2 * R + 1];
  template <int R> __constant__ float d_dz[2 * R + 1];

  template <int R>
  __launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y) * (BLOCK_DIM_Z))
  __global__ void compute_fd_x_gpu_kernel(float *__restrict__ p_out, const float *__restrict__ p_in,
          int line, int slice, int x0, int x1, int y0, int y1, int z0, int z1) {

      const int i = x0 + threadIdx.x + blockIdx.x * blockDim.x;
      const int j = y0 + threadIdx.y + blockIdx.y * blockDim.y;
      const int k = z0 + threadIdx.z + blockIdx.z * blockDim.z;

      if (i >= x1 || j >= y1 || k >= z1) return;

      size_t pos = i + line * j + slice * k;
      int stride = 1; // x-direction

      // Shift pointers such that that p_in points to the first value in the stencil
      p_in += pos - R * stride;
      p_out += pos;

      // Compute the finite difference approximation
      float out = 0.0f;
      for (int r = 0; r <= 2 * R; ++r)
          out += p_in[r * stride] * d_dx<R>[r]; // x-direction

      // Write the result
      p_out[0] = out;

  }
```

The above kernel can be modified to compute stencils in the y-direction
and z-direction with `int stride = line;` and `int stride = slice;`,
respectively. This kernel uses `x0,y0,z0` and `x1,y1,z1` to define the
region of interest to apply the stencil. Defining the bounds in such
a way is convenient because it gives explicit control over where to apply
the stencil. If there is a need to overlap communication and
computation, the same kernel can be called multiple times with arguments
that map to e.g., different halo slices.  To restrict attention to all
interior points, the bounds need to be `x0=R`, `x1=nx+R`, `y0=R`,
`y1=ny+R`, `z0=R`, `z1=nz+R`. In the kernel, the indices `i`,`j`,`k` are
shifted by `x0`,`y0`,`z0`. This shift saves three comparison
instructions because it removes the lower bounds check and also
avoids having some idle threads outside the lower bounds. On the other
hand, the shift in the x-direction can negatively impact performance due
to unaligned memory accesses.

## Initial performance

Let us quickly assess the performance of these baseline kernels using two different figure of merits (FOM).
First, like in the [Laplacian series](../../finite-difference/docs/Laplacian_Part1.md), we understand from
the stencil algorithm design that each grid point need only be loaded once and is reusable by
neighboring threads. Hereby, we shall use the *effective memory bandwidth* defined as:

```c++
effective_memory_bandwidth = (theoretical_fetch_size + theoretical_write_size) / average_kernel_execution_time;
```

where the theoretical fetch and write sizes (in bytes) of an `nx * ny * nz` cube are:

```c++
theoretical_fetch_size = (nx * ny * nz + 2 * R * PLANE) * sizeof(float);
theoretical_write_size = (nx * ny * nz) * sizeof(float);
```

where `PLANE` is `ny * nz`, `nx * nz`, and `ny * nz` for the x-direction, y-direction, and z-direction, respectively.
Ideally, we want this FOM to be a close to the achievable memory bandwidth as possible. On a single MI250X GCD,
these are the *effective memory bandwidth*[^1] numbers on a 512 x 512 x 512 cube for various values of `R`:

| Radius (R) | x-direction | y-direction | z-direction |
|------------|:-----------:|:-----------:|:-----------:|
| 1          |   971 GB/s  |   948 GB/s  |   915 GB/s  |
| 2          |   995 GB/s  |   982 GB/s  |   753 GB/s  |
| 3          |   977 GB/s  |   965 GB/s  |   394 GB/s  |
| 4          |   956 GB/s  |   940 GB/s  |   217 GB/s  |

Both larger stencil radii and larger strides in memory worsen the bandwidth.
All of these numbers, including the smaller stencil radii and strides, fall
significantly short of the achievable bandwidth of a single MI250X GCD, which according to the
[BabelStream case studies](https://www.olcf.ornl.gov/wp-content/uploads/2-16-23-node_performance.pdf)
is roughly 1.3 TB/s to 1.4 TB/s. However, these FOMs alone do not offer insight
into how well our GPU implementations are utilizing the hardware.

Therefore, we provide the *achieved memory bandwidth* which is attainable using two different approaches. The first approach is to
use `rocprof` directly and take the sum of the `FETCH_SIZE` and `WRITE_SIZE` metrics divided by the average kernel execution time.
The second approach is to use [omniperf](https://amdresearch.github.io/omniperf/) and take the sum of the reported `L2-EA Rd BW` and `L2-EA Wr BW`.
Ideally we also want this number to be as close to the achievable memory bandwidth as possible.
Below are the corresponding achieved memory bandwidth numbers[^1]:

| Radius  | x-direction | y-direction | z-direction |
|---------|:-----------:|:-----------:|:-----------:|
| 1       |   976 GB/s  |   953 GB/s  |   931 GB/s  |
| 2       |  1003 GB/s  |   995 GB/s  |   953 GB/s  |
| 3       |   986 GB/s  |   981 GB/s  |   925 GB/s  |
| 4       |   967 GB/s  |   959 GB/s  |   891 GB/s  |

Both the x-direction and y-direction present achieved memory bandwidth numbers that align closely with
the reported effective memory bandwidth. When both the effective and achieved memory bandwidth numbers align,
this indicates that the amount of data, i.e., the numerator, moved is the same. Both bandwidth
metrics use the same kernel execution time, so if both bandwidth metrics are low, this suggests
that either a portion of the kernel is compute-heavy and not fully overlapped with memory transfers
and/or there are underlying latency issues.

However, the z-direction appears to have a much higher achieved memory
bandwidth, suggesting that while the hardware utilization may be better
than the x and y direction implementations, the kernel is fetching from and/or
writing to global memory much more than ideal.

## Conclusion

This concludes the first part of developing seismic stencil codes that often
rely on the discretization of the wave equation using high-order
finite difference methods. We first begin with the three-pass approach
and present initial performance numbers for varying stencil radii and
directions. None of the implementations show satisfactory
performance in either the effective or achieved memory bandwidths,
and the next series of posts shall dive into possible
optimizations to elevate the memory bandwidth numbers.

[Accompanying code examples](https://github.com/amd/HPCTrainingExamples/tree/main/rocm-blogs-codes/seismic-stencils)

If you have any questions or comments, please reach out to us on
GitHub [Discussions](https://github.com/amd/amd-lab-notes/discussions)

[^1]:Testing conducted using ROCm version 6.1.0-82. Benchmark results are not
validated performance numbers, and are provided only to demonstrate relative
performance improvements of code modifications. Actual performance results
depend on multiple factors including system configuration and environment
settings, reproducibility of the results is not guaranteed.
