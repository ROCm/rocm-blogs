---
blogpost: true
date: 13 Nov 2024
blog_title: "Introducing AMD's Next-Gen Fortran Compiler"
author: Justin Chang, Brian Cornille, Michael Klemm, Johanna Potyka
tags: Compiler, HPC, Performance
thumbnail: './images/Next-Gen-Fortran.jpeg'
category: Ecosystems and Partners
language: English
myst:
  html_meta:
    "description lang=en": "In this post we present a brief preview of AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md), our new open source Fortran complier optimized for AMD GPUs using OpenMP offloading, offering direct interface to ROCm and HIP."
    compiler focusing on OpenMP offloading."
    "keywords": "HPC, Fortran, Compiler"
    "property=og:locale": "en_US"
---

# Introducing AMD's Next-Gen Fortran Compiler

We are excited to share a brief preview of AMD's
[Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md),
our new open source Fortran complier supporting OpenMP offloading. AMD's
[Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)
is a downstream flavor of [LLVM Flang](https://flang.llvm.org), optimized for AMD GPUs.
Our [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)
enables OpenMP offloading and offers a direct interface to ROCm and HIP.
In this blog post you will:

1. Learn how to use AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) to deploy and accelerate your Fortran codes on AMD GPUs using OpenMP offloading.
2. Learn how to use AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) to interface and invoke HIP and ROCm kernels.
3. See how AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) OpenMP offloading exhibits competitive performance against native HIP/C++ codes, benchmarking on AMD GPUs.
4. Learn how to access a pre-production build of the new AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md).

## Our commitment to Fortran

Fortran is a powerful programming language for scientific and engineering high performance computing applications and is core to many, some very crucial, HPC codebases.
Fortran remains under active development as a standard, supporting both legacy and modern codebases.
The need for a more modern Fortran compiler motivated the creation of the
[LLVM Flang](https://flang.llvm.org/) project and AMD fully supports that path.
In following with community trends, AMD's
[Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)
will be a downstream flavor of [LLVM Flang](https://flang.llvm.org/) and will in time supplant
the current AMD Flang compiler, a downstream flavor of
"[Classic Flang](https://github.com/flang-compiler/flang)".

Our mission is to allow anyone that is using and developing a Fortran HPC codebase
to directly leverage the power of AMD’s GPUs.
AMD's [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)’s
goal is fulfilling our vision by allowing you to deploy and accelerate your Fortran codes
on AMD GPUs using OpenMP offloading, and to directly interface and invoke HIP and ROCm kernels.
This early preview is an opportunity for us to receive feedback from our users and develop an even better product.

Now, let’s dive into AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)!

## AMD's Next-Gen Fortran Compiler - Fortran with OpenMP offloading

To illustrate the functionality of OpenMP offloading in AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md), we use [example code](https://github.com/amd/HPCTrainingExamples/tree/main/rocm-blogs-codes/jacobi) from the
["Jacobi Solver with HIP and OpenMP offloading"](https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html) blog
and translate it to Fortran with OpenMP offloading directives.
The Fortran version of this code follows the same structure with the same set of methods.
The main differences are that Fortran supports multidimensional arrays natively so there is no manual index calculation required.
For the sake of brevity, we only show the relevant fragments of our Jacobi implementation.

Recall that there are four key kernels in the execution of the Jacobi iterative method for the finite difference approximation
of the Poisson's equation $-\nabla^2 u(x,y) = f$:

1. Compute the Laplacian
2. Apply the boundary conditions
3. Update the solution vector
4. Compute the residual norm

These four kernels rely on four key data structures: the solution vector `u`, the right hand side vector `rhs`, the finite difference approximation `au`, and
the residual vector `res`. In our example, they are allocated as follows:

```Fortran
module jacobi_mod
  use kind_mod, only: RK
  implicit none

  ! RK is a type parameter that can be real32 or real64:
  ! integer, parameter RK = ...
  type :: jacobi_t
    real(kind=RK), allocatable :: u(:,:), rhs(:,:), au(:,:), res(:,:)
  end type jacobi_t

contains

  subroutine init_jacobi(this, mesh)
    type(jacobi_t), intent(inout) :: this
    type(mesh_t), intent(inout) :: mesh

    allocate(this%u(mesh%n_x,mesh%n_y))
    allocate(this%au(mesh%n_x,mesh%n_y))
    allocate(this%rhs(mesh%n_x,mesh%n_y))
    allocate(this%res(mesh%n_x,mesh%n_y))

    ! Initialize values
    this%u = 0._RK
    this%rhs = 0._RK
    this%au = 0._RK

    ! ... some code removed for brevity ...

    this%res = this%rhs
    !$omp target enter data map(to:this%u,this%rhs,this%au,this%res)

    ! ... some code removed for brevity ...
  end subroutine init_jacobi
end module jacobi_mod
```

After using the `allocate` statements to reserve memory for four key data arrays, the code uses the `!$omp target enter data`
directive with corresponding `map` clauses to allocate device buffers for the arrays in the GPU memory.

> **Note:** For architectures that support unified shared memory, like the AMD Instinct™ MI300A Accelerated Processing Units (APU),
> the Fortran compiler can elide the entire `!$omp target enter data` directive when in [zero-copy mode](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/openmp.html#zero-copy-behavior-on-mi300a-and-discrete-gpus).
> This mode is enabled via [XNACK](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/openmp.html#xnack-capability) by setting the environment variable `HSA_XNACK=1`.
> There is no cost to leaving OpenMP `map` statements as implicit zero-copy mode still prevents redundant data copies on the MI300A APU.
> Leaving these directives in, however, maintains code portability between AMD Instinct(tm) GPUs in discrete memory mode and APUs.

The following code shows the order of the subroutine calls in each Jacobi iteration to compute the four steps outlined above.

```Fortran
subroutine run_jacobi(this, mesh)
  type(jacobi_t), intent(inout) :: this
  type(mesh_t), intent(inout) :: mesh

  ! ... some code removed for brevity ...

  do while (this%iters < max_iters .and. resid > tolerance)
    ! Compute Laplacian
    call laplacian(mesh,this%u,this%au)

    ! Apply boundary conditions
    call boundary_conditions(mesh,this%u,this%au)

    ! Update the solution
    call update(mesh,this%rhs,this%au,this%u,this%res)

    ! Compute residual norm ||U||
    resid = norm(mesh,this%res)

    this%iters = this%iters + 1
  end do

  ! ... some code removed for brevity ...
end subroutine run_jacobi
```

The following side-by-side comparison shows the original C++ OpenMP offload implementation of the four subroutines
together with their Fortran OpenMP offload counterpart. Please note, that the `target` directives do not require any further `map` clauses.
For the four data arrays, the corresponding `target enter data` directive created the GPU buffers (if needed, see the note above).
For the scalar variables, the Fortran OpenMP compiler will implicitly add localized `map` clauses where needed to copy the scalar variable to the GPU memory.
In addition, the OpenMP API mandates that such scalars are automatically made `firstprivate`, which is a sensible default for most scalar variables.
Similar to the data arrays, the compiler and runtime system will automatically elide these copies for unified shared memory systems.

<table>
<tr>
<th>
Laplacian - C++
</th>
<th>
Laplacian - Fortran
</th>
</tr>
<tr>
<td style="vertical-align:top">

```cpp
template<typename FT>
void Laplacian(mesh_t& mesh, const FT _1bydx2,
    const FT _1bydy2, FT* U, FT* AU)
{
  int stride = mesh.Nx;
  int localNx = mesh.Nx-2;
  int localNy = mesh.Ny-2;

  #pragma omp target teams distribute parallel for collapse(2)
  for (int j=0; j<localNy; j++) {
    for (int i=0; i<localNx; i++) {

      const int id = (i+1) + (j+1)*stride;

      const int id_l = id - 1;
      const int id_r = id + 1;
      const int id_d = id - stride;
      const int id_u = id + stride;

       AU[id] = (-U[id_l] + 2*U[id] - U[id_r])*_1bydx2 +
                (-U[id_d] + 2*U[id] - U[id_u])*_1bydy2;
    }
  }
}
```

</td>
<td style="vertical-align:top">

```Fortran
subroutine laplacian(mesh, u, au)
  type(mesh_t), intent(inout) :: mesh
  real(kind=RK), intent(inout) :: u(:,:)
  real(kind=RK), intent(inout) :: au(:,:)
  integer :: i, j
  real(kind=RK) :: invdx2, invdy2

  invdx2 = mesh%dx**-2
  invdy2 = mesh%dy**-2

  !$omp target teams distribute parallel do collapse(2)
  do j = 2, mesh%n_y-1
    do i = 2, mesh%n_x-1
      au(i,j) = (-u(i-1,j) + 2._RK*u(i,j) - u(i+1,j))*invdx2 &
              + (-u(i,j-1) + 2._RK*u(i,j) - u(i,j+1))*invdy2
    end do
  end do
end subroutine laplacian
```

</td>
</tr>
</table>

<table>
<tr>
<th>
Boundary Conditions - C++
</th>
<th>
Boundary Conditions - Fortran
</th>
</tr>
<tr>
<td style="vertical-align:top">

```cpp
template<typename FT>
void BoundaryConditions(mesh_t& mesh,
    const FT _1bydx2, const FT _1bydy2,
    FT* U, FT* AU) {

  const int Nx = mesh.Nx;
  const int Ny = mesh.Ny;

  #pragma omp target teams distribute parallel for
  for (int id=0; id<2*Nx+2*Ny-2; id++) {

    int i, j;
    if (id < Nx) {                 //bottom
      i = id;
      j = 0;
    } else if (id<2*Nx) {          //top
      i = id - Nx;
      j = Ny - 1;
    } else if (id < 2*Nx + Ny-1) { //left
      i = 0;
      j = id - 2*Nx + 1;
    } else {                       //right
      i = Nx - 1;
      j = id - 2*Nx - Ny + 2;
    }

    const int iid = i + j*Nx;

    const FT U_d = (j==0)    ?  0.0 : U[iid - Nx];
    const FT U_u = (j==Ny-1) ?  0.0 : U[iid + Nx];
    const FT U_l = (i==0)    ?  0.0 : U[iid - 1];
    const FT U_r = (i==Nx-1) ?  0.0 : U[iid + 1];

    AU[iid] = (-U_l + 2*U[iid] - U_r)*_1bydx2 +
              (-U_d + 2*U[iid] - U_u)*_1bydy2;
  }
}
```

</td>
<td style="vertical-align:top">

```Fortran
subroutine boundary_conditions(mesh, u, au)
  type(mesh_t), intent(inout) :: mesh
  real(kind=RK), intent(inout) :: u(:,:)
  real(kind=RK), intent(inout) :: au(:,:)
  integer :: id, i, j, n_x, n_y
  real(kind=RK) :: invdx2, invdy2

  n_x = mesh%n_x
  n_y = mesh%n_y
  invdx2 = mesh%invdx2
  invdy2 = mesh%invdy2

  !$omp target teams distribute parallel do private(i,j)
  do id=1, 2*n_x + 2*n_y-4
    if (id == 1) then
      au(1,1) = (2._RK*u(1,1) - u(2,1))*invdx2 &
              + (2._RK*u(1,1) - u(1,2))*invdy2
    else if (id <= n_x-1) then
      i = id
      au(i,1) = (-u(i-1,1) + 2._RK*u(i,1) - u(i+1,1))*invdx2 &
              + (2._RK*u(i,1) - u(i,2))*invdy2
    else if (id == n_x) then
      au(n_x,1) = (2._RK*u(n_x,1) - u(n_x-1,1))*invdx2 &
                + (2._RK*u(n_x,1) - u(n_x,2))*invdy2
    else if (id == n_x+1) then
      au(1,n_y) = (2._RK*u(1,n_y) - u(2,1))*invdx2 &
                + (2._RK*u(1,n_y) - u(1,n_y-1))*invdy2
    else if (id <= 2*n_x-1) then
      i = id - n_x
      au(i,n_y) = (-u(i-1,n_y) + 2._RK*u(i,n_y) - u(i+1,n_y))*invdx2 &
                + (2._RK*u(i,n_y) - u(i,n_y-1))*invdy2
    else if (id == 2*n_x) then
      au(n_x,n_y) = (2._RK*u(n_x,n_y) - u(n_x-1,1))*invdx2 &
                  + (2._RK*u(n_x,n_y) - u(n_x,n_y-1))*invdy2
    else if (id <= 2*n_x+n_y-2) then
      j = id - 2*n_x + 1
      au(1,j) = (2._RK*u(1,j) - u(2,j))*invdx2 &
              + (-u(1,j-1) + 2._RK*u(1,j) - u(1,j+1))*invdy2
    else
      j = id - 2*n_x - n_y + 3
      au(n_x,j) = (2._RK*u(n_x,j) - u(n_x-1,j))*invdx2 &
                + (-u(n_x,j-1) + 2._RK*u(n_x,j) - u(n_x,j+1))*invdy2
    end if
  end do
end subroutine boundary_conditions
```

</td>
</tr>
</table>

<table>
<tr>
<th>
Update - C++
</th>
<th>
Update - Fortran
</th>
</tr>
<tr>
<td style="vertical-align:top">

```cpp
template<typename FT>
void Update(mesh_t& mesh, const FT factor,
    FT* RHS, FT* AU,
    FT* RES, FT* U)
{
  const int N = mesh.N;
  #pragma omp target teams distribute parallel for
  for (int id=0; id<N; id++)
  {
    const FT r_res = RHS[id] - AU[id];
    RES[id] = r_res;
    U[id] += r_res*factor;
  }
}
```

</td>
<td style="vertical-align:top">

```Fortran
subroutine update(mesh,rhs,au,u,res)
  type(mesh_t), intent(inout) :: mesh
  real(kind=RK), intent(inout) :: rhs(:,:), au(:,:)
  real(kind=RK), intent(inout) :: u(:,:)
  real(kind=RK), intent(inout) :: res(:,:)
  integer :: i, j
  real(kind=RK) :: temp

  factor = (2._real64/mesh%dx**2 + 2._real64/mesh%dy**2)**-1

  !$omp target teams distribute parallel do collapse(2) private(temp)
  do j = 1,mesh%n_y
    do i = 1,mesh%n_x
      temp = rhs(i,j) - au(i,j)
      res(i,j) = temp
      u(i,j) = u(i,j) + temp*factor
    end do
  end do
end subroutine
```

</td>
</tr>
</table>

<table>
<tr>
<th>
Norm - C++
</th>
<th>
Norm - Fortran
</th>
</tr>
<tr>
<td style="vertical-align:top">

```cpp
template<typename FT>
FT Norm(mesh_t& mesh, FT *U)
{
  FT norm = 0.0;
  const int N = mesh.N;
  const FT dx = mesh.dx;
  const FT dy = mesh.dy;

  #pragma omp target teams distribute parallel for reduction(+:norm)
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id] * dx * dy;
  }
  return sqrt(norm)/N;
}

```

</td>
<td style="vertical-align:top">

```Fortran
function norm(mesh, res) result(norm_val)
  type(mesh_t), intent(inout) :: mesh
  real(kind=RK), intent(inout) :: res(:,:)
  real(kind=RK) :: norm_val
  integer :: i, j

  dx = mesh%dx
  dy = mesh%dy

  !$omp target teams distribute parallel do collapse(2)
  !$omp& reduction(+:norm_val)
  do j = 1,mesh%n_y
    do i = 1,mesh%n_x
      norm_val = norm_val + res(i,j)**2*dx*dy
    end do
  end do

  norm_val = sqrt(norm_val)/(mesh%n_x*mesh%n_y)
end function norm

```

</td>
</tr>
</table>

The entire Jacobi solver source code, as well as several other Fortran OpenMP offloading examples, are located in the
[AMD HPCTrainingExamples](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples/OpenMP/Fortran) repository.
We encourage all interested readers to examine the accompanying code samples for a more detailed picture of what AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)
running on AMD GPUs can do with OpenMP offloading.
Please check back often as this repository will be updated over time with more sophisticated examples.

## Interoperability with HIP and ROCm

An important extension to the OpenMP offload capabilities of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) is for Fortran codes to
invoke kernels written with the HIP language or call into subroutines provided by the AMD ROCm™ libraries.

Consider this sample Fortran program that needs to compute the element-wise sum of two vectors `a` and `b` and store into `c`.
Like we had shown earlier, after allocating the arrays in host memory, a corresponding `!$omp target enter data` moves
the data over to the GPU (in case we target a discrete GPU) and `!$omp target exit data` cleans up when the data is no longer
in use on the GPU.  In between, the code runs the element-wise vector operation using `!$omp target teams distribute parallel do`
and copies the updated `c` array from GPU memory back to the host via `!$omp target update`.
As usual, for APUs these data movement operations will be elided.

```Fortran
program example
  use iso_fortran_env, only: real64
  implicit none

  real(kind=real64), allocatable, dimension(:) :: a, b
  integer :: i
  integer, parameter :: N = 1024

  allocate(a(N),b(N))
  a = 1.0
  b = 2.0
  !$omp target enter data map(to:a,b)

  ! Compute element-wise b = a + b
  !$omp target teams distribute parallel do
  do i=1,N
    b(i) = a(i) + b(i)
  end do

  ! Verification
  !$omp target update from(b)
  if (b(1) /= 3.0) then
    print *, "Answer should be 3.0"
    stop 1
  end if

  !$omp target exit data map(delete:a,b)
  deallocate(a)
  deallocate(b)
end program example
```

This operation was simple enough to leverage `!$omp target teams distribute parallel do`, but there
may be reasons to invoke HIP kernels instead. Usually, this is the case when special, low-level
features of the GPU architecture should be used (e.g., explicit usage of the Local Data Store
of AMD Instinct™ Accelerators).  In those cases [AMDFlang](https://repo.radeon.com/rocm/misc/flang/) provides
elegant ways to invoke HIP kernels and ROCm library functionality.

### Option 1: Fortran to HIP interface

With AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md), users can interface to native C++/HIP code by introducing an explicit `interface` that uses `bind(C)` to bind the Fortran function or subroutine to a C function that then launches a HIP kernel.
Since HIP is based on C++, the invoked C++ function needs to have the `extern "C"` attribute to follow the C calling convention and make it accessible via the Fortran `interface` block.

Here's how the code could look:

```Fortran
module hip_interface
  interface
    subroutine hip_function(a, b, n) bind(C, name="hip_function")
      use iso_c_binding, only: c_ptr, c_int
      implicit none
      type(c_ptr), value :: a, b
      integer(c_int), value :: n
    end subroutine hip_function
  end interface
end module hip_interface
```

The explicit interface in the `hip_interface` describes to the Fortran compiler how the corresponding C++ function must be invoked to correctly pass data to it.  Because C++ functions use Call-by-Value for all parameters, the interface declares the parameters as `value` and uses the proper data types to pass pointer arguments and integers to a C++ function.

Once we defined the `hip_interface` module, we can import it into the example code and replace the OpenMP offload region with a subroutine call to invoke the HIP function:

```Fortran
program example
  use hip_interface
  use iso_c_binding, only: c_double, c_loc
  implicit none

  real(c_double), allocatable, target, dimension(:) :: a, b
  integer :: i
  integer, parameter :: N = 1024

  allocate(a(N),b(N))
  a = 1.0
  b = 2.0
  !$omp target enter data map(to:a,b)

  ! Compute element-wise b = a + b
  !$omp target data use_device_addr(a,b)
  call hip_function(c_loc(a),c_loc(b),N)
  !$omp end target data

  ! Verification
  !$omp target update from(b)
  if (b(1) /= 3.0) then
    print *, "Answer should be 3.0"
    stop 1
  end if

  !$omp target exit data map(delete:a,b)
  deallocate(a)
  deallocate(b)
end program example
```

One slight complication is that HIP requires the code to supply the kernel code with device pointers.
Since we are using the OpenMP API to handle buffer management and data movement (if and where needed),
we need a way to get the corresponding device pointers for the `a` and `b` arrays. This is accomplished by
the `use_device_addr` clause. It dives deep into the OpenMP runtime system and looks for the the device address
of where the OpenMP implementation has allocated the data on the GPU. We can then use the `c_loc` intrinsic
function to retrieve that device address as a pointer of type `c_ptr` that we can then pass to the Fortran interface for `hip_function`.

Finally, here's the implementation of the `vector_add` kernel and it's trampoline function `hip_function`:

```cpp
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}

__global__ void vector_add(double *a, double *b, int n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    b[i] = a[i] + b[i];
}

extern "C"
void hip_function(double *a, double *b, int n)
{
  vector_add<<<n/256, 256>>>(a, b, n);
  HIP_CHECK(hipDeviceSynchronize());
}
```

The `hip_function` code receives the pointers to the Fortran arrays `a` and `b` as regular pointers as well as the size
information via `n`.  It then uses the triple-chevron syntax to launch the `vector_add` kernel that performs the vector addition.
Alternatively, the C++ code can also call math libraries like rocBLAS, rocFFT, etc. or other libraries that contain HIP kernel code.

The Fortran code and the C++/HIP source files must be compiled separately using AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) and HIPCC,
including linking to the HIP runtime libraries. Here is how the `Makefile` looks when stitching everything together:

```shell
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

# Compiler
CXX         = clang -x hip
FC          = amdflang

# Compiler flags
CXX_FLAGS   = -O3 --offload-arch=${ROCM_GPU}
FC_FLAGS    = -O3 -fopenmp --offload-arch=${ROCM_GPU}
LD_FLAGS    = -O3 -L${ROCM_PATH}/lib -lamdhip64 -lstdc++

all:
  ${CXX} ${HIP_FLAGS} -c kernel.cpp -o kernel.o
  ${FC} ${FC_FLAGS} -c driver.f90 -o driver.o
  ${FC} ${FC_FLAGS} ${LD_FLAGS} -o driver driver.o kernel.o
```

On unified memory models like the AMD Instinct™ MI300A APU, the data management directives including the `!$omp target data use_device_addr()`
directive are not required since the host and device addresses are the same, but
[XNACK](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/openmp.html#xnack-capability) must
be enabled via the environment variable `HSA_XNACK=1`.  In that case, the Fortran example can be greatly simplified:

```Fortran
program example
  use hip_interface
  use iso_c_binding, only: c_double, c_loc
  !$omp requires unified_shared_memory
  implicit none

  real(c_double), allocatable, target, dimension(:) :: a, b
  integer :: i
  integer, parameter :: N = 1024

  allocate(a(N),b(N))
  a = 1.0
  b = 2.0

  ! Compute element-wise b = a + b
  call hip_interface(c_loc(a),c_loc(b),N)

  ! Verification
  if (b(1) /= 3.0) then
    print *, "Answer should be 3.0"
    stop 1
  end if

  deallocate(a)
  deallocate(b)
end program example
```

As seen above, we have removed all OpenMP directives, except `!$omp requires unified_shared_memory`.  `!$omp requires unified_shared_memory` informs the compiler that this code is supposed to run only on a GPU architecture supporting unified shared memory.```

### Option 2: AMD ROCm™ libraries

Another option is to interface with AMD ROCm™ math libraries using [hipfort](https://rocm.docs.amd.com/projects/hipfort/en/latest/). Hipfort is a Fortran library
that exposes the HIP API and the ROCm libraries to Fortran code with an open and portable set of standard Fortran module interfaces. This is particularly
useful when a user simply needs to leverage math libraries like rocBLAS and does wish to not use any native HIP kernels.  Here is how the Fortran
example would look like when using the `hipfort` module:

```Fortran
program example
  use iso_c_binding, only: c_ptr, c_double
  use iso_fortran_env, only: real64
  use hipfort
  use hipfort_check
  use hipfort_rocblas

  implicit none

  real(kind=real64), allocatable, target, dimension(:) :: a, b
  integer :: i
  integer, parameter :: N = 1024
  type(c_ptr) :: rocblas_handle
  real(c_double), target :: alpha = 1.0

  allocate(a(N),b(N))
  a = 1.0; b = 2.0
  !$omp target enter data map(to:a,b)

  ! Compute element-wise b = a + b
  call rocblasCheck(rocblas_create_handle(rocblas_handle))
  call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0))
  !$omp target data use_device_addr(a,b)
  call rocblasCheck(rocblas_daxpy(rocblas_handle, N, alpha, c_loc(a), 1, c_loc(b), 1))
  !$omp end target data
  call hipCheck(hipDeviceSynchronize())

  ! Verification
  !$omp target update from(b)
  if (b(1) /= 3.0) then
    print *, "Answer should be 3.0"
    stop 1
  end if

  !$omp target exit data map(delete:a,b)
  deallocate(a)
  deallocate(b)
  call rocblasCheck(rocblas_destroy_handle(rocblas_handle))
end program example
```

The hipfort library provides a file [`Makefile.hipfort`](https://github.com/ROCm/hipfort/blob/develop/bin/Makefile.hipfort) to aid in using the right compiling and linking flags.
Here's how it can be incorporated into a `Makefile` to build the example with `amdflang`:

```shell
# Compiler
HIPFORT_HOME ?= /opt/rocm-afar/hipfort # This must point to the hipfort built with amdflang
HIPFORT_COMPILER ?= amdflang
HIPCC_LIBS := -lrocblas
include $(HIPFORT_HOME)/share/hipfort/Makefile.hipfort # Sets FC based on HIPFORT_COMPILER

# Compiler flags
FC_FLAGS    = -O3 -fopenmp --offload-arch=${GPU}
LD_FLAGS    = -O3 ${LINKOPTS}

all:
  ${FC} ${FC_FLAGS} -c driver.f90 -o driver.o
  ${FC} ${FC_FLAGS} ${LD_FLAGS} -o driver driver.o
```

> **NOTE**: Users must use a build of `hipfort` that is compiled with the same Fortran compiler used to target the example.
> See the ["Using hipfort" section of the user guide](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md#using-hipfort) for build instructions using `amdflang`.
> This is also important when using third party Fortran compilers such GFortran or the Cray Fortran compiler.

## Performance portability for Fortran

One advantage of a directive-based approach to GPU programming is the ability to achieve near
device-native programming performance at a much lower complexity and therefore higher productivity.
In the previous sections, we showed how HIP code integrates into a Fortran code base
but at the cost of some boilerplate code on top of the HIP kernel.
If OpenMP offloading achieves similar performance to optimized HIP kernels,
then it would be clearly preferred as it does not require extra steps like writing explicit `bind(C)` interfaces.

A nearly trivial example that illustrates this comparison is the [BabelStream benchmark](https://github.com/UoB-HPC/BabelStream).
The goal of this benchmark suite is to measure achievable device bandwidth in a vast array of programming models.
For our purposes it contains both Fortran OpenMP offloading and HIP implementations.
Below we show the relative performance of these two implementation using pre-production builds of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) and the AMD C/C++ Compiler (`amdclang++`) for GPUs.

Using the [`develop` branch of BabelStream](https://github.com/UoB-HPC/BabelStream/tree/develop),
the following build and run commands were used for each version in order to collect performance numbers.
We then report the percentage of achieved performance of the Fortran OpenMP code compared to the HIP code.

```shell
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream
git checkout develop
export ROCM_PATH=<install path of amdflang>
```

### HIP

```shell
cmake -Bbuild -H. -DMODEL=hip -DCMAKE_CXX_COMPILER=hipcc
cmake --build build
./build/hip-stream -s $((1024*1024*256))
```

### Fortran

```shell
cd src/fortran
make IMPLEMENTATION=OpenMPTarget COMPILER=flang GPUARCH=gfx90a FC=amdflang
./BabelStream.flang.OpenMPTarget -s $((1024*1024*256))
```

### Relative performance of Fortran OpenMP offloading BabelStream compared to HIP BabelStream

| Benchmark | MI250X | MI300A | MI300X |
| --------- | -----: | -----: | -----: |
| Copy      |    91% |    89% |    88% |
| Mul       |    91% |    92% |    88% |
| Add       |    91% |    97% |    95% |
| Triad     |    91% |    96% |    91% |
| Dot       |    98% |   113% |    89% |

> **NOTE** BabelStream was run with `-s $((1024*1024*256))` or 2 GiB arrays.
> Results may vary based array size.

## Preview of the AMD Next-gen Fortran compiler

We are excited to provide public access to a pre-production build of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)!
The compiler comes prebuilt for Ubuntu, RHEL, and SLES operating systems alongside some of the basic components of ROCm.
You will need a [supported AMD GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html),
an [existing install of ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), and the appropriate GPU
drivers in order to use the preview of the AMD Next-gen Fortran compiler. Installing is as easy as extracting the files
from the compressed archive for your operating system into a location of your choosing. Through the usual `PATH` and `LD_LIBRARY_PATH` variables,
the pre-production compiler and associated ROCm components can be used in concert with an existing ROCm installation on the system.

The latest preview of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) is available through [repo.redeon.com](https://repo.radeon.com/rocm/misc/flang/).
Download the file corresponding to your operating system.
You may untar your the file anywhere on your system.

```shell
tar jxf rocm-afar-<latest version>-<OS>.tar.bz2 -C <path to install>
```

It is then recommended to add `<path to install>/rocm-afar-<version>/bin` to your `PATH` and
`<path to install>/rocm-afar-<version>/lib` to `LD_LIBRARY_PATH`.
For a more complete reference on using the compiler, please consult our [user guide](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md).

## Summary

In this blog post we shared a preview of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md), showing how our new compiler can help you deploy and accelerate your Fortran codebase on AMD GPUs using OpenMP offloading, and how you can use it to interface with HIP and ROCm kernels. We demonstrated AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md)’s competitive performance against native HIP/C++ codes when running on AMD GPUs, and provided access to the pre-production build of our new compiler.
> **NOTE** Usage of the compiler comes with the following disclaimer:
>
> PRE-PRODUCTION SOFTWARE:  The software accessible on this page may be a pre-production version, intended to provide advance access to
> features that may or may not eventually be included into production version of the software.  Accordingly, pre-production software may
> not be fully functional, may contain errors, and may have reduced or different security, privacy, accessibility, availability, and reliability
> standards relative to production versions of the software. Use of pre-production software may result in unexpected results, loss of data,
> project delays or other unpredictable damage or loss.  Pre-production software is not intended for use in production, and your use of
> pre-production software is at your own risk.

When early-adopters of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md) encounter bugs and other issues,
they can report their findings to [AMD's fork of the LLVM project](https://github.com/ROCm/llvm-project/issues).
We encourage everyone to report issues and give feedback because we believe that user community involvement
will greatly improve the functionality and quality of AMD’s [Next-Gen Fortran Compiler](https://github.com/amd/InfinityHub-CI/blob/main/fortran/README.md). We are especially interested
in your needs and use cases for OpenMP offloading for AMD GPUs as you try out this new compiler.
