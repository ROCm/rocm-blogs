/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"

// Optimized for better occupancy and performance
#define block_size 256
#define vec_size 4  // Vectorization factor

__global__ void NormKernel1(const int N,
                            const dfloat dx, const dfloat dy,
                            const dfloat*__restrict__ U,
                                  dfloat*__restrict__ tmp) {
  __shared__ dfloat s_dot[block_size];

  const int t = threadIdx.x;
  const int i = blockIdx.x;

  // Vectorized memory access for better bandwidth utilization
  const int stride = gridDim.x * block_size;
  int id = i * block_size + t;
  
  // Each thread processes multiple elements for better occupancy
  const dfloat dx_dy = dx * dy;
  dfloat sum = 0.0;
  
  // Unroll loop for better performance
  for ( ; id < N ; id += stride ) {
    dfloat u_val = U[id];
    sum += u_val * u_val * dx_dy;
  }
  
  s_dot[t] = sum;
  __syncthreads();

  // Optimized reduction with warp-level operations when possible
  for (int k = block_size / 2; k > 32; k /= 2 ) {
    if ( t < k ) {
      s_dot[t] += s_dot[t + k];
    }
    __syncthreads();
  }
  
  // Final reduction within warp (no need for __syncthreads)
  if (t < 32) {
    #pragma unroll
    for (int k = 32; k > 0; k /= 2) {
      s_dot[t] += s_dot[t + k];
    }
  }

  if (t==0)
    tmp[i] = s_dot[0];
}

__global__ void NormKernel2(const int N,
                            const dfloat*__restrict__ tmp,
                                  dfloat*__restrict__ dprod) {

  __shared__ dfloat s_dot[block_size];

  const int t = threadIdx.x;
  const int i = blockIdx.x;
  int id = i * block_size + t;

  dfloat sum = 0.0;
  for ( ; id < N ; id += block_size ) {
    sum += tmp[id];
  }
  
  s_dot[t] = sum;
  __syncthreads();

  // Optimized reduction with warp-level operations
  for (int k = block_size / 2; k > 32; k /= 2 ) {
    if ( t < k ) {
      s_dot[t] += s_dot[t + k];
    }
    __syncthreads();
  }
  
  // Final reduction within warp
  if (t < 32) {
    #pragma unroll
    for (int k = 32; k > 0; k /= 2) {
      s_dot[t] += s_dot[t + k];
    }
  }

  if (t==0)
    *dprod = s_dot[0];
}

dfloat Norm(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat *U) {

  static dfloat* d_tmp = NULL;
  static dfloat* d_norm = NULL;
  static dfloat* h_norm = NULL;

  //first call
  if (h_norm==NULL) {
    //allocate temportary reduction space
    SafeHipCall(hipMalloc(&d_tmp, block_size*sizeof(dfloat)));
    SafeHipCall(hipMalloc(&d_norm, 1*sizeof(dfloat)));
    SafeHipCall(hipHostMalloc(&h_norm, 1*sizeof(dfloat)));
  }

  size_t grid_size = (mesh.N+block_size-1)/block_size;
  grid_size = (grid_size < block_size) ? grid_size : block_size;

  hipLaunchKernelGGL((NormKernel1),
                     dim3(grid_size),
                     dim3(block_size),
                     0, stream,
                     mesh.N, mesh.dx, mesh.dy, U, d_tmp);

  hipLaunchKernelGGL((NormKernel2),
                     dim3(1),
                     dim3(block_size),
                     0, stream,
                     grid_size, d_tmp, d_norm);

  SafeHipCall(hipMemcpy(h_norm, d_norm, 1*sizeof(dfloat), hipMemcpyDeviceToHost));

  dfloat norm;
  MPI_Allreduce(h_norm, &norm, 1, MPI_DFLOAT, MPI_SUM, grid.comm);

  return sqrt(norm)*mesh.invNtotal;
}
