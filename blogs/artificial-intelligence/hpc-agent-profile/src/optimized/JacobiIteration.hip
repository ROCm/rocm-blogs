/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"

//Jacobi iterative method
// U = U + D^{-1}*(RHS - AU)
__global__ void JacobiIterationKernel(const int N,
                                 const dfloat dx,
                                 const dfloat dy,
                                 const dfloat *__restrict__ RHS,
                                 const dfloat *__restrict__ AU,
                                       dfloat *__restrict__ RES,
                                       dfloat *__restrict__ U) {

  const int tid = threadIdx.x;
  const int blockSize = blockDim.x;
  const int id = blockIdx.x*blockSize + tid;

  // Pre-compute the denominator to avoid repeated division
  const dfloat inv_diag = 1.0 / (2.0/(dx*dx) + 2.0/(dy*dy));

  // Process multiple elements per thread for better occupancy
  const int stride = gridDim.x * blockSize;
  
  for (int idx = id; idx < N; idx += stride) {
    dfloat r_res = RHS[idx] - AU[idx];
    RES[idx] = r_res;
    U[idx] += r_res * inv_diag;
  }
}

void JacobiIteration(grid_t& grid, mesh_t& mesh,
                     hipStream_t stream,
                     dfloat* d_RHS,
                     dfloat* d_AU,
                     dfloat* d_RES,
                     dfloat* d_U) {

  // Optimized thread block size for better occupancy
  int xthreads = 256;
  
  // Calculate optimal number of blocks for better GPU utilization
  int numBlocks = (mesh.N + xthreads - 1) / xthreads;
  // Ensure sufficient blocks for good occupancy
  numBlocks = max(numBlocks, 64);

  dim3 threads(xthreads,1,1);
  dim3 blocks(numBlocks, 1, 1);

  hipLaunchKernelGGL(JacobiIterationKernel,
                     blocks,
                     threads,
                     0, stream,
                     mesh.N,
                     mesh.dx, mesh.dy,
                     d_RHS, d_AU, d_RES, d_U);
}
