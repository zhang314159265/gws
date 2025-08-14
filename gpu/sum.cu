#include "pre.h"

__device__ bfloat16 _sumrow(bfloat16 *rowptr, int N) {
  float accum = 0.0;
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    accum += (float) rowptr[i];
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  if (blockDim.x <= 32) {
    return accum;
  }

  // multi-warp reduction
  __shared__ float buffer[32]; // TODO use dynamic shared memory
  if (laneId == 0) {
    buffer[warpId] = accum;
  }

  if (warpId != 0) {
    return -1.0;
  }
  __syncthreads();
  accum = buffer[laneId];
  if (laneId >= blockDim.x / 32) {
    accum = 0;
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  return accum;
}

extern "C" __global__ void sum_kernel(bfloat16 *x, bfloat16 *y, int M, int N) {
  int rowIdx = blockIdx.x;
  bfloat16 *rowptr = x + rowIdx * N;
  bfloat16 out = _sumrow(rowptr, N);
  if (threadIdx.x == 0) {
    y[rowIdx] = out;
  }
}
