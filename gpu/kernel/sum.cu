#include "pre.h"

template <int num_warps>
__device__ bfloat16 _sumrow(__restrict__ bfloat16 *rowptr, int N) {
  float accum = 0.0;
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  assert(N % 8 == 0);
  for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
    int4 vec = __ldg((int4*) (&rowptr[i]));
    bfloat16 *sub = (bfloat16 *) &vec;
    for (int j = 0; j < 8; ++j) {
      accum += (float) sub[j];
    }
  }

  // warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  if (blockDim.x <= 32) {
    return accum;
  }

  // multi-warp reduction
  #if 1
  extern __shared__ float buffer[];
  #else
  __shared__ float buffer[32];
  #endif
  if (laneId == 0) {
    buffer[warpId] = accum;
  }

  if (warpId != 0) {
    return -1.0;
  }
  __syncthreads();

  if (laneId < num_warps) {
    accum = buffer[laneId];
  } else {
    accum = 0;
  }

  // warp shuffle
  for (int mask = 1; mask < num_warps; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  return accum;
}

extern "C" __global__ void sum_kernel(__restrict__ bfloat16 *x, __restrict__ bfloat16 *y, int M, int N) {
  int rowIdx = blockIdx.x;
  bfloat16 *rowptr = x + rowIdx * N;
  int num_warps = blockDim.x / 32;
  assert(num_warps > 0 && ((num_warps & (num_warps - 1)) == 0));

  bfloat16 out;
  switch (num_warps) {
  // case 8: out = _sumrow<8>(rowptr, N); break;
  // case 16: out = _sumrow<16>(rowptr, N); break;
  case 32: out = _sumrow<32>(rowptr, N); break;
  default:
    assert(false && "miss specialization for the number of warps");
  }
  if (threadIdx.x == 0) {
    y[rowIdx] = out;
  }
}
