#include "pre.h"

__device__ float sumrow(bfloat16 *rowptr, int C) {
  float accum = 0.0;
  for (int idx = threadIdx.x; idx < C; idx += blockDim.x) {
    float item = (float) rowptr[idx];
    accum += item * item;
  }

  // intra warp shuffle
  for (int mask = 1; mask <= 16; mask *= 2) {
    accum += __shfl_xor_sync(0xffffffff, accum, mask);
  }

  // write to shared memory
  __shared__ extern float smem[];
  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;
  if (laneId == 0) {
    smem[warpId] = accum;
  }

  __syncthreads();

  // warp 0 do another round of shuffles and write the result back to the shared memory
  int num_warps = blockDim.x / 32;
  if (warpId == 0) {
    if (laneId < num_warps) {
      accum = smem[laneId];
    } else {
      accum = 0;
    }
    for (int mask = 1; mask < num_warps; mask *= 2) {
      accum += __shfl_xor_sync(0xffffffff, accum, mask);
    }
    if (laneId == 0) {
      smem[0] = accum;
    }
  }

  __syncthreads();
  return smem[0];
}

extern "C" __global__ void rmsnorm_kernel(bfloat16 *xptr, bfloat16 *wptr, bfloat16 *optr, int BT, int C) {
  bfloat16 *rowptr = xptr + blockIdx.x * C;
  double eps = 1e-5;
  float rsqrt = 1.0 / sqrt(sumrow(rowptr, C) / C + eps);

  bfloat16 *outrowptr = optr + blockIdx.x * C;
  for (int idx = threadIdx.x; idx < C; idx += blockDim.x) {
    float x = (float) rowptr[idx];
    float w = (float) wptr[idx];
    float out = x * rsqrt * w;
    outrowptr[idx] = (bfloat16) out;
  }
}
