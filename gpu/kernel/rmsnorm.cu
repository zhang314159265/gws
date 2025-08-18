#include "pre.h"

__device__ float sumrow(bfloat16 *rowptr, int C) {
  float accum = 0.0;
  // for (int idx = threadIdx.x * 8; idx < C; idx += blockDim.x * 8) 
  int idx = threadIdx.x * 8;
  { // assumes C == blockDim.x * 8
    int4 xint4 = loadInt4(rowptr + idx, L1EVICT_LAST);

    for (int i = 0; i < 8; ++i) {
      float item = (float) ((bfloat16*)&xint4)[i];
      accum += item * item;
    }
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

extern "C" __global__ void rmsnorm_kernel(bfloat16 *xptr, bfloat16 *wptr, bfloat16 *optr, int C) {
  bfloat16 *rowptr = xptr + blockIdx.x * C;
  float eps = 1e-5;
  float rsqrt = 1.0 / sqrt(sumrow(rowptr, C) / C + eps);

  bfloat16 *outrowptr = optr + blockIdx.x * C;
  // for (int idx = threadIdx.x * 8; idx < C; idx += blockDim.x * 8) 
  int idx = threadIdx.x * 8;
  { // assumes C == blockDim.x * 8
    int4 xintx = loadInt4(rowptr + idx, L1EVICT_FIRST);
    int4 wintx = loadInt4(wptr + idx, L1EVICT_LAST);

    bfloat16 outpiece[8];
    for (int i = 0; i < 8; ++i) {
      float x = (float) ((bfloat16*)&xintx)[i];
      float w = (float) ((bfloat16*)&wintx)[i];
      float out = x * rsqrt * w;
      outpiece[i] = (bfloat16) out;
    }
    storeInt4(outrowptr + idx, *(int4 *) &outpiece);
  }
}
