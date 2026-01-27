#include "cuda_bf16.h"

using bfloat16 = __nv_bfloat16;

extern "C" __global__ void add(bfloat16 *aptr, bfloat16 *bptr, bfloat16 *optr, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tot = gridDim.x * blockDim.x;
  for (; idx < N; idx += tot) {
    bfloat16 a = aptr[idx];
    bfloat16 b = bptr[idx];
    optr[idx] = a + b;
  }
}
