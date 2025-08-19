#include "pre.h"

extern "C" __global__ void matmul_cuda_kernel_cuda_core_accum_in_regs(
  bfloat16 *aptr, bfloat16 *bptr, bfloat16 *cptr,
  int M, int N, int K,
  int stride_a0, int stride_a1,
  int stride_b0, int stride_b1,
  int stride_c0, int stride_c1,
  int BSIZ
) {
  __shared__ extern bfloat16 smem[];
  bfloat16 *asmem = &smem[0];
  bfloat16 *bsmem = &smem[0] + BSIZ * BSIZ;
  // assert(blockDim.x * 4 == BSIZ * BSIZ);
  assert(BSIZ == 64);
  assert(blockDim.x == 1024);
  
  float accums[4] = {0.0, 0.0, 0.0, 0.0};

  int start_row = blockIdx.x * BSIZ;
  int start_col = blockIdx.y * BSIZ;

  assert(M % BSIZ == 0);
  assert(N % BSIZ == 0);
  assert(K % BSIZ == 0);

  // move pointers
  aptr = aptr + start_row * stride_a0;
  bptr = bptr + start_col * stride_b1;
  cptr = cptr + start_row * stride_c0 + start_col * stride_c1;

  for (int k = 0; k < K; k += BSIZ) {
    load_smem(asmem, aptr + k * stride_a1, stride_a0, stride_a1, BSIZ, BSIZ);
    load_smem(bsmem, bptr + k * stride_b0, stride_b0, stride_b1, BSIZ, BSIZ);

    __syncthreads();

    // dot product two blocks from shared memory
    for (int i = 0; i < 4; ++i) {
      int flat_idx = threadIdx.x + i * blockDim.x;
      int br = flat_idx / BSIZ;
      int bc = flat_idx % BSIZ;

      float accum = 0.0;
      for (int l = 0; l < BSIZ; ++l) {
        accum += (float) asmem[br * BSIZ + l] * (float) bsmem[l * BSIZ + bc];
      }
      accums[i] += accum;
    }
  }

  __syncthreads();

  // store csmem to the buffer pointed by cptr
  for (int i = 0; i < 4; ++i) {
    int flat_idx = threadIdx.x + i * blockDim.x;
    int br = flat_idx / BSIZ;
    int bc = flat_idx % BSIZ;
    cptr[br * stride_c0 + bc * stride_c1] = accums[i];
  }
}
