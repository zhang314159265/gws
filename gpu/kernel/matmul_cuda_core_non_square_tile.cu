#include "pre.h"

extern "C" __global__ void matmul_cuda_kernel_cuda_core_non_square_tile(
  bfloat16 *aptr, bfloat16 *bptr, bfloat16 *cptr,
  int M, int N, int K,
  int stride_a0, int stride_a1,
  int stride_b0, int stride_b1,
  int stride_c0, int stride_c1,
  int BSIZ
) {
  // assert(blockDim.x * 4 == BSIZ * BSIZ);
  assert(BSIZ == 64);
  assert(blockDim.x == 1024);
  int KSIZ = BSIZ / 2;

  __shared__ extern bfloat16 smem[];
  bfloat16 *asmem = &smem[0];
  bfloat16 *bsmem = &smem[0] + BSIZ * KSIZ;
  
  float accums[4] = {0.0, 0.0, 0.0, 0.0};

  int start_row = blockIdx.x * BSIZ;
  int start_col = blockIdx.y * BSIZ;

  assert(M % BSIZ == 0);
  assert(N % BSIZ == 0);
  assert(K % KSIZ == 0);

  // move pointers
  aptr = aptr + start_row * stride_a0;
  bptr = bptr + start_col * stride_b1;
  cptr = cptr + start_row * stride_c0 + start_col * stride_c1;

  for (int k = 0; k < K; k += KSIZ) {
    load_smem(asmem, aptr + k * stride_a1, stride_a0, stride_a1, BSIZ, KSIZ);
    load_smem(bsmem, bptr + k * stride_b0, stride_b0, stride_b1, KSIZ, BSIZ);

    __syncthreads();

    // dot product two blocks from shared memory
    for (int i = 0; i < 4; ++i) {
      int flat_idx = threadIdx.x + i * blockDim.x;
      int br = flat_idx / BSIZ;
      int bc = flat_idx % BSIZ;

      float accum = 0.0;
      for (int l = 0; l < KSIZ; ++l) {
        accum += (float) asmem[br * KSIZ + l] * (float) bsmem[l * BSIZ + bc];
      }
      accums[i] += accum;
    }
    __syncthreads();
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
