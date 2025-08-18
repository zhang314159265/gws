#include "pre.h"

extern "C" __global__ void matmul_cuda_kernel_cuda_core(
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
  float *csmem = (float *) (&smem[0] + BSIZ * BSIZ * 2);

  clear_smem(csmem, BSIZ * BSIZ);

  int start_row = blockIdx.x * BSIZ;
  int start_col = blockIdx.y * BSIZ;

  assert(M % BSIZ == 0);
  assert(N % BSIZ == 0);
  assert(K % BSIZ == 0);

  for (int k = 0; k < K; k += BSIZ) {
    load_smem(asmem, aptr + start_row * stride_a0 + k * stride_a1, stride_a0, stride_a1, BSIZ, BSIZ);
    load_smem(bsmem, bptr + k * stride_b0 + start_col * stride_b1, stride_b0, stride_b1, BSIZ, BSIZ);

    __syncthreads();

    // dot product two blocks from shared memory
    dot(asmem, bsmem, BSIZ, BSIZ, BSIZ, csmem);
  }

  __syncthreads();

  // store csmem to the buffer pointed by cptr
  store_smem(csmem, cptr + start_row * stride_c0 + start_col * stride_c1,
      stride_c0, stride_c1, BSIZ, BSIZ);
}
