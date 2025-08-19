// 1d thread tile: 5.863tflops/s -> 8.434tflops/s

#include "pre.h"

extern "C" __global__ void matmul_cuda_core_kernel_1d_thread_tile(
  bfloat16 *aptr, bfloat16 *bptr, bfloat16 *cptr,
  int M, int N, int K,
  int stride_a0, int stride_a1,
  int stride_b0, int stride_b1,
  int stride_c0, int stride_c1,
  int BSIZ
) {
  assert(BSIZ == 64);
  assert(blockDim.x == 512);
  int BM = BSIZ, BN = BSIZ;
  int BK = BSIZ / 2;

  __shared__ extern bfloat16 smem[];
  bfloat16 *asmem = &smem[0];
  bfloat16 *bsmem = &smem[0] + BM * BK;
  
  float accums[8] = {0.0};

  int start_row = blockIdx.x * BM;
  int start_col = blockIdx.y * BN;
  int nelem_per_thread = 8;
  assert(BM * BN == blockDim.x * nelem_per_thread);

  assert(M % BM == 0);
  assert(N % BN == 0);
  assert(K % BK == 0);

  // move pointers
  aptr = aptr + start_row * stride_a0;
  bptr = bptr + start_col * stride_b1;
  cptr = cptr + start_row * stride_c0 + start_col * stride_c1;

  int thread_start_row = threadIdx.x / BN * nelem_per_thread;
  int thread_col = threadIdx.x % BN;

  for (int k = 0; k < K; k += BK) {
    load_smem(asmem, aptr + k * stride_a1, stride_a0, stride_a1, BM, BK);
    load_smem(bsmem, bptr + k * stride_b0, stride_b0, stride_b1, BK, BN);

    __syncthreads();

    for (int l = 0; l < BK; ++l) {
      float belem = (float) bsmem[l * BN + thread_col];
      for (int i = 0; i < nelem_per_thread; ++i) {
        accums[i] += (float) asmem[(thread_start_row + i) * BK + l] * belem;
      }
    }

    __syncthreads();
  }

  __syncthreads();

  for (int i = 0; i < nelem_per_thread; ++i) {
    cptr[(thread_start_row + i) * stride_c0 + thread_col * stride_c1] = accums[i];
  }
}
