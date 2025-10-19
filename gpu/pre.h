#include "cuda_bf16.h"

using bfloat16 = __nv_bfloat16;

enum L1EVICT {
  L1EVICT_NONE,
  L1EVICT_FIRST,
  L1EVICT_LAST,
};

__device__ int4 loadInt4(void *ptr, int l1evict) {
  switch (l1evict) {
  case L1EVICT_NONE:
    return *(int4*) ptr;
  case L1EVICT_FIRST: {
    int4 ans;
    asm(
      "ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
      : "=r"(ans.x), "=r"(ans.y), "=r"(ans.z), "=r"(ans.w)
      : "l"(ptr)
    );
    return ans;
  }
  case L1EVICT_LAST: {
    int4 ans;
    asm(
      "ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
      : "=r"(ans.x), "=r"(ans.y), "=r"(ans.z), "=r"(ans.w)
      : "l"(ptr)
    );
    return ans;
  }
  default:
    assert(false);
  }
  return int4{0, 0, 0, 0}; // can not reach here
}

__device__ void storeInt4(void *ptr, int4 content) {
  *(int4*) ptr = content;
}

// clear the shared memory using a single block of threads
__device__ void clear_smem(float *smem, int N) {
  assert(blockDim.y == 1 && blockDim.z == 1);
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    smem[i] = 0.0;
  }
}

// TODO: can leverage the case when stride0 or stride1 is 1
__device__ void load_smem(bfloat16 *smem, bfloat16 *gmem, int stride0, int stride1, int M, int N) {
  // load a block from global memory to shared memory.
  assert(blockDim.y == 1 && blockDim.z == 1);

  // load 2 bfloat16 per thread
  for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
    int r = i / N;
    int c = i % N;
    bfloat16 *gaddr = gmem + r * stride0 + c * stride1;
    smem[r * N + c] = *gaddr;
  }
}

__device__ void store_smem(float *smem, bfloat16 *gmem, int stride0, int stride1, int M, int N) {
  // store a block from shared memory to global memory.
  assert(blockDim.y == 1 && blockDim.z == 1);

  for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
    int r = i / N;
    int c = i % N;
    bfloat16 *gaddr = gmem + r * stride0 + c * stride1;
    *gaddr = (bfloat16) smem[r * N + c];
  }
}

__device__ void dot(bfloat16 *aptr, bfloat16 *bptr, int M, int N, int K, float *accumptr) {
  // abptr, bptr, accumptr are row-major in shared memory
  assert(blockDim.y == 1 && blockDim.z == 1);

  for (int i = threadIdx.x; i < M * N; i += blockDim.x) {
    int r = i / N;
    int c = i % N;

    float accum = 0;
    for (int k = 0; k < K; ++k) {
      accum += (float) aptr[r * K + k] * (float) bptr[k * N + c];
    }
    accumptr[r * N + c] += accum;
  }
}
