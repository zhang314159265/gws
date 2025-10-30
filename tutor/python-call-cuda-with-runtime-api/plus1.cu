#include <iostream>

__global__ void plus1_kernel(float *src, float *dst, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    dst[tid] = src[tid] + 1;
  }
}

void plus1(long _src, long _dst, int N) {
  float *src = (float *) _src;
  float *dst = (float *) _dst;
  int nthread = 256;
  int nblock = (N + nthread - 1) / nthread;
  plus1_kernel<<<nblock, nthread>>>(src, dst, N);
}


