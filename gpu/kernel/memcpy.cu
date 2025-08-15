extern "C" __global__ void memcpy_kernel(float *psrc, float *pdst, int N) {
  int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = gridDim.x * blockDim.x;
  for (int idx = startIdx; idx < N; idx += total) {
    pdst[idx] = psrc[idx];
  }
}
