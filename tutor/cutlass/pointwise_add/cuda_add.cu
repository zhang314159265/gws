extern "C" void __global__ cuda_add_kernel(float *x, float *y, float *z, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    z[idx] = x[idx] + y[idx];
  }
}
