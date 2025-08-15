
extern "C" __global__ void memcpy_kernel(float * __restrict__ psrc, float * __restrict__ pdst, int N) {
  assert(N % 4 == 0);
  int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = gridDim.x * blockDim.x;
  for (int idx = startIdx * 4; idx < N; idx += total * 4) {
    *(float4 *) (pdst + idx) = __ldg((float4 *) (psrc + idx));
  }
}
