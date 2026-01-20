#include <cute/tensor_impl.hpp>

using namespace cute;

extern "C" __global__ void cutlass_kernel(float *x, float *y, int M, int N) {
  __shared__ extern float smem[];
  int row = blockIdx.x;

  constexpr int BLOCK_SIZE = 256;
  assert(blockDim.x == BLOCK_SIZE);
  Tensor gInput = make_tensor(
    make_gmem_ptr(x),
    make_shape(M, N),
    make_stride(N, 1));

  Tensor gRow = gInput(row, _);
  Tensor sPartials = make_tensor(
    make_smem_ptr(smem),
    make_shape(Int<BLOCK_SIZE>{})
  );

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    local_sum += gRow(i);
  }
  sPartials(threadIdx.x) = local_sum;
  __syncthreads();

  CUTE_UNROLL
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sPartials(threadIdx.x) += sPartials(threadIdx.x + stride);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[row] = sPartials[0];
  }
}
