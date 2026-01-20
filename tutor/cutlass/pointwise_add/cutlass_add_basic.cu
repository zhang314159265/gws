#include <cute/tensor_impl.hpp>

using namespace cute;

extern "C" __global__ void cutlass_kernel(float *x, float *y, float *z, int N) {
  auto tx = make_tensor(make_gmem_ptr(x), make_shape(N));
  auto ty = make_tensor(make_gmem_ptr(y), make_shape(N));
  auto tz = make_tensor(make_gmem_ptr(z), make_shape(N));
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    tz[idx] = tx[idx] + ty[idx];
  }
}
