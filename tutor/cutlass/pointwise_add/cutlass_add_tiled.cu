#include <cute/tensor_impl.hpp>

using namespace cute;


extern "C" __global__ void cutlass_kernel(float *A, float *B, float *C, int M, int N) {
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto thread_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto block_coord = make_coord(blockIdx.x, blockIdx.y);

  auto gtensor_layout = make_layout(make_shape(M, N), make_stride(N, 1));
  auto gA = make_tensor(make_gmem_ptr(A), gtensor_layout);
  auto gB = make_tensor(make_gmem_ptr(B), gtensor_layout);
  auto gC = make_tensor(make_gmem_ptr(C), gtensor_layout);

  auto tileA = local_tile(gA, block_shape, block_coord);
  auto tileB = local_tile(gB, block_shape, block_coord);
  auto tileC = local_tile(gC, block_shape, block_coord);

  auto thrA = local_partition(tileA, thread_layout, threadIdx.x);
  auto thrB = local_partition(tileB, thread_layout, threadIdx.x);
  auto thrC = local_partition(tileC, thread_layout, threadIdx.x);

  CUTE_UNROLL
  for (int i = 0; i < size(thrA); ++i) {
    thrC(i) = thrA(i) + thrB(i);
  }
}
