#include <cute/tensor.hpp>  // tensor.hpp rather than tensor_impl.hpp for copy

using namespace cute;

extern "C" __global__ void cutlass_kernel(float *x, float *y, int M, int N) {
  auto gS = make_tensor(make_gmem_ptr(x), make_layout(make_shape(M, N), make_stride(N, 1)));
  auto gD = make_tensor(make_gmem_ptr(y), make_layout(make_shape(M, N), make_stride(N, 1)));
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto block_idx = make_coord(blockIdx.x, blockIdx.y);
  auto thread_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));

  auto tile_S = local_tile(gS, block_shape, block_idx);
  auto tile_D = local_tile(gD, block_shape, block_idx);

  auto thr_tile_S = local_partition(tile_S, thread_layout, threadIdx.x);
  auto thr_tile_D = local_partition(tile_D, thread_layout, threadIdx.x);

  #if 0
  auto fragment = make_tensor_like(thr_tile_S);
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
  #else
  copy(thr_tile_S, thr_tile_D);
  #endif
}
