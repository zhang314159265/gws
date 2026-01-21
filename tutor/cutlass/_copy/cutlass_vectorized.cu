#include <cute/tensor.hpp>

using namespace cute;

extern "C" __global__ void cutlass_kernel(float *x, float *y, int _M, int _N) {
  // either branch works.
  #if 0
  auto M = Int<256>{};
  auto N = Int<512>{};
  #else
  int M = _M;
  int N = _N;
  #endif
  assert(M == _M);
  assert(N == _N);

  #if 0
  // Use row major layout will fail.
  auto gS = make_tensor(make_gmem_ptr(x), make_layout(make_shape(M, N), make_stride(N, Int<1>{})));
  auto gD = make_tensor(make_gmem_ptr(y), make_layout(make_shape(M, N), make_stride(N, Int<1>{})));
  #else
  // ***MUST*** use Int<1>{} rather than '1' directly.
  auto gS = make_tensor(make_gmem_ptr(x), make_layout(make_shape(M, N), make_stride(Int<1>{}, M)));
  auto gD = make_tensor(make_gmem_ptr(y), make_layout(make_shape(M, N), make_stride(Int<1>{}, M)));
  #endif
  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto block_idx = make_coord(blockIdx.x, blockIdx.y);
  auto thread_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  auto tile_S = local_tile(gS, block_shape, block_idx);
  auto tile_D = local_tile(gD, block_shape, block_idx);

  using CopyOp = UniversalCopy<uint_byte_t<16>>; // 16 bytes
  using Atom = Copy_Atom<CopyOp, float>;
  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thread_layout, val_layout);

  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  Tensor thr_tile_S = thr_copy.partition_S(tile_S);
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);

  // either branch works.
  #if 0
  Tensor fragment = make_fragment_like(thr_tile_D);
  #else
  Tensor fragment = make_fragment_like(thr_tile_S);
  #endif

  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}
