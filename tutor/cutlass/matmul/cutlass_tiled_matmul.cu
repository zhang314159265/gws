#include <cute/tensor.hpp>

using namespace cute;

extern "C" __global__ void matmul_kernel(
    float *x, float *y, float *z,
    int M, int N, int K,
    int stride_x0, int stride_x1,
    int stride_y0, int stride_y1,
    int stride_z0, int stride_z1,
    int _bM, int _bN, int _bK) {
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  assert(bM == _bM);
  assert(bN == _bN);
  assert(bK == _bK);
  
  // XXX do uint128_t copy for copyA cause error in the copy call. Need debug more!
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
    Layout<Shape<_32, _8>>{},
    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
    Layout<Shape<_32, _8>>{},
    Layout<Shape<_4, _1>>{});

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<float>{},
    Layout<Shape<_16, _16, _1>>{});
 
  // XXX the following trick is needed to allow uint128_t copy for copyB
  #if 0
  auto mA = make_tensor(make_gmem_ptr(x), make_shape(M, K), make_stride(stride_x0, stride_x1));
  auto mB = make_tensor(make_gmem_ptr(y), make_shape(N, K), make_stride(stride_y0, stride_y1));
  #else
  assert(stride_x1 == 1);
  assert(stride_y0 == 1);
  auto mA = make_tensor(make_gmem_ptr(x), make_shape(M, K), make_stride(stride_x0, Int<1>{}));
  auto mB = make_tensor(make_gmem_ptr(y), make_shape(N, K), make_stride(Int<1>{}, stride_y1));
  #endif
  auto mC = make_tensor(make_gmem_ptr(z), make_shape(M, N), make_stride(stride_z0, stride_z1));

  auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);
  auto block_tiler = make_shape(bM, bN, bK);
  // (bM, bK, k)
  auto gA = local_tile(mA, block_tiler, block_coord, tuple<_1, X, _1>{});
  // (bN, bK, k)
  auto gB = local_tile(mB, block_tiler, block_coord, tuple<X, _1, _1>{});
  // (bM, bN)
  auto gC = local_tile(mC, block_tiler, block_coord, tuple<_1, _1, X>{});

  auto sA_layout = make_layout(make_shape(bM, bK));
  auto sB_layout = make_layout(make_shape(bN, bK));

  __shared__ float smemA[size_v<decltype(sA_layout)>];
  __shared__ float smemB[size_v<decltype(sB_layout)>];
  auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  ThrCopy thr_copy_b = copyB.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);

  ThrMMA thr_mma = mmaC.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);

  clear(tCrC);
  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    #if 0
    print_type(copyA);
    print_type(tAgA);
    print_type(tAsA);
    #endif
    copy(copyA, tAgA(_, _, _, k_tile), tAsA);
    copy(copyB, tBgB(_, _, _, k_tile), tBsB);
    __syncthreads();

    gemm(mmaC, tCsA, tCsB, tCrC); 
    __syncthreads();
  }
  copy(tCrC, tCgC);
}
