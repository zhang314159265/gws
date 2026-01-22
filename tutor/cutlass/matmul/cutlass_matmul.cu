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

  auto mA = make_tensor(make_gmem_ptr(x), make_shape(M, K), make_stride(stride_x0, stride_x1));
  auto mB = make_tensor(make_gmem_ptr(y), make_shape(N, K), make_stride(stride_y0, stride_y1));
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
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  __shared__ float smemA[size_v<decltype(sA_layout)>];
  __shared__ float smemB[size_v<decltype(sB_layout)>];
  auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // partitioning
  auto tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THR_K, k)
  auto tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THR_K)

  auto tBgB = local_partition(gB, tB, threadIdx.x);
  auto tBsB = local_partition(sB, tB, threadIdx.x);

  // (THR_M, bK)
  auto tCsA = local_partition(sA, tC, threadIdx.x, tuple<_1, X>{});
  auto tCsB = local_partition(sB, tC, threadIdx.x, tuple<X, _1>{});
  auto tCgC = local_partition(gC, tC, threadIdx.x);
  auto tCrC = make_tensor_like(tCgC);

  auto mma = MMA_Atom<UniversalFMA<float, float, float, float>>{};

  clear(tCrC);
  for (int k_tile = 0; k_tile < size<2>(gA); ++k_tile) {
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);
    __syncthreads();
    #if 0
    gemm(tCsA, tCsB, tCrC);
    #else
    gemm(mma, tCsA, tCsB, tCrC);
    #endif
    __syncthreads();
  }
  copy(tCrC, tCgC);
}
