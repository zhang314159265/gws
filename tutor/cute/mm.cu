// follows cutlass/examples/cute/tutorial/sgemm_1.cu

#include "cute/tensor.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "../cuda/matmul_ref.cuh"
#include "../cuda/cublaslt_matmul.cuh"

template <class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class AThreadLayout,
    class TB, class BStride, class BSmemLayout, class BThreadLayout,
    class TC, class CStride, class CSmemLayout, class CThreadLayout,
    class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
    TB const *B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
    TC       *C, CStride dC, CSmemLayout          , CThreadLayout tC,
    Alpha alpha, Beta beta)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
  CUTE_STATIC_ASSERT_V(size(tA) == size(tC));

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) == size<0>(ASmemLayout{}));
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) == size<0>(CSmemLayout{}));
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) == size<0>(BSmemLayout{}));
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) == size<1>(CSmemLayout{}));
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(ASmemLayout{}));
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(BSmemLayout{}));

  // full tensor
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  // tile
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  // shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // parition A/B tiles across threads
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);  // (THR_M, THR_K, k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);  // (THR_M, THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);  // (THR_N, THR_K, k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);  // (THR_N, THR_K)

  // Define A/B partitioning and C accumulators
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M, BLK_K)
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X, _1>{}); // (THR_N, BLK_K)
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M, THR_N)
  Tensor tCrC = make_tensor_like(tCgC);

  // Clear the accumulators
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    gemm(tCsA, tCsB, tCrC); // c += a @ b

    __syncthreads(); // wait for all threads to read from smem
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
    class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
    Alpha alpha,
    TA const *A, int ldA,
    TB const *B, int ldB,
    Beta beta,
    TC *C, int ldC,
    cudaStream_t stream = 0) {
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  // thread layout
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
    size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
    prob_shape, cta_tiler,
    A, dA, sA, tA,
    B, dB, sB, tB,
    C, dC, sC, tC,
    alpha, beta
  );
}

template <class TA, class TB, class TC,
    class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
    Alpha alpha,
    TA const *A, int ldA,
    TB const *B, int ldB,
    Beta beta,
    TC *C, int ldC,
    cudaStream_t stream = 0) {
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
  auto sB = make_layout(make_shape(bN, bK), LayoutRight{});
  auto sC = make_layout(make_shape(bM, bN));

  // thread layout
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
    size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
    prob_shape, cta_tiler,
    A, dA, sA, tA,
    B, dB, sB, tB,
    C, dC, sC, tC,
    alpha, beta
  );
}

template <class TA, class TB, class TC,
    class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
    Alpha alpha,
    TA const *A, int ldA,
    TB const *B, int ldB,
    Beta beta,
    TC *C, int ldC,
    cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "not implemented");
}

#define matmul_ref_wrapper matmul_ref_wrapper_cublaslt

template <class TA, class TB, class TC,
    class Alpha, class Beta>
void matmul_ref_wrapper_cublaslt(char transA, char transB, int m, int n, int k,
    Alpha alpha,
    TA const *A, int ldA,
    TB const *B, int ldB,
    Beta beta,
    TC *C, int ldC,
    cudaStream_t stream = 0) {
  assert(alpha == 1.0f);
  assert(beta == 0.0f);
  // TODO: verify that ldA, ldB, ldC match m/n/k

  matmul_cublaslt(C, A, B, m, n, k, transA == 'T', transB == 'T');
}

template <class TA, class TB, class TC,
    class Alpha, class Beta>
void matmul_ref_wrapper_cuda(char transA, char transB, int m, int n, int k,
    Alpha alpha,
    TA const *A, int ldA,
    TB const *B, int ldB,
    Beta beta,
    TC *C, int ldC,
    cudaStream_t stream = 0) {

  int stride_am, stride_ak, stride_bk, stride_bn;

  if (transA == 'N') {
    stride_am = 1;
    stride_ak = m;
  } else if (transA == 'T') {
    stride_am = k;
    stride_ak = 1;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    stride_bk = 1;
    stride_bn = k;
  } else if (transB == 'T') {
    stride_bk = n;
    stride_bn = 1;
  } else {
    assert(false);
  }

  int stride_cm = 1;
  int stride_cn = m;
  matmul_ref(
    m, n, k,
    A, stride_am, stride_ak,
    B, stride_bk, stride_bn,
    C, stride_cm, stride_cn
  );
}

int main() {
  int m = 5120;
  int n = 5120;
  int k = 4096;
  char transA = 'N';
  char transB = 'T';

  if (getenv("TN")) {
    transA = 'T';
    transB = 'N';
  }

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta = 0.0;

  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  cute::device_init(0);

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);

  #if SDBG
  for (int j = 0; j < m * k; ++j) h_A[j] = j / 10.0;
  for (int j = 0; j < n * k; ++j) h_B[j] = j / 10.0;
  #else
  for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  #endif
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0 * m *n * k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;
  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  gemm(transA, transB, m, n, k,
    alpha,
    d_A.data().get(), ldA,
    d_B.data().get(), ldB,
    beta,
    d_C.data().get(), ldC);

  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  thrust::fill(d_C.begin(), d_C.end(), 0.0f);
  matmul_ref_wrapper(transA, transB, m, n, k,
    alpha,
    d_A.data().get(), ldA,
    d_B.data().get(), ldB,
    beta,
    d_C.data().get(), ldC
  );
  cudaDeviceSynchronize();
  thrust::host_vector<TC> ref_result = d_C;
  if (thrust::equal(ref_result.begin(), ref_result.end(), cute_result.begin())) {
    printf("Pass the numeric check\n");
  } else {
    printf("Fail the numeric check\n");
    int cnt = 0;
    for (int i = 0; i < ref_result.size(); ++i) {
      if (ref_result[i] != cute_result[i]) {
        printf("%d: ref %.3f, act %.3f\n", i, ref_result[i], cute_result[i]);
        ++cnt;
        if (cnt == 10) {
          break;
        }
      }
    }
    abort();
  }

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
      alpha,
      d_A.data().get(), ldA,
      d_B.data().get(), ldB,
      beta,
      d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();

  printf("CUTE_GEMM:    [%6.1f]GFlop/s (%6.4f)ms\n", gflops / cute_time, cute_time * 1000);
  return 0;
}
