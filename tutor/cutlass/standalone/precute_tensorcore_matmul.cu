// Pass -arch=sm_100a to nvcc is important (even though the arch used in code is sm80)
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

template <typename scalar_t>
void reference_gemm_host(
    scalar_t const* A,
    scalar_t const* B,
    scalar_t const* C,
    scalar_t* D,
    int M, int N, int K,
    float alpha, float beta) {

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = static_cast<float>(A[m * K + k]);
        float b = static_cast<float>(B[n * K + k]);
        acc += a * b;
      }
      float c = static_cast<float>(C[m * N + n]);
      D[m * N + n] = static_cast<scalar_t>(alpha * acc + beta * c);
    }
  }
}



template <typename scalar_t>
bool verify_results(
    scalar_t const* computed,
    scalar_t const* reference,
    int M, int N,
    float tolerance = 0.01f) {

  int errors = 0;
  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;
  int max_error_m = 0, max_error_n = 0;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float comp = static_cast<float>(computed[m * N + n]);
      float ref = static_cast<float>(reference[m * N + n]);

      float abs_diff = std::abs(comp - ref);
      float rel_diff = abs_diff / (std::abs(ref) + 1e-6f);

      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
        max_rel_diff = rel_diff;
        max_error_m = m;
        max_error_n = n;
      }

      if (rel_diff > tolerance && abs_diff > 1e-5f) {
        if (errors < 10) {
          std::cout << "  Error at (" << m << ", " << n << "): "
                    << "computed=" << comp << ", reference=" << ref
                    << ", diff=" << abs_diff << std::endl;
        }
        errors++;
      }
    }
  }

  std::cout << "Max absolute diff: " << max_abs_diff
            << " at (" << max_error_m << ", " << max_error_n << ")" << std::endl;
  std::cout << "Max relative diff: " << max_rel_diff << std::endl;
  std::cout << "Total errors: " << errors << " / " << (M * N) << std::endl;

  return errors == 0;
}

template <
    typename scalar_t_,
    typename ArchTag_,
    int kTileM,
    int kTileN,
    int kTileK>
struct GemmKernelConfig {

  // Expose template parameters as member types
  using scalar_t = scalar_t_;
  using ArchTag = ArchTag_;
  using accum_t = float;
  using output_t = scalar_t;

  // Layouts
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // Alignment
  static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<scalar_t>::value;
  static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<scalar_t>::value;
  static constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<output_t>::value;

  static constexpr int kWarpSize = 32;

  // Tile shapes
  using ThreadblockShape = cutlass::gemm::GemmShape<kTileM, kTileN, kTileK>;

  using WarpShape = cutlass::gemm::GemmShape<32, 32, kTileK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Operator
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr int kStages = 3;
  using Operator = cutlass::arch::OpMultiplyAdd;

  // MMA
  using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
      scalar_t, LayoutA, kAlignmentA,
      scalar_t, LayoutB, kAlignmentB,
      accum_t, LayoutC,
      OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      kStages, Operator
  >;

  using Mma = typename DefaultMma::ThreadblockMma;
  using IteratorA = typename DefaultMma::IteratorA;
  using IteratorB = typename DefaultMma::IteratorB;
  using MmaCore = typename DefaultMma::MmaCore;

  // Threads
  static constexpr int kThreadCount = MmaCore::kThreads;

  // Epilogue
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      output_t, kAlignmentC, accum_t, accum_t
  >;

  using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape,
      typename Mma::Operator,
      ThreadblockShape::kK / WarpShape::kK,
      EpilogueOutputOp,
      EpilogueOutputOp::kCount
  >;

  using Epilogue = typename DefaultEpilogue::Epilogue;
  using OutputTileIterator = typename Epilogue::OutputTileIterator;

  // Shared memory
  struct SharedStorage {
    typename Mma::SharedStorage mma;
    typename Epilogue::SharedStorage epilogue;
  };
};

template <typename Config>
struct GemmKernel {

  using scalar_t = typename Config::scalar_t;
  using output_t = typename Config::output_t;
  using accum_t = typename Config::accum_t;

  struct Params {
    scalar_t const* A;
    scalar_t const* B;
    output_t const* C;
    output_t* D;
    int M, N, K;
    int lda, ldb, ldc, ldd;
    accum_t alpha;
    accum_t beta;
  };

  static __device__ void run(
      Params const& params,
      typename Config::SharedStorage& shared_storage) {

    using Mma = typename Config::Mma;
    using Epilogue = typename Config::Epilogue;
    using IteratorA = typename Config::IteratorA;
    using IteratorB = typename Config::IteratorB;
    using OutputTileIterator = typename Config::OutputTileIterator;
    using EpilogueOutputOp = typename Config::EpilogueOutputOp;

    int thread_idx = threadIdx.x;
    int warp_idx = thread_idx / Config::kWarpSize;
    int lane_idx = thread_idx % Config::kWarpSize;


    int block_m = blockIdx.x * Config::ThreadblockShape::kM;
    int block_n = blockIdx.y * Config::ThreadblockShape::kN;

    cutlass::MatrixCoord extent_A{params.M - block_m, params.K};
    cutlass::MatrixCoord extent_B{params.K, params.N - block_n};
    cutlass::MatrixCoord extent_C{params.M - block_m, params.N - block_n};

    typename IteratorA::Params params_A(
        typename Config::MmaCore::LayoutA(params.lda));
    typename IteratorB::Params params_B(
        typename Config::MmaCore::LayoutB(params.ldb));

    IteratorA iterator_A(
        params_A,
        const_cast<scalar_t*>(params.A) + block_m * params.lda,
        extent_A,
        thread_idx,
        cutlass::MatrixCoord{0, 0});

    IteratorB iterator_B(
        params_B,
        const_cast<scalar_t*>(params.B) + block_n * params.ldb,
        extent_B,
        thread_idx,
        cutlass::MatrixCoord{0, 0});

    Mma mma(shared_storage.mma, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accum;
    accum.clear();

    int gemm_k_iterations = (params.K + Mma::Shape::kK - 1) / Mma::Shape::kK;
    mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

    __syncthreads();

    typename EpilogueOutputOp::Params epilogue_op_params{params.alpha, params.beta};
    EpilogueOutputOp epilogue_op(epilogue_op_params);

    typename OutputTileIterator::Params params_D(params.ldd);
    OutputTileIterator iterator_D(
        params_D,
        params.D + block_m * params.ldd + block_n,
        extent_C,
        thread_idx);

    typename OutputTileIterator::Params params_C(params.ldc);
    OutputTileIterator iterator_C(
        params_C,
        const_cast<output_t*>(params.C) + block_m * params.ldc + block_n,
        extent_C,
        thread_idx);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(epilogue_op, iterator_D, accum, iterator_C);
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadCount)
gemm_kernel(typename GemmKernel<Config>::Params params) {
  extern __shared__ char smem[];
  auto& shared_storage = *reinterpret_cast<typename Config::SharedStorage*>(smem);
  GemmKernel<Config>::run(params, shared_storage);
}



template <typename scalar_t>
cudaError_t launch_gemm(
    scalar_t const* A,
    scalar_t const* B,
    scalar_t const* C,
    scalar_t* D,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr) {

  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 32;

  using Config = GemmKernelConfig<scalar_t, cutlass::arch::Sm80, kTileM, kTileN, kTileK>;

  typename GemmKernel<Config>::Params params;
  params.A = A;
  params.B = B;
  params.C = C;
  params.D = D;
  params.M = M;
  params.N = N;
  params.K = K;
  params.lda = K;
  params.ldb = K;
  params.ldc = N;
  params.ldd = N;
  params.alpha = alpha;
  params.beta = beta;

  dim3 grid(
      (M + kTileM - 1) / kTileM,
      (N + kTileN - 1) / kTileN,
      1);
  dim3 block(Config::kThreadCount, 1, 1);
  size_t smem_size = sizeof(typename Config::SharedStorage);

  gemm_kernel<Config><<<grid, block, smem_size, stream>>>(params);

  return cudaGetLastError();
}



int main() {
  int M = 256, N = 256, K = 128;
  float alpha = 1.0f;
  float beta = 0.0f;

  using scalar_t = cutlass::half_t;

  std::cout << "GEMM Verification Test" << std::endl;
  std::cout << "=======================" << std::endl;
  std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl << std::endl;

  std::vector<scalar_t> h_A(M * K);
  std::vector<scalar_t> h_B(K * N);
  std::vector<scalar_t> h_C(M * N);
  std::vector<scalar_t> h_D_computed(M * N);
  std::vector<scalar_t> h_D_reference(M * N);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<scalar_t>(dist(rng));
  for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<scalar_t>(dist(rng));
  for (int i = 0; i < M * N; ++i) h_C[i] = static_cast<scalar_t>(dist(rng));

  std::cout << "Computing reference GEMM on CPU..." << std::endl;
  reference_gemm_host(h_A.data(), h_B.data(), h_C.data(), h_D_reference.data(),
                      M, N, K, alpha, beta);

  scalar_t *d_A, *d_B, *d_C, *d_D;
  cudaMalloc(&d_A, M * K * sizeof(scalar_t));
  cudaMalloc(&d_B, K * N * sizeof(scalar_t));
  cudaMalloc(&d_C, M * N * sizeof(scalar_t));
  cudaMalloc(&d_D, M * N * sizeof(scalar_t));

  cudaMemcpy(d_A, h_A.data(), M * K * sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), K * N * sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C.data(), M * N * sizeof(scalar_t), cudaMemcpyHostToDevice);

  std::cout << "Computing GEMM on GPU with CUTLASS..." << std::endl;
  cudaError_t status = launch_gemm(d_A, d_B, d_C, d_D, M, N, K, alpha, beta);

  if (status != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  cudaMemcpy(h_D_computed.data(), d_D, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost);

  std::cout << std::endl << "Verification Results:" << std::endl;
  std::cout << "---------------------" << std::endl;

  bool passed = verify_results(h_D_computed.data(), h_D_reference.data(), M, N, 0.02f);

  std::cout << std::endl;
  std::cout << (passed ? "*** PASSED ***" : "*** FAILED ***") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return passed ? 0 : 1;
}


