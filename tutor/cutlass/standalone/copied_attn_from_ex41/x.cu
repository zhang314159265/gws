/*! \file
    \brief CUTLASS Attention Example.

    This workload computes a fused multi head attention.
    Because it keeps the attention matrix in shared memory, it's both faster and
    uses less global memory.

    This is based on `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_,
    and very similar to `"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" <https://arxiv.org/abs/2205.14135>`_.

    Algorithm:
      In short, we can compute the output incrementally in blocks of size B,
      we just need to divide the final result by the sum of all coefficients in
      the softmax (which we compute incrementally) with the following pseudo-code:

      ```
      s_prime = torch.zeros([num_queries, B])
      O = torch.zeros([num_queries, head_size_v])
      for i in range(0, K.shape[0], B):
        si = exp((Q . K[i * B:(i+1) * B].t) * scale)
        sum_coefs += attn_unscaled.sum(-1)
        O  += si . V[i * B:(i+1) * B]
      O = O / s_prime
      ```

      In practice, and for numerical stability reasons,
      we also subtract the maximum so far (`mi`) before doing
      the exponential. When we encounter new keys, the maximum
      used to compute O so far (`m_prime`) can differ from the
      current maximum, so we update O before accumulating with

      ```
      O       = O * exp(m_prime - mi)
      m_prime = mi
      ```

    Implementation details:
      - `si` is stored in shared memory between the 2 back to back gemms
      - we keep and accumulate the output
      directly in registers if we can (`head_size_v <= 128`).
      Otherwise, we store it & accumulate in global memory (slower)
      - blocks are parallelized across the batch dimension, the number
      of heads, and the query sequence size


    Examples:

      # Run an attention example with default setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen

      # Run an attention example with custom setup
      $ ./examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen --head_number=2 --batch_size=3 --head_size=32 --head_size_v=64 --seq_length=512 --seq_length_kv=1024 --causal=true

      Acknowledgement: Fixed-sequence-length FMHA code was upstreamed by Meta xFormers (https://github.com/facebookresearch/xformers).
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"
#include "myimpl/kernel_forward.h"

#include "myimpl/x.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

template <typename Attention>
class TestbedAttention {
public:

  //
  // Type definitions
  //

  using ElementQ = typename Attention::scalar_t;
  using ElementK = typename Attention::scalar_t;
  using ElementP = typename Attention::accum_t;
  using ElementAccumulator = typename Attention::accum_t;
  using ElementV = typename Attention::scalar_t;
  using ElementO = typename Attention::output_t;

  using ElementCompute = typename Attention::accum_t;

  using ElementNorm = typename Attention::accum_t;
  using ElementSum = typename Attention::accum_t;
  using ElementSoftmaxCompute = typename Attention::accum_t;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutP = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using MatrixCoord = typename LayoutP::TensorCoord;

private:

  //
  // Data members
  //

  Options & options;

  /// Initialization
  uint32_t seed = 3080;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device0;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device1;

  std::vector<int64_t> offset_Q;
  std::vector<int64_t> offset_K;
  std::vector<int64_t> offset_P;
  std::vector<int64_t> offset_V;
  std::vector<int64_t> offset_O;

  std::vector<int64_t> ldq_host;
  std::vector<int64_t> ldk_host;
  std::vector<int64_t> ldp_host;
  std::vector<int64_t> ldv_host;
  std::vector<int64_t> ldo_host;
  std::vector<int64_t> seqlen_host;

  cutlass::DeviceAllocation<int64_t> ldq;
  cutlass::DeviceAllocation<int64_t> ldk;
  cutlass::DeviceAllocation<int64_t> ldv;
  cutlass::DeviceAllocation<int64_t> ldo;
  cutlass::DeviceAllocation<int64_t> seqlen;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementO> block_O;

public:

  //
  // Methods
  //

  TestbedAttention(Options &options_): options(options_){ }

  int problem_count() const {
    return (options.head_number * options.batch_size);
  }

private:


  /// Initializes data structures
  void initialize_() {
    // construct a few problems of random sizes
    srand(seed);

    int64_t total_elements_Q = 0;
    int64_t total_elements_K = 0;
    int64_t total_elements_V = 0;
    int64_t total_elements_O = 0;

    ldq_host.resize(problem_count());
    ldk_host.resize(problem_count());
    ldp_host.resize(problem_count());
    ldv_host.resize(problem_count());
    ldo_host.resize(problem_count());
    seqlen_host.resize(problem_count());

    // Create tensors in BMHK format, where
    // B = batch_size
    // M = sequence length
    // H = num_heads
    // K = embedding size per head
    int64_t batch_offset_Q, batch_offset_K, batch_offset_V, batch_offset_O;

    for (int32_t b = 0; b < options.batch_size; ++b) {
      batch_offset_Q = total_elements_Q;
      batch_offset_K = total_elements_K;
      batch_offset_V = total_elements_V;
      batch_offset_O = total_elements_O;
      for (int32_t h = 0; h < options.head_number; ++h) {
        int32_t i = h + b * options.head_number;

        auto problem0 = options.problem_sizes0.at(i);
        auto problem1 = options.problem_sizes1.at(i);

        ldq_host.at(i) = LayoutQ::packed({problem0.m(), options.head_number * problem0.k()}).stride(0);
        ldk_host.at(i) = LayoutK::packed({options.head_number * problem0.k(), problem0.n()}).stride(0);
        ldp_host.at(i) = LayoutP::packed({problem0.m(), problem0.n()}).stride(0);
        ldv_host.at(i) = LayoutV::packed({problem1.k(), options.head_number * problem1.n()}).stride(0);
        ldo_host.at(i) = LayoutO::packed({problem1.m(), options.head_number * problem1.n()}).stride(0);

        // m = n for attention problems.
        seqlen_host.at(i) = problem0.m();

        offset_Q.push_back(batch_offset_Q + h * problem0.k());
        offset_K.push_back(batch_offset_K + h * problem0.k());
        offset_V.push_back(batch_offset_V + h * problem0.k());
        offset_O.push_back(batch_offset_O + h * problem1.n());

        int64_t elements_Q = problem0.m() * problem0.k();
        int64_t elements_K = problem0.k() * problem0.n();
        int64_t elements_V = problem1.k() * problem1.n();
        int64_t elements_O = problem1.m() * problem1.n();

        total_elements_Q += elements_Q;
        total_elements_K += elements_K;
        total_elements_V += elements_V;
        total_elements_O += elements_O;
      }
    }

    problem_sizes_device0.reset(problem_count());
    problem_sizes_device1.reset(problem_count());
    problem_sizes_device0.copy_from_host(options.problem_sizes0.data());
    problem_sizes_device1.copy_from_host(options.problem_sizes1.data());

    ldq.reset(problem_count());
    ldk.reset(problem_count());
    ldv.reset(problem_count());
    ldo.reset(problem_count());
    seqlen.reset(problem_count());

    ldq.copy_from_host(ldq_host.data());
    ldk.copy_from_host(ldk_host.data());
    ldv.copy_from_host(ldv_host.data());
    ldo.copy_from_host(ldo_host.data());
    seqlen.copy_from_host(seqlen_host.data());

    //
    // Assign pointers
    //

    block_Q.reset(total_elements_Q);
    block_K.reset(total_elements_K);
    block_V.reset(total_elements_V);
    block_O.reset(total_elements_O);

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor_(block_Q.get(), total_elements_Q, seed + 1);
    initialize_tensor_(block_K.get(), total_elements_K, seed + 2);
    initialize_tensor_(block_V.get(), total_elements_V, seed + 3);
  }

  template<typename Element>
  bool verify_tensor_(std::vector<Element> vector_Input,
                       std::vector<Element> vector_Input_Ref,
                       int64_t verify_length = -1) {

    int64_t size = (vector_Input.size() < vector_Input_Ref.size()) ? vector_Input.size() : vector_Input_Ref.size();
    size = (verify_length == -1) ? size : verify_length;

    // 0.05 for absolute error
    float abs_tol = 5e-2f;
    // 10% for relative error
    float rel_tol = 1e-1f;
    for (int64_t i = 0; i < size; ++i) {
      float diff = (float)(vector_Input.at(i) - vector_Input_Ref.at(i));
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref.at(i) + 1e-5f);
      float relative_diff = abs_diff / abs_ref;
      if ( (isnan(vector_Input_Ref.at(i)) || isnan(abs_diff) || isinf(abs_diff)) ||  (abs_diff > abs_tol && relative_diff > rel_tol)) {
        printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n", int(i), int(size), abs_diff, relative_diff, (float)(vector_Input.at(i)), (float)(vector_Input_Ref.at(i)));
        return false;
      }

    }

    return true;
  }

  /// Verifies the result is a GEMM
  bool verify_() {

    bool passed = true;

    for (int32_t b = 0; b < options.batch_size; ++b) {
      int32_t i = b * options.head_number;
      // Problem size is the same for all heads
      cutlass::gemm::GemmCoord problem0 = options.problem_sizes0.at(b * options.head_number);
      cutlass::gemm::GemmCoord problem1 = options.problem_sizes1.at(b * options.head_number);

      MatrixCoord extent_Q{problem0.m(), problem0.k()};
      MatrixCoord extent_K{problem0.k(), problem0.n()};
      MatrixCoord extent_P{problem0.m(), problem0.n()};
      MatrixCoord extent_V{problem1.k(), problem1.n()};
      MatrixCoord extent_O{problem1.m(), problem1.n()};

      LayoutO layout_O(ldo_host.at(i));
      std::vector<ElementO> matrix_O(layout_O.capacity(extent_O));
      cutlass::device_memory::copy_to_host(matrix_O.data(),   block_O.get() + offset_O.at(i), matrix_O.size());
      cutlass::DeviceAllocation<ElementO>    block_Ref_O(layout_O.capacity(extent_O));

      for (int32_t h = 0; h < options.head_number; ++h) {
        i = h + b * options.head_number;

        LayoutQ layout_Q(ldq_host.at(i));
        LayoutK layout_K(ldk_host.at(i));
        LayoutP layout_P(ldp_host.at(i));
        LayoutV layout_V(ldv_host.at(i));

        cutlass::TensorView<ElementQ, LayoutQ> view_Q(block_Q.get() + offset_Q.at(i), layout_Q, extent_Q);
        cutlass::TensorView<ElementK, LayoutK> view_K(block_K.get() + offset_K.at(i), layout_K, extent_K);
        cutlass::TensorView<ElementV, LayoutV> view_V(block_V.get() + offset_V.at(i), layout_V, extent_V);
        cutlass::TensorView<ElementO, LayoutO> view_Ref_O_device(block_Ref_O.get() + offset_O.at(i) - offset_O.at(b * options.head_number), layout_O, extent_O);

        cutlass::DeviceAllocation<ElementP>    block_Ref_P(layout_P.capacity(extent_P));
        cutlass::TensorView<ElementP, LayoutP> view_Ref_P_device(block_Ref_P.get(), layout_P, extent_P);

        // Reference GEMM
        cutlass::reference::device::GemmComplex<
            ElementQ, LayoutQ,
            ElementK, LayoutK,
            ElementP, LayoutP, 
            ElementCompute, ElementAccumulator
        >(
          problem0,
          ElementAccumulator(options.alpha0), 
          view_Q,
          Attention::MM0::Mma::kTransformA,
          view_K,
          Attention::MM0::Mma::kTransformB,
          ElementAccumulator(options.beta), 
          view_Ref_P_device, 
          view_Ref_P_device, 
          ElementAccumulator(0)
        );

        // Compute softmax for P. We need to explicitly compute softmax
        // over P because softmax is fused to the second GEMM in the
        // profiled implementation.
        std::vector<ElementP> matrix_Ref(layout_P.capacity(extent_P));
        cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref_P.get(), matrix_Ref.size());
        cutlass::TensorView<ElementP, LayoutP> view_Ref_host(matrix_Ref.data(), layout_P, extent_P);
        std::vector<ElementNorm> vector_Norm_Ref(problem0.m());
        std::vector<ElementSum> vector_Sum_Ref(problem0.m());

        int n_dim = problem0.n();

        // Compute softmax for reference matrix
        for (int m = 0; m < problem0.m(); m++) {
          int n_dim_row = n_dim;
          if (options.causal) {
            n_dim_row = std::min(m + 1, n_dim);
          }
          ElementSoftmaxCompute max = ElementSoftmaxCompute(view_Ref_host.ref().at({m, 0}));
          for (int n = 1; n < n_dim_row; n++) {
            max = std::max(max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
          }

          vector_Norm_Ref.at(m) = ElementNorm(max);

          ElementSoftmaxCompute sum = ElementSoftmaxCompute();
          for (int n = 0; n < n_dim_row; n++) {
            sum += std::exp( ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max );
          }
          ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

          vector_Sum_Ref.at(m) = ElementSum(inv_sum);

          for (int n = 0; n < n_dim_row; n++) {
            view_Ref_host.ref().at({m, n}) = ElementP(
              std::exp( ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max ) * inv_sum
            );
          }
          // Mask out the rest of the attention matrix
          for (int n = n_dim_row; n < n_dim; ++n) {
            view_Ref_host.ref().at({m, n}) = ElementP(0);
          }
        }

        cutlass::device_memory::copy_to_device(block_Ref_P.get(), matrix_Ref.data(), matrix_Ref.size());

        // Reference GEMM
        cutlass::reference::device::GemmComplex<
            ElementP, LayoutP,
            ElementV, LayoutV,
            ElementO, LayoutO, 
            ElementCompute, ElementAccumulator
        >(
          problem1,
          ElementAccumulator(options.alpha1), 
          view_Ref_P_device,
          Attention::MM0::Mma::kTransformA,
          view_V,
          Attention::MM0::Mma::kTransformB,
          ElementAccumulator(options.beta), 
          view_Ref_O_device, 
          view_Ref_O_device, 
          ElementAccumulator(0)
        );
      }

      // Copy to host memory
      std::vector<ElementO> matrix_Ref_O(layout_O.capacity(extent_O));
      cutlass::device_memory::copy_to_host(matrix_Ref_O.data(), block_Ref_O.get(), matrix_Ref_O.size());

      // printf("Pb %d: \n    Q=(offset=%d, ldq=%d)\n    K=(offset=%d, ldk=%d)\n    O=(offset=%d, ldo=%d)\n",
      //   int(i), int(offset_Q[i]), int(ldq_host[i]), int(offset_K[i]), int(ldk_host[i]), int(offset_O[i]), int(ldo_host[i]));
  
      bool verified_O = false;

      if (!verified_O) {
        verified_O = verify_tensor_<ElementO>(matrix_O, matrix_Ref_O);
      }

      passed = passed && verified_O;

      if (!passed) {
        std::cerr << "\n***\nError - problem " << i << " (batch " << b << ") failed the QA check\n***\n" << std::endl;

        if (!verified_O) {
          std::cout << "Final matrix output is incorrect" << std::endl;
        }

        return passed;
      }
    }

    return passed;
  }

public:


  /// Executes a CUTLASS Attention kernel and measures runtime.
  Result profile() {

    Result result;
    result.passed = false;

    // Initialize the problem
    initialize_();

    typename Attention::Params p;
    { // set parameters
      p.query_ptr = block_Q.get();
      p.key_ptr = block_K.get();
      p.value_ptr = block_V.get();
      p.output_ptr = block_O.get();

      p.scale = options.alpha0;

      p.num_heads = options.head_number;
      p.num_batches = options.batch_size;
      p.head_dim = options.head_size;
      p.head_dim_value = options.head_size_v;
      p.num_queries = options.seq_length;
      p.num_keys = options.seq_length_kv;

      // All tensors are in BMHK shapes
      p.q_strideH = options.head_size;
      p.k_strideH = options.head_size;
      p.v_strideH = options.head_size_v;
      p.q_strideM = int32_t(ldq_host[0]);
      p.k_strideM = int32_t(ldk_host[0]);
      p.v_strideM = int32_t(ldv_host[0]);
      p.q_strideB = p.q_strideM * options.seq_length;
      p.k_strideB = p.k_strideM * options.seq_length_kv;
      p.v_strideB = p.v_strideM * options.seq_length_kv;
      p.o_strideM = p.head_dim_value * p.num_heads;
    }

    // launch kernel :)
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return result;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_();
    }

    //
    // Warm-up run
    //

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Attention kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << std::endl;
    std::cout << "CUTLASS Attention:\n"
      << "====================================================" << std::endl;
    std::cout << "    " << " {seq length Q, seq length KV, head size, head size V, head number, batch size} = {" << options.seq_length \
      << ", " << options.seq_length_kv << ", " << options.head_size << ", " << options.head_size_v << ", " << options.head_number\
      << ", " << options.batch_size << "}." << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

template <
  int kQueriesPerBlock,
  int kKeysPerBlock
>
int run_attention(Options& options) {
  using Attention = AttentionKernel<
    cutlass::half_t,      // scalar_t
    cutlass::arch::Sm80,  // ArchTag
    kQueriesPerBlock,
    kKeysPerBlock
  >;

  //
  // Test and profile
  //

  TestbedAttention<Attention> testbed(options);

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "Profiling CUTLASS attention has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
  }

  std::cout << "\nPassed\n";
  return 0;
}

int main() {
  Options options;
  
  options.randomize_problems();

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  static int const kQueriesPerBlock = 64;
  static int const kKeysPerBlock = 64;
  return run_attention<kQueriesPerBlock, kKeysPerBlock>(options);
}
