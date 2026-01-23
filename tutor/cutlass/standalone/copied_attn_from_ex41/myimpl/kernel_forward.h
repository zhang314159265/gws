#pragma once

#include <curand_kernel.h>
#include <cmath>
#include <cinttypes>
#include <vector>

#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "../debug_utils.h"
#include "../epilogue/epilogue_pipelined.h"
#include "../epilogue/epilogue_rescale_output.h"
#include "../gemm/mma_from_smem.h"
#include "../gemm_kernel_utils.h"
#include "../transform/tile_smem_loader.h"

using namespace gemm_kernel_utils;


static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    int kQueriesPerBlock_,
    int kKeysPerBlock_
>
struct AttentionKernel {
  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = scalar_t;
  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  using output_accum_t = accum_t;

  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr int kQueriesPerBlock = kQueriesPerBlock_;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int kWarpSize = 32;

  // Launch bounds
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [num_queries, num_heads, head_dim]
    scalar_t* key_ptr = nullptr; // [num_keys, num_heads, head_dim]
    scalar_t* value_ptr = nullptr; // [num_keys, num_heads, head_dim_value]
    output_t* output_ptr = nullptr; // [num_queries, num_heads, head_dim_value]

    // Scale
    accum_t scale = 0.0;

    // Dimensions/strides
    int32_t num_batches = 0;
    int32_t num_heads = 0;
    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    int32_t num_queries = 0;
    int32_t num_keys = 0;

    int32_t q_strideM = 0;
    int32_t k_strideM = 0;
    int32_t v_strideM = 0;

    int32_t o_strideM = 0;

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int32_t q_strideH = 0;
    int32_t k_strideH = 0;
    int32_t v_strideH = 0;

    int64_t q_strideB = 0;
    int64_t k_strideB = 0;
    int64_t v_strideB = 0;

    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto query_start = blockIdx.x * kQueriesPerBlock;

      query_ptr += batch_id * q_strideB + head_id * q_strideH + query_start * q_strideM;
      key_ptr += batch_id * k_strideB + head_id * k_strideH;
      value_ptr += batch_id * k_strideB + head_id * v_strideH;
      output_ptr += batch_id * num_queries * o_strideM + head_id * head_dim_value + query_start * o_strideM;

      num_queries -= query_start;
      num_batches = 0; // no longer used after

      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          ceil_div(num_queries, (int32_t)kQueriesPerBlock),
          num_heads,
          num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA;
    static constexpr int kAlignmentB = DefaultConfig::kAlignmentB;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;

    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        4, // num stages
        typename GemmType::Operator // Operator
        >;

    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using DefaultThreadblockMma = typename DefaultMma::ThreadblockMma;
    using Mma = DefaultThreadblockMma;
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA; // from smem
    static constexpr int kAlignmentB = DefaultConfig::kAlignmentB;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        4, // stage
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Policy::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
            typename DefaultGemm::Mma::Policy::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MM0::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    cutlass::Array<accum_t, kQueriesPerBlock> out_rescale;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorage : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      typename MM0::AccumulatorSharedStorage si;
      typename MM1::Mma::SharedStorage mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    XFORMERS_CHECK(
        p.q_strideM % kAlignmentQ == 0,
        "query is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.k_strideM % kAlignmentK == 0,
        "key is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.v_strideM % kAlignmentV == 0,
        "value is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.q_strideH % kAlignmentQ == 0,
        "query is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.k_strideH % kAlignmentK == 0,
        "key is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.v_strideH % kAlignmentV == 0,
        "value is not correctly aligned (strideH)");
    return true;
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      out_rescale[thread_id()] = accum_t(1.0);
    }
    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto createOutputIter = [&]() -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries, p.head_dim_value},
          thread_id(),
          {0, 0});
    };

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys; iter_key_start += kKeysPerBlock) {
      int32_t problem_size_0_m =
          cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
      int32_t problem_size_0_n = cutlass::fast_min(
          int32_t(kKeysPerBlock), p.num_keys - iter_key_start);
      int32_t const& problem_size_0_k = p.head_dim;
      int32_t const& problem_size_1_n = p.head_dim_value;
      int32_t const& problem_size_1_k = problem_size_0_n;

      auto prologueV = [&]() {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{typename MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, 0});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1,
            iterator_V,
            thread_id(),
            problem_size_1_k);
      };

      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[query_start:query_end, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //

      // Construct iterators to A and B operands
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(p.q_strideM)),
          p.query_ptr,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          {0, 0});

      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(
              typename MM0::MmaCore::LayoutB(p.k_strideM)),
          p.key_ptr + iter_key_start * p.k_strideM,
          {problem_size_0_k, problem_size_0_n},
          thread_id(),
          {0, 0});

      auto my_warp_id = warp_id();
      auto my_lane_id = lane_id();

      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();

      prologueV();

      typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {
              (my_warp_id % MM0::Mma::WarpCount::kM),
              (my_warp_id / MM0::Mma::WarpCount::kM)};

      iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
          accum_o,
          accum,
          mi,
          m_prime,
          s_prime,
          out_rescale,
          shared_storage.addition_storage,
          my_lane_id,
          thread_id(),
          my_warp_id,
          p.num_keys - iter_key_start,
          iter_key_start == 0,
          iteratorC_tile_offset,
          p.scale);

      // Output results to shared-memory
      int warp_idx_mn_0 = my_warp_id %
          (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

      __syncthreads();


      //
      // MATMUL: Attn . V
      // Run the matmul `attn @ V` for a block of attn and V.
      // `attn` is read from shared memory (in `shared_storage_si`)
      // `V` is read from global memory (with iterator_B)
      //

      {
        int gemm_k_iterations =
            (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{typename MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, 0});
        typename MM1::Mma mma_pv(
            // operand A: Pij_dropped in shared memory
            shared_storage.after_mm0.si.accum_ref(),
            // operand B: shared memory staging area for Vj, which is loaded
            // from global memory
            shared_storage.after_mm0.mm1.operand_B_ref(),
            (int)thread_id(),
            (int)my_warp_id,
            (int)my_lane_id);
        mma_pv.set_prologue_done(true);
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
        __syncthreads();
      }
    }

    {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      using ElementCompute = typename DefaultOp::ElementCompute;
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // output
              output_accum_t, // source
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // accum
              output_accum_t, // compute
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // destination
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter();
      EpilogueOutputOp rescale(s_prime, out_rescale);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      MM1::Mma::drain_cp_asyncs();
      epilogue(rescale, dest_iter, accum_o);
    }
  }

  template <typename WarpIteratorC>
  CUTLASS_DEVICE static void iterative_softmax(
      typename WarpIteratorC::Fragment& frag_o, // output so far
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          addition_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool is_first,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling) {
    /* Iterates on the accumulator and corresponding position on result matrix

    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)

    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Make sure we all share the update values for `mi`
    __syncthreads();

    // Doing this `exp` is quite expensive. Let's
    // split it across the warps
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      auto m_prime_id = m_prime[id];
      auto mi_id = mi[id];
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      if (changed) {
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        out_rescale[id] = m_prime_exp;
        s_prime[id] *= m_prime_exp;
      } else {
        // Only when bias is enabled, it's possible that all the first values
        // of attention are masked to `-inf`. In that case we want to avoid
        // `nan = exp2f(-inf - (-inf))` so we temporarily set `mi` to 0
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // Update output fragments
    if (!is_first) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // NOTE: we could atomically add `total_row` to `s_prime`, but
              // it's faster (and deterministic) to avoid atomics here
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
    }
    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      accum_t total_row = s_prime[id];
      if (restore_mi_to_minus_inf) {
        // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        m_prime[id] = mi[id];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      s_prime[id] = total_row;
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }
  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }
  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void attention_kernel_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

