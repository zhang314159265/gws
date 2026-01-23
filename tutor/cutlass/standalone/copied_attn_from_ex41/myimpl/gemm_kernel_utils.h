// Fallback to Simt (FMA on cuda cores) if not in a special case below
template <typename ArchTag, typename scalar_t_, typename Enable = void>
struct DefaultGemmType {
  static constexpr int ThreadK = 8;
  static constexpr int WarpK = 8;
  static constexpr int kMinimumAlignment = 1;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OpClass = cutlass::arch::OpClassSimt;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f32
template <typename ArchTag>
struct DefaultGemmType<
    ArchTag,
    float,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 80>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
};

// Specialization for tensorcores with f16/bf16 - Sm75+
template <typename ArchTag, typename scalar_t>
struct DefaultGemmType<
    ArchTag,
    scalar_t,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 75 &&
        cutlass::sizeof_bits<scalar_t>::value == 16>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f16 - Volta
template <>
struct DefaultGemmType<cutlass::arch::Sm70, cutlass::half_t, void> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 2;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Enables to do
// `auto x = kCondition ? fa(arg) : fb(arg)`
// when `fa` and `fb` have different types
template <bool kVal, typename TA, typename TB>
struct call_conditional;

template <typename TA, typename TB>
struct call_conditional<true, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(ta(arg)) {
    return ta(arg);
  }
};

template <typename TA, typename TB>
struct call_conditional<false, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(tb(arg)) {
    return tb(arg);
  }
};

#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) { \
    return false; \
  }

#include <iostream>

#define XFORMERS_CHECK(COND, ERR) \
  if (!(COND)) { \
    std::cerr << "'" #COND "' failed: " << ERR << "\n"; \
    return false; \
  }
