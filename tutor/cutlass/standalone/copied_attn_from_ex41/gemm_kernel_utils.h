#pragma once

#include "cutlass/arch/mma.h"

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////
#define DISPATCH_TYPES(tensor, func)                                           \
  {                                                                            \
    if (query.scalar_type() == at::ScalarType::Float) {                        \
      using scalar_t = float;                                                  \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::Half) {                  \
      using scalar_t = cutlass::half_t;                                        \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {              \
      using scalar_t = cutlass::bfloat16_t;                                    \
      func();                                                                  \
    } else {                                                                   \
      XFORMERS_CHECK(false, "Only fp32, half & bf16 supported at the moment"); \
    }                                                                          \
  }

#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      using BOOL_NAME = std::true_type;      \
      F();                                  \
    } else {                                \
      using BOOL_NAME = std::false_type;      \
      F();                                  \
    }                                       \
  }

#define DISPATCH_ARCHTAG(CC, func)                                        \
  {                                                                       \
    if (CC >= 80) {                                                       \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (CC >= 75) {                                                \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (CC >= 70) {                                                \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (CC >= 50) {                                                \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      XFORMERS_CHECK(                                                     \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#define ASSIGN_CHECK_OVERFLOW(A, B)                                    \
  {                                                                    \
    A = B;                                                             \
    XFORMERS_CHECK(                                                    \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

namespace gemm_kernel_utils {

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer align_up(integer n, integer m) {
  return ((n + m - 1) / m) * m;
}

#include "myimpl/gemm_kernel_utils.h"

} // namespace gemm_kernel_utils
