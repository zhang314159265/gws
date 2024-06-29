#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#endif

#include "tritoncc/dialect/NVGPU/Dialect.h.inc"

#define GET_OP_CLASSES
#include "tritoncc/dialect/NVGPU/Ops.h.inc"

namespace tritoncc {
}
