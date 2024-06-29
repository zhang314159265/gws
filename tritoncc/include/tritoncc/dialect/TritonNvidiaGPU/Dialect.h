#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h.inc"

namespace tritoncc {
}
