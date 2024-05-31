#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h.inc"
#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.cpp.inc"

void mlir::_tritoncc::TritonNvidiaGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tritoncc/dialect/TritonNvidiaGPU/AttrDefs.cpp.inc"
  >();
}

namespace tritoncc {
}
