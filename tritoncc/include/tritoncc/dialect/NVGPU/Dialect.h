#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#endif

#include "tritoncc/dialect/NVGPU/Dialect.h.inc"
#include "tritoncc/dialect/NVGPU/Dialect.cpp.inc"

#define GET_OP_CLASSES
#include "tritoncc/dialect/NVGPU/Ops.h.inc"

void mlir::_tritoncc::NVGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tritoncc/dialect/NVGPU/AttrDefs.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "tritoncc/dialect/NVGPU/Ops.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "tritoncc/dialect/NVGPU/Ops.cpp.inc"

namespace tritoncc {
}
