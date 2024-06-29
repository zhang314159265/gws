#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "tritoncc/dialect/NVGPU/Dialect.h"

#include "tritoncc/dialect/NVGPU/Dialect.cpp.inc"

void mlir::_tritoncc::nvgpu::NVGPUDialect::initialize() {
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
