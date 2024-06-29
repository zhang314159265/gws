#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h"
#include "tritoncc/dialect/Triton/Dialect.h"
#include "tritoncc/dialect/TritonGPU/Dialect.h"

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.cpp.inc"

void mlir::_tritoncc::TritonNvidiaGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tritoncc/dialect/TritonNvidiaGPU/AttrDefs.cpp.inc"
  >();
}
