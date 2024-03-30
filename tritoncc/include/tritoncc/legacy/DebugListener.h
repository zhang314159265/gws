#pragma once

#include "mlir/IR/Builders.h"

namespace tritoncc {

#if 1 // does not work since call ConversionPatternRewriter's setListener method result in error: ./include/tritoncc/ReduceOpConversion.h:175:25: error: ‘void mlir::OpBuilder::setListener(mlir::OpBuilder::Listener*)’ is inaccessible within this context
class DebugListener : public mlir::OpBuilder::Listener {
 public:
  void notifyOperationInserted(Operation *op, mlir::OpBuilder::InsertPoint previous) {
    std::cerr << "Added operation: "; op->dump();    
  }
};
#endif

};
