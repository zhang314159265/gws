#pragma once

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "shape-inference"

#include "toy/Dialect.h"

namespace mlir { namespace toy {
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"
} }

namespace toy {

struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, mlir::OperationPass<mlir::toy::FuncOp>> {

  void runOnOperation() override {
    auto f = getOperation();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        opWorklist.insert(op);
      }
    });

    while (!opWorklist.empty()) {
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end()) {
        break;
      }

      mlir::Operation *op = *nextop;
      opWorklist.erase(op);

      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = llvm::dyn_cast<mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
            "inference interface: ") << *op;
        return signalPassFailure();
      }
    }

    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  static bool allOperandsInferred(mlir::Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](mlir::Type operandType) {
      return llvm::isa<mlir::RankedTensorType>(operandType);
    });
  }

  static bool returnsDynamicShape(mlir::Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](mlir::Type resultType) {
      return !llvm::isa<mlir::RankedTensorType>(resultType);
    });
  }
};

std::unique_ptr<mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

}

#undef DEBUG_TYPE
