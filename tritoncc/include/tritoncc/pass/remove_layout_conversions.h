#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 1

#if USE_TRITON
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#endif

namespace tritoncc {

// return true if the op is an op with a layout we don't want to change. We
// will propagate the layout starting from anchor ops.
bool isLayoutAnchor(mlir::Operation *op) {
  if (llvm::isa<mlir::triton::LoadOp, mlir::triton::StoreOp>(op)) {
    
  }
  assert(false && "isLayoutAnchor");
}

class LayoutPropagation {
 public:
  explicit LayoutPropagation(mlir::triton::FuncOp F) : funcOp(F) { }

  void initAnchorLayout() {
    funcOp.walk([&](mlir::Operation *op) {
      if (isLayoutAnchor(op)) {
      }
    });
    assert(false && "initAnchorLayout");
  }

  void propagateLayout() {
    assert(false && "propagateLayout");
  }

  void resolveConflicts() {
    assert(false && "resolveConflicts");
  }

  void rewrite() {
    assert(false && "rewrite");
  }

 private:
  mlir::triton::FuncOp funcOp;
};

class RemoveLayoutConversionsPass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit RemoveLayoutConversionsPass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<RemoveLayoutConversionsPass>()) { }

  llvm::StringRef getName() const override {
    return "RemoveLayoutConversionsPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // 1
    moduleOp.walk([](mlir::triton::FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    // moduleOp.dump();
    assert(false && "hlt");
  }
};

#if USE_TRITON
static std::unique_ptr<mlir::Pass> createRemoveLayoutConversionsPass() {
  return mlir::triton::gpu::createRemoveLayoutConversionsPass();
}
#else
static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createRemoveLayoutConversionsPass() {
  return std::make_unique<RemoveLayoutConversionsPass>();
}
#endif
}
