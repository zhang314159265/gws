#pragma once

namespace tritoncc {

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
    moduleOp.dump();
    assert(false && "hlt");
  }
};

static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createRemoveLayoutConversionsPass() {
  return std::make_unique<RemoveLayoutConversionsPass>();
}

}
