#pragma once

#include <assert.h>
#include "llvm/ADT/MapVector.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "tritoncc/pass/util.h"

namespace tritoncc {

static mlir::Value getMemAccessPtr(mlir::Operation* op) {
  if (mlir::triton::LoadOp ld = llvm::dyn_cast<mlir::triton::LoadOp>(op)) {
    return ld.getPtr();
  }
  if (mlir::triton::StoreOp st = llvm::dyn_cast<mlir::triton::StoreOp>(op)) {
    return st.getPtr();
  }
  return nullptr;
}

class CoalescePass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit CoalescePass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<CoalescePass>()) { }

  llvm::StringRef getName() const override {
    return "CoalescePass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void setCoalescedEncoding(mlir::triton::ModuleAxisInfoAnalysis &axisInfoAnalysis, mlir::Operation *op, int numWarps, int threadsPerWarp, llvm::MapVector<mlir::Operation*, mlir::Attribute> &layoutMap) {
    mlir::Value ptr = tritoncc::getMemAccessPtr(op);
    mlir::RankedTensorType refTensorType = ptr.getType().cast<mlir::RankedTensorType>();
    assert(refTensorType);

    llvm::SmallVector<int64_t> contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();

    #if 0
    // output for sum
    // Contiguity is: 1 1024
    // Contiguity is: 2
    std::cerr << "Contiguity is: ";
    for (int64_t c : contiguity) {
      std::cerr << c << " ";
    }
    std::cerr << std::endl;
    #endif

    llvm::SmallVector<unsigned> order = argSortDesc(contiguity);

    assert(ptr.getDefiningOp());

    auto matchesShape = [&refTensorType](const mlir::Value& val) {
      mlir::RankedTensorType otherType = val.getType().dyn_cast<mlir::RankedTensorType>();
      return otherType && otherType.getShape() == refTensorType.getShape();
    };

    // multiRootGetSlice is defined in triton rather than llvm/mlir
    // it collects transitive uses/defs of op
    llvm::SmallSetVector<mlir::Operation*, 32> memAccessesSameOrder;
    llvm::errs() << "Find memAccessesSameOrder for " << *op << "\n";
    for (mlir::Operation *use : mlir::multiRootGetSlice(op)) {
      #if 0
      llvm::errs() << "OpUse: " << *use << '\n';
      #endif
      if (use == op) {
        continue;
      }
      mlir::Value val = tritoncc::getMemAccessPtr(use);
      if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use)) {
        continue;
      }
      llvm::SmallVector<unsigned> currOrder =
        argSortDesc(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      if (order == currOrder) {
        llvm::errs() << "Insert to memAccessSameOrder: " << *use << '\n';
        memAccessesSameOrder.insert(use);
      }
    }

    llvm::SmallVector<int64_t> shapePerCTA = mlir::triton::gpu::getShapePerCTA(refTensorType);
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);
    unsigned perThread = mlir::getNumElementsPerThread(op, order, axisInfoAnalysis);
    for (mlir::Operation *opSameOrder : memAccessesSameOrder) {
      unsigned currPerThread = mlir::getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      perThread = std::max(perThread, currPerThread);
    }
    perThread = std::min<int>(perThread, numElemsPerThread);

    if (!llvm::dyn_cast<mlir::triton::LoadOp>(op)) {
      perThread = std::min<int>(
        perThread, getNumElementsPerThread(op, order, axisInfoAnalysis));
    }
    llvm::SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;
    mlir::triton::gpu::CTALayoutAttr CTALayout = mlir::triton::gpu::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = mlir::triton::gpu::BlockedEncodingAttr::get(
      &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
      threadsPerWarp, CTALayout);
  }

  static mlir::Type getNewType(mlir::Type type, mlir::Attribute encoding) {
    mlir::RankedTensorType tensorType = type.cast<mlir::RankedTensorType>();
    return mlir::RankedTensorType::get(
      tensorType.getShape(),
      tensorType.getElementType(),
      encoding);
  }

  void coalesceOp(mlir::Operation *op, mlir::Attribute encoding) {
    mlir::OpBuilder builder(op);
    llvm::SmallVector<mlir::Value, 4> newArgs;
    for (mlir::Value operand : op->getOperands()) {
      mlir::RankedTensorType tensorType = operand.getType().dyn_cast<mlir::RankedTensorType>();
      if (tensorType && !tensorType.getEncoding().isa<mlir::triton::gpu::SharedEncodingAttr>()) {
        mlir::Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<mlir::triton::gpu::ConvertLayoutOp>(op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // convert output types
    llvm::SmallVector<mlir::Type, 4> newTypes;
    for (mlir::Type t : op->getResultTypes()) {
      bool isAsync = llvm::isa<mlir::triton::gpu::InsertSliceAsyncOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    mlir::Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs, newTypes, op->getAttrs());

    for (size_t i = 0; i < op->getNumResults(); ++i) {
      mlir::Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
    llvm::MapVector<mlir::Operation*, mlir::Attribute> layoutMap;
    moduleOp.walk([&](mlir::Operation* curr) {
      mlir::Value ptr = tritoncc::getMemAccessPtr(curr);
      if (!ptr) {
        return;
      }
      if (mlir::triton::PointerType ptrType = ptr.getType().dyn_cast<mlir::triton::PointerType>()) {
        assert(false && "Pointer to tensor not supported yet. Please use tensor of pointers");
      }
      mlir::RankedTensorType tensorType = ptr.getType().dyn_cast<mlir::RankedTensorType>();
      assert(tensorType);

      mlir::ModuleOp mod = curr->getParentOfType<mlir::ModuleOp>();
      int numWarps = mlir::triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp = mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp, layoutMap);
    });
    for (auto &kv : layoutMap) {
      coalesceOp(kv.first, kv.second);
    }
  }
 private:
};

static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCoalescePass() {
  return std::make_unique<CoalescePass>();
}

}
