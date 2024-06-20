#pragma once

#include <assert.h>
#include "llvm/ADT/MapVector.h"
#include "mlir/Pass/Pass.h"

#include "tritoncc/AxisInfo.h"
#include "tritoncc/dialect/TritonGPU/Dialect.h"

#include "tritoncc/util.h"
#include "tritoncc/nvidia_util.h"
#include "tritoncc/layout_util.h"

#ifdef DEBUG
#undef DEBUG
#endif
#define DEBUG 0

namespace tritoncc {

class CoalescePass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit CoalescePass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<CoalescePass>()) { }

  llvm::StringRef getName() const override {
    return "CoalescePass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void setCoalescedEncoding(tritoncc::ModuleAxisInfoAnalysis &axisInfoAnalysis, mlir::Operation *op, int numWarps, int threadsPerWarp, llvm::MapVector<mlir::Operation*, mlir::Attribute> &layoutMap) {
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
    #if DEBUG
    llvm::errs() << "Find memAccessesSameOrder for " << *op << "\n";
    #endif
    for (mlir::Operation *use : tritoncc::multiRootGetSlice(op)) {
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
        #if DEBUG
        llvm::errs() << "Insert to memAccessSameOrder: " << *use << '\n';
        #endif
        memAccessesSameOrder.insert(use);
      }
    }

    llvm::SmallVector<int64_t> shapePerCTA = tritoncc::getShapePerCTA(refTensorType);
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);
    unsigned perThread = tritoncc::getNumElementsPerThread(op, order, axisInfoAnalysis);
    for (mlir::Operation *opSameOrder : memAccessesSameOrder) {
      unsigned currPerThread = tritoncc::getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      perThread = std::max(perThread, currPerThread);
    }
    perThread = std::min<int>(perThread, numElemsPerThread);

    if (!llvm::dyn_cast<mlir::_tritoncc::LoadOp>(op)) {
      perThread = std::min<int>(
        perThread, tritoncc::getNumElementsPerThread(op, order, axisInfoAnalysis));
    }
    llvm::SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;
    mlir::_tritoncc::gpu::CTALayoutAttr CTALayout = tritoncc::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = mlir::_tritoncc::gpu::BlockedEncodingAttr::get(
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
      if (tensorType && !tensorType.getEncoding().isa<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
        mlir::Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<mlir::_tritoncc::gpu::ConvertLayoutOp>(op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // convert output types
    llvm::SmallVector<mlir::Type, 4> newTypes;
    for (mlir::Type t : op->getResultTypes()) {
      bool isAsync = llvm::isa<mlir::_tritoncc::gpu::InsertSliceAsyncOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    mlir::Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs, newTypes, op->getAttrs());

    for (size_t i = 0; i < op->getNumResults(); ++i) {
      mlir::Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<mlir::_tritoncc::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    tritoncc::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
    llvm::MapVector<mlir::Operation*, mlir::Attribute> layoutMap;
    moduleOp.walk([&](mlir::Operation* curr) {
      mlir::Value ptr = tritoncc::getMemAccessPtr(curr);
      if (!ptr) {
        return;
      }
      if (mlir::_tritoncc::PointerType ptrType = ptr.getType().dyn_cast<mlir::_tritoncc::PointerType>()) {
        assert(false && "Pointer to tensor not supported yet. Please use tensor of pointers");
      }
      mlir::RankedTensorType tensorType = ptr.getType().dyn_cast<mlir::RankedTensorType>();
      assert(tensorType);

      mlir::ModuleOp mod = curr->getParentOfType<mlir::ModuleOp>();
      int numWarps = mlir::_tritoncc::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp = mlir::_tritoncc::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
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
