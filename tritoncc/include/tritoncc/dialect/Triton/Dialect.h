#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif

#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "tritoncc/dialect/Triton/Dialect.h.inc"
#include "tritoncc/dialect/Triton/OpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tritoncc/dialect/Triton/Types.h.inc"

#include "tritoncc/dialect/Triton/Traits.h"

#define GET_OP_CLASSES
#include "tritoncc/dialect/Triton/Ops.h.inc"

namespace tritoncc {

class DialectInferLayoutInterface
    : public mlir::DialectInterface::Base<DialectInferLayoutInterface> {
 public:
  DialectInferLayoutInterface(mlir::Dialect *dialect) : Base(dialect) {}

  virtual mlir::LogicalResult
  inferTransOpEncoding(mlir::Attribute operandEncoding,
      llvm::ArrayRef<int32_t> order,
      mlir::Attribute &resultEncoding) const = 0;
  virtual mlir::LogicalResult
  inferReduceOpEncoding(mlir::Attribute operandEncoding,
      unsigned axis,
      mlir::Attribute &resultEncoding) const = 0;
  virtual mlir::LogicalResult
  inferExpandDimsOpEncoding(mlir::Attribute operandEncoding,
      unsigned axis,
      mlir::Attribute &resultEncoding,
      std::optional<mlir::Location> location) const = 0;
  virtual mlir::LogicalResult
  inferDotOpEncoding(mlir::Attribute operandEncoding,
      unsigned opIdx,
      mlir::Attribute retEncoding,
      std::optional<mlir::Location> location) const = 0;
  virtual mlir::LogicalResult
  inferReshapeOpNoReorderEncoding(llvm::ArrayRef<int64_t> srcShape,
      mlir::Attribute srcEnc,
      llvm::ArrayRef<int64_t> dstShape,
      mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const = 0;
  virtual mlir::LogicalResult
  inferJoinOpEncoding(mlir::Attribute srcEnc, mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const = 0;
  virtual mlir::LogicalResult
  inferSplitOpEncoding(mlir::Attribute srcEnc, mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const = 0;
  virtual mlir::LogicalResult
  verifyDotOpEncodingCompatibility(mlir::Operation *op,
      mlir::Attribute operandEncodingA,
      mlir::Attribute operandEncodingB) const = 0;
};

}

struct TritonInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
      bool wouldBeCloned) const final {
    auto funcOp = llvm::dyn_cast<mlir::_tritoncc::FuncOp>(callable);
    if (!funcOp) {
      return true;
    }
    if (funcOp->hasAttr("noinline")) {
      return !funcOp->getAttrOfType<mlir::BoolAttr>("noinline").getValue();
    }
    return true;
  }

  bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool wouldBeCloned,
      mlir::IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool wouldBeCloned,
      mlir::IRMapping &) const final {
    return true;
  }

  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    assert(false && "handleTerminator");
  }

  void handleTerminator(mlir::Operation *op, mlir::ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = llvm::cast<mlir::_tritoncc::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

namespace tritoncc {
static mlir::Type getLoadOpResultType(mlir::OpBuilder &builder, mlir::Type ptrType) {
  auto ptrTensorType = ptrType.dyn_cast<mlir::RankedTensorType>();
  if (!ptrTensorType) {
    return ptrType.cast<mlir::_tritoncc::PointerType>().getPointeeType();
  }
  auto shape = ptrTensorType.getShape();
  mlir::Type elementType =
    ptrTensorType.getElementType().cast<mlir::_tritoncc::PointerType>().getPointeeType();
  return mlir::RankedTensorType::get(shape, elementType);
}
}
