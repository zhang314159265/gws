#pragma once

#include "mlir/IR/OpDefinition.h"

namespace tritoncc {

bool isTensorPointerType(mlir::Type type) {
  if (mlir::_tritoncc::PointerType ptrType = type.dyn_cast<mlir::_tritoncc::PointerType>()) {
    return ptrType.getPointeeType().isa<mlir::RankedTensorType>();
  }
  return false;
}

namespace impl {

using namespace mlir::OpTrait::impl;

static mlir::LogicalResult verifySameEncoding(mlir::Type typeA, mlir::Type typeB,
    bool allowTensorPointerType) {
  auto getEncoding = [=](mlir::Type type) -> mlir::Attribute {
    mlir::Attribute ret;
    if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
      ret = tensorType.getEncoding();
    }
    if (!allowTensorPointerType) {
      assert(!tritoncc::isTensorPointerType(type));
    }
    return ret;
  };
  auto encodingA = getEncoding(typeA);
  auto encodingB = getEncoding(typeB);
  if (!encodingA || !encodingB) {
    return mlir::success();
  }
  return encodingA == encodingB ? mlir::success() : mlir::failure();
}

mlir::LogicalResult
verifySameOperandsEncoding(mlir::Operation *op,
    bool allowTensorPointerType) {
  if (mlir::failed(verifyAtLeastNOperands(op, 1))) {
    return mlir::failure();
  }

  auto type = op->getOperand(0).getType();
  for (auto opType : llvm::drop_begin(op->getOperandTypes(), 1)) {
    if (mlir::failed(verifySameEncoding(opType, type, allowTensorPointerType))) {
      return op->emitOpError() << "requires the same encoding for all operands";
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifySameOperandsAndResultEncoding(
    mlir::Operation *op, bool allowTensorPointerType = false) {
  if (op->getNumOperands() == 0) {
    return mlir::success();
  }

  if (mlir::failed(verifyAtLeastNOperands(op, 1)) ||
      mlir::failed(verifyAtLeastNResults(op, 1))) {
    return mlir::failure();
  }

  auto type = op->getOperand(0).getType();
  for (auto resultType : op->getResultTypes()) {
    if (mlir::failed(verifySameEncoding(resultType, type, allowTensorPointerType))) {
      return op->emitOpError()
        << "requires the same encoding for all operands and results";
    }
  }
  return verifySameOperandsEncoding(op, allowTensorPointerType);
}

}

template <typename ConcreteType>
class SameOperandsAndResultEncoding
    : public mlir::OpTrait::TraitBase<ConcreteType, SameOperandsAndResultEncoding> {
 public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultEncoding
    : public mlir::OpTrait::TraitBase<ConcreteType, SameLoadStoreOperandsAndResultEncoding> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    assert(false && "verifyTrait");
  }
};

}

namespace mlir { namespace OpTrait {
using tritoncc::SameOperandsAndResultEncoding;
} }
