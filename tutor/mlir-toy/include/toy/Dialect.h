#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace toy {
#include "toy/ShapeInferenceOpInterfaces.h.inc"
}
}

#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

namespace mlir {
namespace toy {
namespace detail {

// This class represents the interal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  llvm::ArrayRef<mlir::Type> elementTypes;
};

}

class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
    detail::StructTypeStorage> {
 public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after the context are forwarded to the storage instance
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  size_t getNumElementTypes() { return getElementTypes().size(); }

  static constexpr llvm::StringLiteral name = "toy.struct";
};

} }
