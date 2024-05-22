#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"

#include "toy/Dialect.cpp.inc"

using namespace mlir::toy;

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
}

// FuncOp
void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    llvm::StringRef name, mlir::FunctionType type,
    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

// TransponseOp
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

// AddOp
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

// MulOp
void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

// GenericCallOp
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    llvm::StringRef callee, llvm::ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
      mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
