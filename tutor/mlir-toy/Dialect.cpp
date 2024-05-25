#include "toy/Dialect.h"
#include "toy/Pattern.h"

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"

#include "toy/Dialect.cpp.inc"

using namespace mlir::toy;
using namespace toy;

struct ToyInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
      bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(mlir::Region *, mlir::Region *, bool, mlir::IRMapping &) const final {
    return true;
  }

  void handleTerminator(mlir::Operation *op, mlir::ValueRange valuesToRepl) const final {
    auto returnOp = llvm::cast<mlir::toy::ReturnOp>(op);

    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }

  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input,
      mlir::Type resultType,
      mlir::Location conversionLoc) const final {
    return builder.create<mlir::toy::CastOp>(conversionLoc, resultType, input);
  }
};

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();

  addInterfaces<ToyInlinerInterface>();
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

void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
    mlir::MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<mlir::RankedTensorType>(getOperand().getType());
  llvm::SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(mlir::RankedTensorType::get(dims, arrayTy.getElementType()));
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

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

// GenericCallOp
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    llvm::StringRef callee, llvm::ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
      mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

mlir::CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
  assert(false);
}

mlir::Operation::operand_range GenericCallOp::getArgOperands() {
  return getInputs();
}

mlir::MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

// ReshapeOp
void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
    mlir::MLIRContext *context) {
  results.add<
      ReshapeReshapeOptPattern,
      FoldConstantReshapeOptPattern,
      RedundantReshapeOptPattern
  >(context);
}

// CastOp
void CastOp::inferShapes() {
  getResult().setType(getInput().getType());
}

bool CastOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  mlir::TensorType input = llvm::dyn_cast<mlir::TensorType>(inputs.front());
  mlir::TensorType output = llvm::dyn_cast<mlir::TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType()) {
    return false;
  }
  return !input.hasRank() || !output.hasRank() || input == output;
}

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
