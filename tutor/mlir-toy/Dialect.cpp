#include "toy/Dialect.h"
#include "toy/Pattern.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
  addTypes<StructType>();
}

mlir::Operation *ToyDialect::materializeConstant(
    mlir::OpBuilder &builder,
    mlir::Attribute value,
    mlir::Type type,
    mlir::Location loc) {
  if (llvm::isa<mlir::toy::StructType>(type)) {
    return builder.create<mlir::toy::StructConstantOp>(loc, type,
        llvm::cast<mlir::ArrayAttr>(value));
  }
  return builder.create<mlir::toy::ConstantOp>(loc, type,
      llvm::cast<mlir::DenseElementsAttr>(value));
}

// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // Note: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.
  
  if (parser.parseKeyword("struct") || parser.parseLess()) {
    return mlir::Type();
  }

  llvm::SmallVector<mlir::Type, 1> elementTypes;
  do {
    mlir::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType)) {
      return nullptr;
    }

    // Check that the type is either a TensorType or another StructType.
    if (!llvm::isa<mlir::TensorType, mlir::toy::StructType>(elementType)) {
      parser.emitError(typeLoc, "element type for a struct must either "
          "be a TensorType or a StructType, got: ")
          << elementType;
      return mlir::Type();
    }
    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater()) {
    return mlir::Type();
  }
  return mlir::toy::StructType::get(elementTypes);
}

// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
    mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  mlir::toy::StructType structType = llvm::cast<mlir::toy::StructType>(type);

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
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

// StructAccessOp

void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state,
    mlir::Value input, size_t index) {
  StructType structTy = llvm::cast<StructType>(input.getType());
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr =
      llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr) {
    return nullptr;
  }

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}

// StructConstantOp
mlir::OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

// ConstantOp
void ConstantOp::inferShapes() {
  getResult().setType(cast<mlir::TensorType>(getValue().getType()));
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
