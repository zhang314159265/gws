#include "tritoncc/dialect/Triton/Dialect.h"
#include "tritoncc/dialect/Triton/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tritoncc/dialect/Triton/Types.cpp.inc"

void mlir::_tritoncc::TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tritoncc/dialect/Triton/Types.cpp.inc"
  >();
}

void mlir::_tritoncc::TritonDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "tritoncc/dialect/Triton/Ops.cpp.inc"
  >();

  addInterfaces<TritonInlinerInterface>();
}

#define GET_OP_CLASSES
#include "tritoncc/dialect/Triton/Ops.cpp.inc"

#include "tritoncc/dialect/Triton/OpsEnums.cpp.inc"

namespace tritoncc {
bool isTensorPointerType(mlir::Type type) {
  if (mlir::_tritoncc::PointerType ptrType = type.dyn_cast<mlir::_tritoncc::PointerType>()) {
    return ptrType.getPointeeType().isa<mlir::RankedTensorType>();
  }
  return false;
}
}

namespace mlir {
namespace _tritoncc {

mlir::LogicalResult ExpandDimsOp::inferReturnTypes(
  mlir::MLIRContext *context, std::optional<mlir::Location> loc,
  mlir::ValueRange operands,
  DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
  llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // infer shape
  auto arg = operands[0];
  auto argTy = arg.getType().cast<mlir::RankedTensorType>();
  auto retShape = argTy.getShape().vec();
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  retShape.insert(retShape.begin() + axis, 1);
  // infer encoding
  mlir::Attribute argEncoding = argTy.getEncoding();
  mlir::Attribute retEncoding;
  if (argEncoding) {
    mlir::Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = llvm::dyn_cast<tritoncc::DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface
        ->inferExpandDimsOpEncoding(argEncoding, axis, retEncoding, loc)
        .failed()) {
      return emitOptionalError(loc, "failed to infer layout for ExpandDimsOp");
    }
  }
  // create type
  auto argEltTy = argTy.getElementType();
  inferredReturnTypes.push_back(
    mlir::RankedTensorType::get(retShape, argEltTy, retEncoding));
  return mlir::success();
}

static llvm::SmallVector<mlir::RankedTensorType>
getInputTypesImpl(const mlir::Operation::operand_range &operands) {
  llvm::SmallVector<mlir::RankedTensorType> srcTys;
  srcTys.reserve(operands.size());
  for (const auto &ty : operands.getTypes()) {
    srcTys.push_back(ty.cast<mlir::RankedTensorType>());
  }
  return srcTys;
}

llvm::SmallVector<mlir::RankedTensorType> ReduceOp::getInputTypes() {
  return getInputTypesImpl(this->getOperands());
}

static llvm::SmallVector<mlir::Type>
getElementTypesImpl(const mlir::Operation::operand_range &operands) {
  llvm::SmallVector<mlir::Type> srcElemTys;
  srcElemTys.reserve(operands.size());
  for (const auto &op : operands) {
    srcElemTys.push_back(
        op.getType().cast<mlir::RankedTensorType>().getElementType());
  }
  return srcElemTys;
}

llvm::SmallVector<mlir::Type> ReduceOp::getElementTypes() {
  return getElementTypesImpl(this->getOperands());
}

static mlir::LogicalResult
inferReduceReturnShape(const mlir::RankedTensorType &argTy, const mlir::Type &retEltTy,
    int axis, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  auto retShape = argTy.getShape().vec();
  retShape.erase(retShape.begin() + axis);
  if (retShape.empty()) {
    // 0d-tensor -> scalar
    inferredReturnTypes.push_back(retEltTy);
  } else {
    // nd-tensor where n > 1
    // infer encoding
    mlir::Attribute argEncoding = argTy.getEncoding();
    mlir::Attribute retEncoding;
    if (argEncoding) {
      mlir::Dialect &dialect = argEncoding.getDialect();
      auto inferLayoutInterface =
          llvm::dyn_cast<tritoncc::DialectInferLayoutInterface>(&dialect);
      if (inferLayoutInterface
          ->inferReduceOpEncoding(argEncoding, axis, retEncoding)
          .failed()) {
        llvm::report_fatal_error("failed to infer layout for ReduceOp");
        return mlir::failure();
      }
    }
    // create type
    inferredReturnTypes.push_back(
        mlir::RankedTensorType::get(retShape, retEltTy, retEncoding));
  }
  return mlir::success();
}

void ReduceOp::build(OpBuilder &builder, OperationState &state,
    ValueRange operands, int axis) {
  llvm::SmallVector<mlir::Type> inferredReturnTypes;
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto argTy = operands[i].getType().cast<mlir::RankedTensorType>();
    auto retEltTy = argTy.getElementType();
    (void) inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes);
  }
  ReduceOp::build(builder, state, inferredReturnTypes, operands, axis);
}

void FuncOp::build(OpBuilder &builder, OperationState &state, llvm::StringRef name,
    FunctionType type, llvm::ArrayRef<NamedAttribute> attrs,
    llvm::ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
      builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), mlir::TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
    builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
    getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
    Value mask, Value other, CacheModifier cache,
    EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, other, /*boundaryCheck=*/{},
    /*padding=*/{}, cache, evict, isVolatile);
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
    Value mask, Value other,
    std::optional<mlir::ArrayRef<int32_t>> boundaryCheck,
    std::optional<PaddingOption> padding, CacheModifier cache,
    EvictionPolicy evict, bool isVolatile) {
  // Operands
  state.addOperands(ptr);
  if (mask) {
    state.addOperands(mask);
    if (other) {
      state.addOperands(other);
    }
  }

  // Attributes
  state.addAttribute(
    getOperandSegmentSizesAttrName(state.name),
    builder.getDenseI32ArrayAttr({1, (mask ? 1 : 0), (other ? 1 : 0)}));
  if (boundaryCheck.has_value()) {
    state.addAttribute(getBoundaryCheckAttrName(state.name),
        builder.getDenseI32ArrayAttr(boundaryCheck.value()));
  }
  if (padding.has_value()) {
    state.addAttribute(
      getPaddingAttrName(state.name),
      PaddingOptionAttr::get(builder.getContext(), padding.value()));
  }
  state.addAttribute(getCacheAttrName(state.name),
      CacheModifierAttr::get(builder.getContext(), cache));
  state.addAttribute(getEvictAttrName(state.name),
      EvictionPolicyAttr::get(builder.getContext(), evict));
  state.addAttribute(getIsVolatileAttrName(state.name),
      builder.getBoolAttr(isVolatile));

  // Result type
  mlir::Type resultType = tritoncc::getLoadOpResultType(builder, ptr.getType());
  state.addTypes({resultType});
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value ptr,
    Value value, Value mask, CacheModifier cache,
    EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, mask, /*boundaryCheck=*/{},
      cache, evict);
}

mlir::Type PointerType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess()) {
    return mlir::Type();
  }

  mlir::Type pointeeType;
  if (parser.parseType(pointeeType)) {
    return mlir::Type();
  }

  int addressSpace = 1;
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace)) {
      return mlir::Type();
    }
  }

  if (parser.parseGreater()) {
    return mlir::Type();
  }

  return PointerType::get(pointeeType, addressSpace);
}

void PointerType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ", " << getAddressSpace() << ">";
}

} }
