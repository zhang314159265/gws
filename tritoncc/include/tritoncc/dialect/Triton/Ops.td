#ifndef TRITON_OPS
#define TRITON_OPS

include "tritoncc/dialect/Triton/Dialect.td"
include "tritoncc/dialect/Triton/Types.td"
include "tritoncc/dialect/Triton/AttrDefs.td"
include "tritoncc/dialect/Triton/Interfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Op Base
class TT_Op<string mnemonic, list<Trait> traits = []> :
  Op<Triton_Dialect, mnemonic,
      !listconcat(traits, [
      //  TensorSizeTrait, VerifyTensorLayoutsTrait
      ])> {
}

// arith.bitcast doesn't support pointers
def TT_BitcastOp : TT_Op<"bitcast", [
  Elementwise,
  Pure,
]> {
  let arguments = (ins TT_Type:$src);
  let results = (outs TT_Type:$result);
}

// Pointer Arith Ops
def TT_AddPtrOp : TT_Op<"addptr", [
  Pure,
  Elementwise,
]> {
  let arguments = (ins TT_PtrLike:$ptr, TT_IntLike:$offset);
  let results = (outs TT_PtrLike:$result);
}

def TT_LoadOp : TT_Op<"load", [
  AttrSizedOperandSegments,
]> {
  let arguments = (ins
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other,
    OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
    OptionalAttr<TT_PaddingOptionAttr>:$padding,
    TT_CacheModifierAttr:$cache,
    TT_EvictionPolicyAttr:$evict,
    BoolAttr:$isVolatile);

  let results = (outs TT_Type:$result);

  let builders = [
    // A tensor of pointers or pointer to a scalar with mask and other
    OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "mlir::_tritoncc::CacheModifier":$cache, "mlir::_tritoncc::EvictionPolicy":$evict, "bool":$isVolatile)>,
    OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other,
      "std::optional<llvm::ArrayRef<int32_t>>":$boundaryCheck,
      "std::optional<PaddingOption>":$padding, "mlir::_tritoncc::CacheModifier":$cache,
      "mlir::_tritoncc::EvictionPolicy":$evict, "bool":$isVolatile)>,
  ];
}

def TT_StoreOp : TT_Op<"store", [
]> {
  let arguments = (ins
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask,
    OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
    DefaultValuedAttr<TT_CacheModifierAttr, "mlir::_tritoncc::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "mlir::_tritoncc::EvictionPolicy::NORMAL">:$evict);

  let builders = [
    // A tensor of pointers or a pointer to a scalar with mask
    OpBuilder<(ins "Value":$ptr, "Value":$value, "Value":$mask, "mlir::_tritoncc::CacheModifier":$cache, "mlir::_tritoncc::EvictionPolicy":$evict)>,
  ];
}

def TT_MakeRangeOp : TT_Op<"make_range", [Pure]> {
  let arguments = (ins I32Attr:$start, I32Attr:$end);
  let results = (outs TT_IntTensor:$result);
}

def TT_SplatOp : TT_Op<"splat", [Pure]> {
  let arguments = (ins TT_Type:$src);
  let results = (outs TT_Tensor:$result);
}

def TT_ExpandDimsOp : TT_Op<"expand_dims", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins TT_Tensor:$src, I32Attr:$axis);
  let results = (outs TT_Tensor:$result);
}

def TT_BroadcastOp : TT_Op<"broadcast", [
  Pure,
  SameOperandsAndResultElementType,
  SameOperandsAndResultEncoding
]> {
  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$result);
}

def TT_TransOp : TT_Op<"trans", [Pure]> {
  let arguments = (
    ins TT_Tensor:$src,
    DenseI32ArrayAttr:$order
  );
  let results = (outs TT_Tensor:$result);
}

def TT_ReduceOp : TT_Op<"reduce", [
  Pure
]> {
  let arguments = (ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis);
  let results = (outs Variadic<TT_Type>:$result);
  let regions = (region SizedRegion<1>:$combineOp);
  let builders = [
    OpBuilder<(ins "ValueRange":$srcs, "int":$axis)>,
  ];
  let extraClassDeclaration = [{
    llvm::SmallVector<mlir::RankedTensorType> getInputTypes();
    llvm::SmallVector<mlir::Type> getElementTypes();
  }];
}

def TT_ReduceReturnOp : TT_Op<"reduce.return", [
  HasParent<"ReduceOp">,
  Pure,
  Terminator,
  ReturnLike
]> {
  let arguments = (ins Variadic<AnyType>:$result);
}

def TT_HistogramOp : TT_Op<"histogram", [Pure]> {
  let arguments = (ins TT_IntTensor:$src);
  let results = (outs TT_IntTensor:$result);
}

// The following ops, including `call`, `func`, and `return` are copied and modified
// from https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td
// We could revert it back once MLIR has a better inliner interface.

def CallOp : TT_Op<"call", [
  CallOpInterface
]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
  ];

  let extraClassDeclaration = [{
    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
    
    // Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    // Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
  }];
}

def FuncOp : TT_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove,
  OpAsmOpInterface
]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    // Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def TT_GetProgramIdOp : TT_Op<"get_program_id", [Pure]> {
  let arguments = (ins TT_ProgramDim:$axis);
  let results = (outs I32:$result);
  let extraClassDeclaration = [{
    int32_t getAxisAsInt() {
      return static_cast<int32_t>(getAxis());
    }
  }];
}

def ReturnOp : TT_Op<"return", [Pure, HasParent<"FuncOp">, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
}

// cat is not `pure` because it may reorder elements
def TT_CatOp : TT_Op<"cat", [
  NoMemoryEffect,
  SameTypeOperands,
  SameOperandsAndResultElementType]> {
  let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);
  let results = (outs TT_Tensor:$result);
}

def TT_ReshapeOp : TT_Op<"reshape", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins TT_Tensor:$src, BoolAttr:$allow_reorder, OptionalAttr<UnitAttr>:$efficient_layout);
  let results = (outs TT_Tensor:$result);
}

def TT_ScanOp : TT_Op<"scan", [
  Pure
]> {
}

def TT_DotOp : TT_Op<"dot", [
  Pure
]> {
}

def TT_AtomicRMWOp : TT_Op<"atomic_rmw", [
]> {
}

def TT_AtomicCASOp : TT_Op<"atomic_cas", [
]> {
}

def TT_JoinOp : TT_Op<"join", [
]> {
}

def TT_SplitOp : TT_Op<"split", [
]> {
}

def TT_AssertOp : TT_Op<"assert", [
]> {
}

#endif
