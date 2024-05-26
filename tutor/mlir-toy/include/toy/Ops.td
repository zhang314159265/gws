#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

include "toy/ShapeInferenceInterface.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    // =======
    // FunctionOpInterface methods
    // =======

    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let skipDefaultBuilders = 1;
}

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let arguments = (ins Variadic<F64Tensor>:$input);

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasCanonicalizer = 1;
}

def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ConstantOp : Toy_Op<"constant", [Pure]> {
  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];
}

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let results = (outs StaticShapeTensorOf<[F64]>);
  let hasCanonicalizer = 1;
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def PrintOp : Toy_Op<"print"> {
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}

def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure,
    SameOperandsAndResultShape
  ]> {

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

#endif
