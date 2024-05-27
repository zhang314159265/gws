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

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for out toy operations.
  let hasConstantMaterializer = 1;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks
  let useDefaultTypePrinterParser = 1;
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// Provide a definition for the toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
        "Toy struct type">;

def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;

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
  let arguments = (ins Variadic<Toy_Type>:$input);

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

def ConstantOp : Toy_Op<"constant", [ConstantLike, Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];
  
  let hasFolder = 1;
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
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Toy_Type>:$inputs);

  let results = (outs Toy_Type);

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

def StructAccessOp : Toy_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let arguments = (ins Toy_StructType:$input, I64Attr:$index);
  let results = (outs Toy_Type:$output);

  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  let hasFolder = 1;
}

def StructConstantOp : Toy_Op<"struct_constant", [ConstantLike, Pure]> {
  let arguments = (ins ArrayAttr:$value);
  let results = (outs Toy_StructType:$output);

  let hasFolder = 1;
}

#endif
