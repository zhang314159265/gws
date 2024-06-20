#ifndef TRITON_TYPES
#define TRITON_TYPES

include "mlir/IR/AttrTypeBase.td"
include "tritoncc/dialect/Triton/Dialect.td"

class TritonTypeDef<string name, string _mnemonic>
    : TypeDef<Triton_Dialect, name> {
  let mnemonic = _mnemonic;
}

// Floating-point Type
def TT_Float : AnyTypeOf<[F16, BF16, F32, F64], "floating-point">;

// Pointer Type in TableGen
class TT_PtrOf<list<Type> pointeeTypes> :
    DialectType<Triton_Dialect,
        And<[CPred<"$_self.isa<::mlir::_tritoncc::PointerType>()">,
             Concat<"[](::mlir::Type pointeeType) { return ",
                    SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                    "; }($_self.cast<::mlir::_tritoncc::PointerType>().getPointeeType())">]>,
                      "ptr", "::mlir::_tritoncc::PointerType">;

// Pointer Type in C++ (corresponding to `TT_PtrOf`)
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
  let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$pointeeType,
      "int":$addressSpace
    ), [{
      return $_get(pointeeType.getContext(), pointeeType, addressSpace);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def TT_Ptr : TT_PtrOf<[AnyType]>;
def TT_FloatTensor : RankedTensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;

def TT_PtrTensor : RankedTensorOf<[TT_Ptr]>;

def TT_PtrLike : AnyTypeOf<[TT_Ptr, TT_PtrTensor]>;

def TT_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def TT_IntTensor : RankedTensorOf<[TT_Int]>;
def TT_IntLike : AnyTypeOf<[TT_Int, TT_IntTensor]>;


def TT_Tensor : RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>;

def TT_BoolTensor : RankedTensorOf<[I1]>;
def TT_BoolLike : AnyTypeOf<[I1, TT_BoolTensor]>;

// Pointer type to tensor type: `ptr<tensor<>>`
def TT_TensorPtr : TT_PtrOf<[TT_Tensor]>;

def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike, TT_TensorPtr]>;

#endif
