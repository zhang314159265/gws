#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

include "tritoncc/dialect/TritonGPU/Dialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

// Type definition copied from TritonIR TritonTypes.td
// def TT_PTR : TT_PtrOf<[AnyType]>;
// def TT_Tensor : RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic,
      // !listconcat(traits, [VerifyTensorLayoutsTrait])> 
      traits>
{
}

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
    [SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    Pure]> {
  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$result);
  let hasCanonicalizer = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// TODO: triton defines more traits for this op
def TTG_InsertSliceAsyncOp : TTG_Op<"insert_slice_async",
    []> {
}

// TODO: triton defines more traits for this op
def TTG_AllocTensorOp : TTG_Op<"alloc_tensor", []> {
}

// TODO: triton defines more traits for this op
def TTG_ExtractSliceOp : TTG_Op<"extract_slice",
    []> {
}

#endif
