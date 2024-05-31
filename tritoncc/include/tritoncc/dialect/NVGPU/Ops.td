#ifndef NVGPU_OPS
#define NVGPU_OPS

include "tritoncc/dialect/NVGPU/Dialect.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

class NVGPU_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<NVGPU_Dialect, mnemonic, traits>;

def NVGPU_ClusterCTAIdOp : NVGPU_Op<"cluster_id", [Pure]> {
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

#endif
