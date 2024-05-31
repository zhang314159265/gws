#ifndef NVGPU_DIALECT
#define NVGPU_DIALECT

include "mlir/IR/OpBase.td"

def NVGPU_Dialect : Dialect {
  let name = "nvgpu2";  // use a different name to triton
  let cppNamespace = "::mlir::_tritoncc";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif
