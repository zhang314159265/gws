#ifndef TRITONNVIDIAGPU_DIALECT
#define TRITONNVIDIAGPU_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def TritonNvidiaGPU_Dialect : Dialect {
  let name = "triton_nvidia_gpu";
  // add a underscore to avoid ambiguity with ::tritoncc
  let cppNamespace = "::mlir::_tritoncc";

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];
}

#endif
