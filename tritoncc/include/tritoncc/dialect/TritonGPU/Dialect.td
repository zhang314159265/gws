#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";
  let cppNamespace = "::mlir::_tritoncc";

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static int getNumCTAs(mlir::ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-ctas")) {
        return 1;
      }
      return mod->getAttr("triton_gpu.num-ctas").cast<mlir::IntegerAttr>().getInt();
    }

    static int getNumWarps(mlir::ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-warps")) {
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      }
      return mod->getAttr("triton_gpu.num-warps").cast<mlir::IntegerAttr>().getInt();
    }

    static int getThreadsPerWarp(mlir::ModuleOp mod) {
      mlir::Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu.threads-per-warp");
      if (!threadsPerWarp) {
        return 32;
      }
      return threadsPerWarp.cast<mlir::IntegerAttr>().getInt();
    }
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif
