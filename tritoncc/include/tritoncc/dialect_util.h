#pragma once

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

namespace tritoncc {
void loadDialects(mlir::MLIRContext &ctx) {
  { // follows ir.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }
  { // follows nvidia.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::_tritoncc::TritonNvidiaGPUDialect,
                    mlir::triton::nvgpu::NVGPUDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }
}

}
