#pragma once

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

namespace tritoncc {
void loadDialects(mlir::MLIRContext& ctx) {
  { // TODO make this a common function that can be called by other test_xx.cpp
    // follows ir.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect>();
    // ctx.loadDialect<mlir::triton::TritonDialect>();
    mlir::registerBuiltinDialectTranslation(registry); 
    mlir::registerLLVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }

  { // follws triton.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    mlir::triton::nvgpu::NVGPUDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }


}

}
