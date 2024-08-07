#pragma once

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h"
#include "tritoncc/dialect/NVGPU/Dialect.h"
#include "tritoncc/dialect/Triton/Dialect.h"
#include "tritoncc/dialect/TritonGPU/Dialect.h"

namespace tritoncc {
void loadDialects(mlir::MLIRContext &ctx) {
  { // follows ir.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::_tritoncc::TritonDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }
  { // follows nvidia.load_dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::_tritoncc::TritonNvidiaGPUDialect,
                    mlir::_tritoncc::nvgpu::NVGPUDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }
}

}
