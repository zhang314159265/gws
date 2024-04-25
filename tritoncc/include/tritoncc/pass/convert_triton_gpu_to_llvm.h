#pragma once

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "TypeConverter.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/reduce.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/make_range.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/func_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/view.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/spmd_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/load_store.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/elementwise_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/control_flow_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/func_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/convert_layout.h"

namespace tritoncc {

class TritonLLVMConversionTarget : public mlir::ConversionTarget {
 public:
  explicit TritonLLVMConversionTarget(mlir::MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::LLVM::LLVMDialect>();
    addLegalDialect<mlir::NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<mlir::triton::TritonDialect>();
    addIllegalDialect<mlir::triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public mlir::ConversionTarget {
 public:
  explicit TritonLLVMFunctionConversionTarget(mlir::MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::index::IndexDialect>();
    addLegalDialect<mlir::LLVM::LLVMDialect>();
    addLegalDialect<mlir::NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPUToLLVM : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<ConvertTritonGPUToLLVM>()) {
    this->computeCapability = computeCapability;
  }

  llvm::StringRef getName() const override {
    return "ConvertTritonGPUToLLVM";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertTritonGPUToLLVM>(*this);
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);

    // the following line is super critical to make sure the output
    // llir does not contains tt.reduce but contains llir shuffle instrs.
    option.overrideIndexBitwidth(32);

    TritonLLVMConversionTarget convTarget(*context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    int benefit = 10;
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

    // lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      mlir::RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, funcPatterns);
      if (failed(applyPartialConversion(mod, funcTarget, std::move(funcPatterns)))) {
        return signalPassFailure();
      }
    }

    initSharedMemory(typeConverter);

    mlir::RewritePatternSet patterns(context);
    tritoncc::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    tritoncc::populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
    // TODO dot op to llvm
    tritoncc::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, computeCapability, benefit);
    tritoncc::populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, benefit);
    patterns.add<tritoncc::ReduceOpConversion>(typeConverter);

    tritoncc::populateControlFlowOpToLLVMPattern(typeConverter, patterns, benefit);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    tritoncc::populateSPMDOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateMakeRangeOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
 private:
  void initSharedMemory(mlir::LLVMTypeConverter &typeConverter) {
    mlir::ModuleOp mod = getOperation();
    mlir::OpBuilder B(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(B.getIntegerType(8));

    auto arrayTy = mlir::LLVM::LLVMArrayType::get(elemTy, 0);
    B.create<mlir::LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/false, mlir::LLVM::Linkage::External,
      "global_smem", /*value=*/Attribute(), /*alignment=*/16,
      static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }

  int computeCapability;
};

}
