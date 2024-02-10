#pragma once

#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "tritoncc/Util.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"

#include "TypeConverter.h"

#include "tritoncc/ElementwiseOpToLLVM.h"

#if 1
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
}
}
#endif

namespace mlir { namespace triton {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit);
} }

namespace tritoncc {

// encounter link error for missing typeinfo with this:
// https://gist.github.com/shunting314/6c1eba0c51b3d6a49e3da75bc5343cef
// Update: this is resolve by add -fno-rtti compliation options.
// Thanks to this post https://discourse.llvm.org/t/python-api-problem/945
#if 0
struct MyConvertTritonGPUToLLVM : public mlir::triton::impl::ConvertTritonGPUToLLVMBase<MyConvertTritonGPUToLLVM> {
 public:
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;
  MyConvertTritonGPUToLLVM(int32_t computeCapability, mlir::triton::Target target)
    : ConvertTritonGPUToLLVMBase({computeCapability, target}) { }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    // TODO
  }

  void runOnOperation() override {
  }
 private:
};
#endif

namespace { // copied from TritonGPUToLLVM.cpp
using namespace mlir;
class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx, mlir::triton::Target target)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    #if 0
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    #endif
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};
}

#if 1
struct MyConvertTritonGPUToLLVM : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  MyConvertTritonGPUToLLVM(int32_t computeCapability, mlir::triton::Target target)
    : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<MyConvertTritonGPUToLLVM>()) {
    this->computeCapability = computeCapability;
    this->target = target;
  }
  MyConvertTritonGPUToLLVM(const MyConvertTritonGPUToLLVM& other) : mlir::OperationPass<mlir::ModuleOp>(other) { }

  llvm::StringRef getName() const override {
    return "MyConvertTritonGPUToLLVM";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<MyConvertTritonGPUToLLVM>(*this);
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    TritonLLVMConversionTarget convTarget(*context, target);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    int benefit = 10;

    initSharedMemory(typeConverter);

    mlir::RewritePatternSet patterns(context);

    // mlir::triton::populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);

    patterns.add<tritoncc::FAddOpConversion>(typeConverter);
    #if 0
    mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, computeCapability, benefit);
    #endif

    // This error:
    // a.out: /home/shunting/ws/triton/lib/Conversion/TritonGPUToLLVM/Utility.h:372: mlir::Value mlir::LLVM::getStackPointer(mlir::PatternRewriter&, mlir::FunctionOpInterface): Assertion `globalBase' failed.
    // may indicate we need some shared memory pass first.
    #if 1
    mlir::triton::populateReduceOpToLLVMPatterns(
      typeConverter, patterns, computeCapability, benefit);
    #endif

    assert(!failed(applyPartialConversion(mod, convTarget, std::move(patterns))));
  }

  /*
   * Workaround: LLVM ERROR: Loading a dialect (llvm) while in a multi-threaded execution context (maybe the PassManager): this can indicate a missing `dependentDialects` in a pass for example.
   */
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect>();
  }
 private:
  // Copied from triton code
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    mlir::ModuleOp mod = getOperation();
    mlir::OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }

  int32_t computeCapability;
  mlir::triton::Target target;
};
#endif

void processLLIR(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
  mlir::triton::gpu::TMAMetadataTy* tmaMetadata = nullptr;
  #if 0
  // after this pass the generate llir file for test_sum is super large: https://gist.github.com/shunting314/89db675f75cec48229fb95febb290574 
  // don't know why yet.
  //
  // But the llir file for test_add still looks reasonable: https://gist.github.com/shunting314/02d2b35604353698a59d1628b74a1d06
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(
    opt.capability, mlir::triton::NVVM, tmaMetadata
  ));
  #endif
  #if 1
  pm.addPass(std::make_unique<MyConvertTritonGPUToLLVM>(opt.capability, mlir::triton::NVVM));
  #endif

  assert(!mlir::failed(pm.run(M.getOperation())));
}

}
