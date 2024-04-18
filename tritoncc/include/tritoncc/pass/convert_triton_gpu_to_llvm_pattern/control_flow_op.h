#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0
#if USE_TRITON
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#endif

namespace tritoncc {

struct ReturnOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::ReturnOp> {
  using ConvertOpToLLVMPattern<mlir::triton::ReturnOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ReturnOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (funcOp->hasAttr("nvvm.kernel")) {
      // A GPU kernel
      if (op.getNumOperands() > 0) {
        return rewriter.notifyMatchFailure(
          op, "Kernel functions do not support return with operands");
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, mlir::TypeRange(), mlir::ValueRange(), op->getAttrs());
    } else {
      assert(false && "matchAndRewrite device function");
    }
    return mlir::success();
  }
};

#if USE_TRITON

void populateControlFlowOpToLLVMPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    PatternBenefit benefit) {
  mlir::triton::NVIDIA::populateControlFlowOpToLLVMPattern(typeConverter, patterns, benefit);
}

#else
void populateControlFlowOpToLLVMPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
}
#endif

}
