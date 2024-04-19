#pragma once

namespace tritoncc {

struct GetProgramIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<mlir::triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::GetProgramIdOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value programId = mlir::LLVM::NVIDIA::llGetPid(
        op.getAxisAsInt(), op->getLoc(),
        op->getParentOfType<mlir::ModuleOp>(), rewriter);
    rewriter.replaceOp(op, programId);
    return mlir::success();
  }
};

void populateSPMDOpToLLVMPattern(
    mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
}

}
