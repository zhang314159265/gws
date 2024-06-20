#pragma once

namespace tritoncc {

struct GetProgramIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::_tritoncc::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<mlir::_tritoncc::GetProgramIdOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::GetProgramIdOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value programId = tritoncc::llGetPid(
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
