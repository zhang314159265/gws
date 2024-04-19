#pragma once

#include <cassert>
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"

namespace tritoncc {

struct GetProgramIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = mlir::LLVM::NVIDIA::llGetPid(op.getAxisAsInt(), op->getLoc(),
                               op->getParentOfType<ModuleOp>(), rewriter);
    rewriter.replaceOp(op, programId);
    return success();
  }
};

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
}

}
