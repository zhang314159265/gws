#pragma once

#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace tritoncc {

struct FuncOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::_tritoncc::FuncOp> {
  FuncOpConversion(mlir::LLVMTypeConverter &converter, int numWarps,
      mlir::PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps) { }

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::FuncOp funcOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!tritoncc::isKernel(funcOp)) {
      assert(false && "amendFuncOp");
    }
    mlir::LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
      amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return mlir::failure();
    }

    auto ctx = funcOp->getContext();

    if (tritoncc::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("nvvm.kernel",
        rewriter.getIntegerAttr(tritoncc::type::u1Ty(ctx), 1));
    } else {
      assert(false && "device function");
    }
    newFuncOp->setAttr("nvvm.maxntid",
        rewriter.getDenseI32ArrayAttr(32 * numWarps));
    // required by AxisInfoAnalysis
    rewriter.eraseOp(funcOp);
    return mlir::success();
  }
 private:
  int numWarps{0};
};

}
