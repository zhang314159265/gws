#pragma once

#include "mlir/IR/PatternMatch.h"

#include "toy/Dialect.h"
#include "toy/Pattern.inc"  // generated

namespace toy {

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<mlir::toy::TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::toy::TransposeOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::TransposeOp op,
      mlir::PatternRewriter &rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    mlir::toy::TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<mlir::toy::TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp) {
      return mlir::failure();
    }

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return mlir::success();
  }
};

}
