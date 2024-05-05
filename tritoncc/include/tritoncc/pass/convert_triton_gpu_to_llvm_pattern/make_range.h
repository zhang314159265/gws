#pragma once

namespace tritoncc {

struct MakeRangeOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::MakeRangeOp> {
  using ConvertOpToLLVMPattern<mlir::triton::MakeRangeOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::MakeRangeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    auto rankedTy = op.getResult().getType().cast<mlir::RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    mlir::Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    auto idxs = emitIndices(loc, rewriter, layout, rankedTy, true);
    unsigned elems = idxs.size();
    llvm::SmallVector<mlir::Value> retVals(elems);
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = macro_add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    mlir::Value result = tritoncc::packLLElements(loc, typeConverter, retVals, rewriter, rankedTy);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

void populateMakeRangeOpToLLVMPattern(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  mlir::PatternBenefit benefit
) {
  patterns.add<MakeRangeOpConversion>(typeConverter, benefit);
}

}
