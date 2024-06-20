#pragma once

namespace tritoncc {

struct SplatOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::SplatOp> {
  using ConvertOpToLLVMPattern<mlir::triton::SplatOp>::ConvertOpToLLVMPattern;

  static mlir::Value convertSplatLikeOp(
      mlir::Type elemType, mlir::Type resType, mlir::Value constVal,
      const mlir::LLVMTypeConverter *typeConverter,
      mlir::ConversionPatternRewriter &rewriter,
      mlir::Location loc) {
    auto tensorTy = resType.cast<mlir::RankedTensorType>();
    auto srcType = typeConverter->convertType(tensorTy);
    if (auto structTy = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(srcType)) {
      srcType = structTy.getBody()[0];
    }
    // If the type sizes don't match we need to pack constants
    if (srcType.isIntOrFloat() && constVal.getType().getIntOrFloatBitWidth() != srcType.getIntOrFloatBitWidth()) {
      assert(false && "bit width mismatch");
    }
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = tritoncc::getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<mlir::Value> elems(elemsPerThread, llSrc);
    return tritoncc::packLLElements(loc, typeConverter, elems, rewriter, resType);
  }

  mlir::LogicalResult matchAndRewrite(mlir::triton::SplatOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    auto llStruct = convertSplatLikeOp(
        src.getType(), op.getType(), src,
        typeConverter, rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return mlir::success();
  }
};

struct ExpandDimsOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::ExpandDimsOp> {
  using ConvertOpToLLVMPattern<mlir::triton::ExpandDimsOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ExpandDimsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto srcVals = tritoncc::unpackLLElements(loc, adaptor.getSrc(), rewriter);

    auto srcTy = op.getSrc().getType().cast<mlir::RankedTensorType>();
    auto resultTy = op.getType().template cast<mlir::RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<mlir::_tritoncc::SliceEncodingAttr>();
    if (!srcLayout) {
      return emitOptionalError(
        loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }

    auto resultLayout = resultTy.getEncoding();

    auto srcOffsets = tritoncc::emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = tritoncc::emitOffsetForLayout(resultLayout, resultTy);
    std::map<mlir::SmallVector<unsigned>, mlir::Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); ++i) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    llvm::SmallVector<mlir::Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); ++i) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.at(offset));
    }
    mlir::Value ret = tritoncc::packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return mlir::success();
  }
};

struct BroadcastOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<mlir::triton::BroadcastOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::BroadcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    mlir::Value src = adaptor.getSrc();
    mlir::Value result = op.getResult();
    auto srcTy = op.getSrc().getType().cast<mlir::RankedTensorType>();
    auto resultTy = op.getType().cast<mlir::RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    auto typeConverter = getTypeConverter();

    assert(rank == resultTy.getRank());
    auto order = tritoncc::getOrder(srcLayout);
    auto srcOffsets = tritoncc::emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = tritoncc::emitOffsetForLayout(resultLayout, resultTy);
    llvm::SmallVector<mlir::Value> srcVals = tritoncc::unpackLLElements(loc, src, rewriter);

    std::map<llvm::SmallVector<unsigned>, mlir::Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); ++i) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    llvm::SmallVector<mlir::Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); ++i) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); ++j) {
        if (srcShape[j] == 1) {
          offset[j] = 0;
        }
      }
      resultVals.push_back(srcValues.at(offset));
    }

    mlir::Value resultStruct = tritoncc::packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);

    rewriter.replaceOp(op, {resultStruct});
    return mlir::success();
  }
};

void populateViewOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
  mlir::PatternBenefit benefit
) {
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
}

}
