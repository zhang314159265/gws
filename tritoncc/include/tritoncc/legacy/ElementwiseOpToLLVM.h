#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace tritoncc {

#if 0
class FAddOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::arith::AddFOp> {
 public:
  using SourceOp = mlir::arith::AddFOp;
  using DestOp = LLVM::FAddOp;

  FAddOpConversion(mlir::LLVMTypeConverter& typeConverter) : mlir::ConvertOpToLLVMPattern<SourceOp>(typeConverter, 10) {
  }

  Value createDestOps(ConversionPatternRewriter& rewriter, Type elemTy, SmallVector<Value>& operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands[0], operands[1]);
  }

  // If we return success(), we must have made some rewriting.
  //
  // std::cout op will just print an address.
  //
  // op.dump() will print details of the op
  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type resultTy = op.getType();
    Type resultElementType = mlir::getElementTypeOrSelf(resultTy);
    Type elemTy = getTypeConverter()->convertType(resultElementType);
    // print the op
    std::cout << "FAddOpConversion got:" << std::endl;
    op.dump();

    mlir::Type argTy = op->getOperand(0).getType();
    Location loc = op->getLoc();
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      std::cout << "Got operand:" << std::endl;
      operand.dump();
      SmallVector<Value> subOperands = unpackLLElements(loc, operand, rewriter);

      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands)) {
        allOperands[v.index()].push_back(v.value());
      }
    }
    assert(allOperands.size() > 0);
    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end; ++it) {
      resultVals.push_back(createDestOps(rewriter, elemTy, *it, loc));
    }
    Value view = packLLElements(loc, getTypeConverter(), resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, view);
    #if 1
    return success();
    #else
    return failure();
    #endif
  }
 private:
};
#endif

}
