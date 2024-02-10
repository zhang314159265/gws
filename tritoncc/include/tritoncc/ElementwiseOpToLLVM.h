#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace tritoncc {

static Value packLLElements(Location loc, const LLVMTypeConverter* typeConverter, ValueRange resultVals, ConversionPatternRewriter& rewriter, Type type) {
  LLVM::LLVMStructType structType = typeConverter->convertType(type).dyn_cast<LLVM::LLVMStructType>();
  assert(structType);
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  auto elementTypes = structType.getBody();
  for (const auto& v : llvm::enumerate(resultVals)) {
    assert(v.value());
    assert(v.value().getType() == elementTypes[v.index()]);
    rewriter.create<LLVM::InsertValueOp>(loc, structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

static SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct, ConversionPatternRewriter& rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      llvmStruct.getType().isa<triton::PointerType>() ||
      llvmStruct.getType().isa<LLVM::LLVMPointerType>()) {
    return {llvmStruct};
  }
  ArrayRef<Type> types = llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  std::cout << "Unpack to " << types.size() << " elements" << std::endl;
  SmallVector<Value> results(types.size());
  for (int i = 0; i < types.size(); ++i) {
    results[i] = rewriter.create<LLVM::ExtractValueOp>(loc, types[i], llvmStruct, i);
  }
  return results;
}

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

}
