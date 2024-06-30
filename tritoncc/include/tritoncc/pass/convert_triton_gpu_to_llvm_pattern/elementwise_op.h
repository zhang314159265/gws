#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#endif

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h"

namespace tritoncc {

class MultipleOperandsRange : public llvm::iterator_range<llvm::SmallVector<llvm::SmallVector<mlir::Value>>::iterator> {
  using ContainerT = llvm::SmallVector<llvm::SmallVector<mlir::Value>>;

 public:
  using llvm::iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

llvm::SmallVector<mlir::Value> reorderValues(const llvm::SmallVector<mlir::Value> &values, mlir::Type inType, mlir::Type ouType) {
  auto inTensorTy = inType.dyn_cast<mlir::RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<mlir::RankedTensorType>();
  if (!inTensorTy || !ouTensorTy) {
    return values;
  }
  auto inEncoding = llvm::dyn_cast<mlir::_tritoncc::gpu::DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding = llvm::dyn_cast<mlir::_tritoncc::gpu::DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding) {
    return values;
  }
  assert(false && "reorderValues");
}

template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase : public mlir::ConvertOpToLLVMPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(
      mlir::LLVMTypeConverter &typeConverter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      mlir::PatternBenefit benefit)
    : mlir::ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
      axisAnalysisPass(axisAnalysisPass) { }

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    mlir::Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    mlir::Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = tritoncc::unpackLLElements(loc, operand, rewriter);
      subOperands = tritoncc::unpackI32(subOperands, argTy, rewriter, loc,
          this->getTypeConverter());
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands)) {
        allOperands[v.index()].push_back(v.value());
      }
    }
    if (allOperands.size() == 0) {
      allOperands.push_back({});
    }

    llvm::SmallVector<mlir::Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
        op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0) {
        return mlir::failure();
      }
      for (auto v : curr) {
        if (!static_cast<bool>(v)) {
          return mlir::failure();
        }
        resultVals.push_back(v);
      }
      it += curr.size();
    }

    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals = maybeDeduplicate(op, resultVals);
    resultVals = tritoncc::packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    mlir::Value view = tritoncc::packLLElements(loc, this->getTypeConverter(), resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, view);
    return mlir::success();
  }

  // don't do deduplication or now.
  llvm::SmallVector<mlir::Value> maybeDeduplicate(SourceOp op,
      llvm::SmallVector<mlir::Value> resultVals) const {
    return resultVals;
  }
 protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

mlir::Type getElementType(mlir::Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
    return tensorType.getElementType();
  }
  return type;
}

struct FAddOpConversion : ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion> {
  using Base = ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  llvm::SmallVector<mlir::Value> createDestOps(
      mlir::arith::AddFOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter,
      mlir::Type elemTy, MultipleOperandsRange operands,
      mlir::Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      assert(false && "FAddOp BF16");
    } else {
      return {rewriter.create<mlir::LLVM::FAddOp>(loc, elemTy, operands[0][0], operands[0][1])};
    }
  }
};

struct FSubOpConversion : ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion> {
  using Base = ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  llvm::SmallVector<mlir::Value> createDestOps(
      mlir::arith::SubFOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter,
      mlir::Type elemTy, MultipleOperandsRange operands,
      mlir::Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      assert(false && "FSubOp BF16");
    } else {
      return {rewriter.create<mlir::LLVM::FSubOp>(loc, elemTy, operands[0][0], operands[0][1])};
    }
  }
};


struct AddPtrOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::_tritoncc::AddPtrOp> {
  using ConvertOpToLLVMPattern<mlir::_tritoncc::AddPtrOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::AddPtrOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = resultTy.dyn_cast<mlir::RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = tritoncc::getTotalElemsPerThread(resultTy);
      mlir::Type elemTy = typeConverter->convertType(
          resultTensorTy.getElementType().cast<mlir::_tritoncc::PointerType>().getPointeeType());
      mlir::Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = tritoncc::unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = tritoncc::unpackLLElements(loc, adaptor.getOffset(), rewriter);
      llvm::SmallVector<mlir::Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      mlir::Value view = tritoncc::packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(false && "AddPtr Not RankedTensorType");
    }
    return mlir::success();
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
        SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
    ElementwiseOpConversionBase<SourceOp,
      ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  llvm::SmallVector<DestOp> createDestOps(
      SourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter,
      mlir::Type elemTy, MultipleOperandsRange operands,
      mlir::Location loc) const {
    #if 0
    if (operands[0].size() == 2) {
      auto rhs = operands[0][1];
      std::string str;
      llvm::raw_string_ostream os(str);
      os << rhs;
      std::cerr << "str is " << str << std::endl; // TODO
    }
    #endif
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
        adaptor.getAttributes().getValue()) };
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<mlir::arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<mlir::arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  llvm::SmallVector<mlir::LLVM::ICmpOp> createDestOps(
      mlir::arith::CmpIOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter,
      mlir::Type elemTy, MultipleOperandsRange operands,
      mlir::Location loc) const {
    return {rewriter.create<mlir::LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static mlir::LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(mlir::arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__) \
  case mlir::arith::CmpIPredicate::item__: \
    return mlir::LLVM::ICmpPredicate::item__
      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(slt);
      __PRED_ENUM(sge);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(ult);
      __PRED_ENUM(uge);
      __PRED_ENUM(ule);
#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

#if USE_TRITON
void populateElementwiseOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  ModuleAxisInfoAnalysis &axisInfoAnalysis,
  int computeCapability,
  mlir::PatternBenefit benefit
) {
  mlir::triton::NVIDIA::TargetInfo targetInfo(computeCapability);
  mlir::triton::NVIDIA::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, computeCapability, targetInfo, benefit);
}

#else
void populateElementwiseOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  ModuleAxisInfoAnalysis &axisInfoAnalysis,
  int computeCapability,
  mlir::PatternBenefit benefit
) {
#define POPULATE_ELEMENTWISE_OP(SRC_OP, DST_OP) \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>( \
    typeConverter, axisInfoAnalysis, benefit);

  POPULATE_ELEMENTWISE_OP(mlir::arith::AddIOp, mlir::LLVM::AddOp) // +
  POPULATE_ELEMENTWISE_OP(mlir::arith::AndIOp, mlir::LLVM::AndOp) // &
  POPULATE_ELEMENTWISE_OP(mlir::arith::MulIOp, mlir::LLVM::MulOp) // *
#undef POPULATE_ELEMENTWISE_OP
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}
#endif

}
