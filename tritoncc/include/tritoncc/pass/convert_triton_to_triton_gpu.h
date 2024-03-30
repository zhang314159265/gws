#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#endif

#include "llvm/Support/Debug.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-triton-to-triton-gpu"

namespace tritoncc {

static void addNamedAttrs(mlir::Operation *op, mlir::DictionaryAttr dictAttrs) {
  for (const mlir::NamedAttribute attr : dictAttrs.getValue()) {
    if (!op->hasAttr(attr.getName())) {
      op->setAttr(attr.getName(), attr.getValue());
    }
  }
}

template <typename Op>
class GenericOpPattern : public mlir::OpConversionPattern<Op> {
 public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> retTypes;
    if (mlir::failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
        retTypes))) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
        op->getAttrs());
    return mlir::success();
  }
};

class TritonGPUTypeConverter : public mlir::TypeConverter {
 public:
  TritonGPUTypeConverter(mlir::MLIRContext *context, int numWarps, int threadsPerWarp, int numCTAs) : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp), numCTAs(numCTAs) {
    addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
      return type;
    });

    // add encoding for tensor
    addConversion([this](mlir::RankedTensorType tensorType) -> mlir::RankedTensorType {
      if (tensorType.getEncoding()) {
        return tensorType;
      }
      llvm::ArrayRef<int64_t> shape = tensorType.getShape();
      mlir::triton::gpu::BlockedEncodingAttr encoding =
        mlir::triton::gpu::getDefaultBlockedEncoding(this->context, shape, this->numWarps,
            this->threadsPerWarp, this->numCTAs);
      return mlir::RankedTensorType::get(shape, tensorType.getElementType(), encoding);
    });
    
    addTargetMaterialization([&](
      mlir::OpBuilder &builder, mlir::RankedTensorType tensorType,
      mlir::ValueRange inputs, mlir::Location loc) {

      mlir::triton::gpu::ConvertLayoutOp cast = builder.create<mlir::triton::gpu::ConvertLayoutOp>(loc, tensorType, inputs);
      return std::optional<mlir::Value>(cast.getResult());
    });
  }
 private:
  mlir::MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

class TritonGPUConversionTarget : public mlir::ConversionTarget {
 public:
  explicit TritonGPUConversionTarget(mlir::MLIRContext &context, TritonGPUTypeConverter &typeConverter)
      : mlir::ConversionTarget(context) {
    addLegalDialect<mlir::triton::gpu::TritonGPUDialect>();

    addDynamicallyLegalDialect<mlir::arith::ArithDialect, mlir::triton::TritonDialect>([&](mlir::Operation *op) -> std::optional<bool> {
      bool hasLegalRegions = true;
      for (mlir::Region &region : op->getRegions()) {
        hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
      }
      return hasLegalRegions && typeConverter.isLegal(op);
    });
  }
};

class TritonFuncOpPattern : public mlir::OpConversionPattern<mlir::triton::FuncOp> {
 public:
  using mlir::OpConversionPattern<mlir::triton::FuncOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::triton::FuncOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter* converter = getTypeConverter();
    mlir::triton::FuncOp newOp = rewriter.replaceOpWithNewOp<mlir::triton::FuncOp>(
      op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
        newOp.getBody().end());
    if (mlir::failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter))) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

class TritonExpandDimsPattern : public mlir::OpConversionPattern<mlir::triton::ExpandDimsOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::triton::ExpandDimsOp op,
      OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType argType = adaptor.getSrc().getType().cast<mlir::RankedTensorType>();
    mlir::Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding) {
      return mlir::failure();
    }
    mlir::triton::gpu::BlockedEncodingAttr argEncoding = _argEncoding.cast<mlir::triton::gpu::BlockedEncodingAttr>();
    std::vector<long> retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);

    // return encoding
    llvm::SmallVector<unsigned> retSizePerThread = argEncoding.getSizePerThread();
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    llvm::SmallVector<unsigned> retThreadsPerWarp = argEncoding.getThreadsPerWarp();
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    llvm::SmallVector<unsigned> retWarpsPerCTA = argEncoding.getWarpsPerCTA();
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    llvm::SmallVector<unsigned> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    // cta layout
    mlir::triton::gpu::CTALayoutAttr argCTALayout = argEncoding.getCTALayout();
    llvm::SmallVector<unsigned> retCTAsPerCGA{argCTALayout.getCTAsPerCGA()};
    retCTAsPerCGA.insert(retCTAsPerCGA.begin() + op.getAxis(), 1);
    llvm::SmallVector<unsigned> retCTASplitNum{argCTALayout.getCTASplitNum()};
    retCTASplitNum.insert(retCTASplitNum.begin() + op.getAxis(), 1);
    llvm::SmallVector<unsigned> retCTAOrder = insertOrder(argCTALayout.getCTAOrder(), op.getAxis());
    mlir::triton::gpu::CTALayoutAttr retCTALayout = mlir::triton::gpu::CTALayoutAttr::get(
      getContext(),
      retCTAsPerCGA,
      retCTASplitNum,
      retCTAOrder);
    mlir::triton::gpu::BlockedEncodingAttr retEncoding =
        mlir::triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
          retThreadsPerWarp, retWarpsPerCTA,
          retOrder, retCTALayout);
    // convert operand to slice of return type
    mlir::Attribute newArgEncoding = mlir::triton::gpu::SliceEncodingAttr::get(
      getContext(), op.getAxis(), retEncoding
    );
    mlir::RankedTensorType newArgType = mlir::RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    mlir::triton::gpu::ConvertLayoutOp newSrc = rewriter.create<mlir::triton::gpu::ConvertLayoutOp>(
      op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(
      rewriter.replaceOpWithNewOp<mlir::triton::ExpandDimsOp>(
        op, newSrc, adaptor.getAxis()),
      adaptor.getAttributes());
    return mlir::success(); 
  }
 private:
  llvm::SmallVector<unsigned> insertOrder(llvm::ArrayRef<unsigned> order, unsigned axis) const {
    llvm::SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (int i = 0; i < resOrder.size(); ++i) {
      if (resOrder[i] >= axis) {
        ++resOrder[i];
      }
    }
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

class TritonBroadcastPattern : public mlir::OpConversionPattern<mlir::triton::BroadcastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::triton::BroadcastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType srcType = adaptor.getSrc().getType().cast<mlir::RankedTensorType>();
    mlir::Attribute srcEncoding = srcType.getEncoding();
    if (!srcEncoding) {
      return mlir::failure();
    }
    mlir::Type retType = mlir::RankedTensorType::get(
      op.getType().getShape(), op.getType().getElementType(), srcEncoding);

    addNamedAttrs(rewriter.replaceOpWithNewOp<mlir::triton::BroadcastOp>(
      op, retType, adaptor.getOperands()), adaptor.getAttributes());
    return mlir::success();
  }
};

class TritonReducePattern : public mlir::OpConversionPattern<mlir::triton::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::triton::ReduceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::triton::ReduceOp newReduce = rewriter.create<mlir::triton::ReduceOp>(op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    mlir::Region &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp, newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return mlir::success();
  }
};

class TritonCatPattern : public mlir::OpConversionPattern<mlir::triton::CatOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::triton::CatOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType retType = getTypeConverter()->convertType(op.getType()).cast<mlir::RankedTensorType>();
    mlir::triton::gpu::BlockedEncodingAttr retEncoding
        = retType.getEncoding().cast<mlir::triton::gpu::BlockedEncodingAttr>();
    mlir::Type lhsType = adaptor.getLhs().getType();
    mlir::Type rhsType = adaptor.getRhs().getType();
    int lhsTotalElemsPerThread = mlir::triton::gpu::getTotalElemsPerThread(lhsType);
    int rhsTotalElemsPerThread = mlir::triton::gpu::getTotalElemsPerThread(rhsType);
    int retTotalElemsPerThread = mlir::triton::gpu::getTotalElemsPerThread(retType);
    llvm::ArrayRef<long> retShape = retType.getShape();
    llvm::ArrayRef<unsigned> retOrder = retEncoding.getOrder();
    llvm::SmallVector<unsigned> retSizePerThread = retEncoding.getSizePerThread();
    llvm::SmallVector<unsigned> retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    llvm::SmallVector<unsigned> retWarpsPerCTA = retEncoding.getWarpsPerCTA();

    int newRetTotalElemsPerThread = mlir::nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    // An assetion added by shunting
    assert(newRetTotalElemsPerThread == retTotalElemsPerThread);
    llvm::SmallVector<unsigned> newRetSizePerThread = retSizePerThread;
    newRetSizePerThread[retOrder[0]] *= newRetTotalElemsPerThread / retTotalElemsPerThread;
    mlir::triton::gpu::BlockedEncodingAttr newRetEncoding =
      mlir::triton::gpu::BlockedEncodingAttr::get(
        getContext(), newRetSizePerThread, retThreadsPerWarp,
        retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
    mlir::RankedTensorType newRetType = mlir::RankedTensorType::get(retShape, retType.getElementType(), newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<mlir::triton::CatOp>(
      op, newRetType, adaptor.getOperands()),
      adaptor.getAttributes());
    std::cerr << "For cat: lhsTotalElemsPerThread " << lhsTotalElemsPerThread << ", rhsTotalElemsPerThread " << rhsTotalElemsPerThread << ", retTotalElemsPerThread " << retTotalElemsPerThread << std::endl;
    return mlir::success();
  }
};

void populateArithPatterns(TritonGPUTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<
    GenericOpPattern<mlir::arith::AddIOp>,
    GenericOpPattern<mlir::arith::AndIOp>,
    GenericOpPattern<mlir::arith::MulIOp>,
    GenericOpPattern<mlir::arith::AddFOp>,
    GenericOpPattern<mlir::arith::CmpIOp>
  >(typeConverter, context);
}

class ConvertTritonToTritonGPUPass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit ConvertTritonToTritonGPUPass(int numWarps, int threadsPerWarp,
      int numCTAs, int computeCapability) : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<ConvertTritonToTritonGPUPass>()) {
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  llvm::StringRef getName() const override {
    return "ConvertTritonToTritonGPUPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::IntegerType i32_ty = mlir::IntegerType::get(context, 32);
    mod->setAttr(
      "triton_gpu.num-warps",
      mlir::IntegerAttr::get(i32_ty, llvm::APInt(32, numWarps))
    );
    mod->setAttr(
      "triton_gpu.threads-per-warp",
      mlir::IntegerAttr::get(i32_ty, llvm::APInt(32, threadsPerWarp))
    );
    mod->setAttr(
      "triton_gpu.num-ctas",
      mlir::IntegerAttr::get(i32_ty, llvm::APInt(32, numCTAs))
    );
    mod->setAttr(
      "triton_gpu.compute-capability",
      mlir::IntegerAttr::get(i32_ty, llvm::APInt(32, computeCapability))
    );

    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp, numCTAs);
    mlir::RewritePatternSet patterns(context);

    populateArithPatterns(typeConverter, patterns);
    patterns.insert<
      GenericOpPattern<mlir::triton::LoadOp>,
      GenericOpPattern<mlir::triton::StoreOp>,
      GenericOpPattern<mlir::triton::SplatOp>,
      GenericOpPattern<mlir::triton::AddPtrOp>,
      GenericOpPattern<mlir::triton::MakeRangeOp>,
      TritonFuncOpPattern,
      TritonExpandDimsPattern,
      TritonBroadcastPattern,
      TritonReducePattern,
      TritonCatPattern
    >(typeConverter, context);

    TritonGPUConversionTarget target(*context, typeConverter);
    if (mlir::failed(mlir::applyPartialConversion(
      mod, target, std::move(patterns)
    ))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Module after triton to triton gpu pass\n";
      mod.dump();
    });
  }
 private:
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
  int computeCapability;
};

#if USE_TRITON
static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertTritonToTritonGPUPass(int numWarps, int threadsPerWarp, int numCTAs, int computeCapability) {
  return mlir::triton::createConvertTritonToTritonGPUPass(numWarps, threadsPerWarp, numCTAs, computeCapability);
}
#else
static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertTritonToTritonGPUPass(int numWarps, int threadsPerWarp, int numCTAs, int computeCapability) {
  llvm::errs() << "createConvertTritonToTritonGPUPass numWarps " << numWarps
    << ", threadsPerWarp " << threadsPerWarp
    << ", numCTAs " << numCTAs
    << ", computeCapability " << computeCapability << "\n";
  return std::make_unique<ConvertTritonToTritonGPUPass>(
    numWarps, threadsPerWarp, numCTAs, computeCapability);
}
#endif

}

#undef DEBUG_TYPE
