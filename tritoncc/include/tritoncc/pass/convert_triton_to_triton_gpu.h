#pragma once

#define USE_TRITON 0

#if USE_TRITON
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#endif

#include "mlir/Transforms/DialectConversion.h"

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

void populateArithPatterns(TritonGPUTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<
    GenericOpPattern<mlir::arith::AddIOp>,
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
      TritonFuncOpPattern
    >(typeConverter, context);

    TritonGPUConversionTarget target(*context, typeConverter);
    if (mlir::failed(mlir::applyPartialConversion(
      mod, target, std::move(patterns)
    ))) {
      return signalPassFailure();
    }

    llvm::errs() << "Module after triton to triton gpu pass\n";
    mod.dump();
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
