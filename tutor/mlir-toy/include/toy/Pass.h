#pragma once

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"

#define DEBUG_TYPE "shape-inference"

#include "toy/Dialect.h"

namespace mlir { namespace toy {
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"
} }

namespace toy { // shape inference pass

struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, mlir::OperationPass<mlir::toy::FuncOp>> {

  void runOnOperation() override {
    auto f = getOperation();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        opWorklist.insert(op);
      }
    });

    while (!opWorklist.empty()) {
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end()) {
        break;
      }

      mlir::Operation *op = *nextop;
      opWorklist.erase(op);

      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = llvm::dyn_cast<mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
            "inference interface: ") << *op;
        return signalPassFailure();
      }
    }

    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  static bool allOperandsInferred(mlir::Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](mlir::Type operandType) {
      return llvm::isa<mlir::RankedTensorType>(operandType);
    });
  }

  static bool returnsDynamicShape(mlir::Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](mlir::Type resultType) {
      return !llvm::isa<mlir::RankedTensorType>(resultType);
    });
  }
};

std::unique_ptr<mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

}

namespace toy { // lowering to affine pass

static mlir::MemRefType convertTensorToMemRef(mlir::RankedTensorType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
    mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

using LoopIterationFn = llvm::function_ref<mlir::Value(
    mlir::OpBuilder &rewriter, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs)>;

static void lowerOpToLoops(mlir::Operation *op, mlir::ValueRange operands,
    mlir::PatternRewriter &rewriter,
    LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<mlir::RankedTensorType>(*op->result_type_begin());
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  llvm::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
  mlir::affine::buildAffineLoopNest(
    rewriter, loc, lowerBounds, tensorType.getShape(), steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      mlir::Value valueToStore = processIteration(nestedBuilder, operands, ivs);
      nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, valueToStore, alloc,
          ivs);
    });

  rewriter.replaceOp(op, alloc);
}

struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(mlir::toy::TransposeOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
            mlir::ValueRange loopIvs) -> mlir::Value {
          mlir::toy::TransposeOpAdaptor adaptor(memRefOperands);
          mlir::Value input = adaptor.getInput();

          llvm::SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return builder.create<mlir::affine::AffineLoadOp>(loc, input,
              reverseIvs);
        });
    return mlir::success();
  }
};

struct FuncOpLowering : public mlir::OpConversionPattern<mlir::toy::FuncOp> {
  using OpConversionPattern<mlir::toy::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::FuncOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main") {
      return mlir::failure();
    }

    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](mlir::Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(
      op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConstantOpLowering : public mlir::OpRewritePattern<mlir::toy::ConstantOp> {
  using OpRewritePattern<mlir::toy::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::toy::ConstantOp op,
      mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create those constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    llvm::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
          0, *std::max_element(valueShape.begin(), valueShape.end()))) {
        constantIndices.push_back(
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
      }
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }

    llvm::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.value_begin<mlir::FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::affine::AffineStoreOp>(
          loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++), alloc,
          llvm::ArrayRef(indices));
        return;
      }

      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(0);

    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
            mlir::ValueRange loopIvs) {
          typename BinaryOp::Adaptor adaptor(memRefOperands);

          auto loadedLhs = builder.create<mlir::affine::AffineLoadOp>(
            loc, adaptor.getLhs(), loopIvs);
          auto loadedRhs = builder.create<mlir::affine::AffineLoadOp>(
            loc, adaptor.getRhs(), loopIvs);

          return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });
    return mlir::success();
  }
};

using AddOpLowering = BinaryOpLowering<mlir::toy::AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<mlir::toy::MulOp, mlir::arith::MulFOp>;

struct PrintOpLoweringPartial : public mlir::OpConversionPattern<mlir::toy::PrintOp> {
  using OpConversionPattern<mlir::toy::PrintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::PrintOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands
    rewriter.modifyOpInPlace(op,
        [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::toy::ReturnOp> {
  using OpRewritePattern<mlir::toy::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::toy::ReturnOp op,
      mlir::PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlines.
    if (op.hasOperand()) {
      return mlir::failure();
    }

    // We lower "toy.return" directly to "func.return"
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
    return mlir::success();
  }
};

struct ToyToAffineLoweringPass
    : public mlir::PassWrapper<ToyToAffineLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<
        mlir::func::FuncDialect,
        mlir::affine::AffineDialect,
        mlir::memref::MemRefDialect
    >();
  }

  void runOnOperation() final {
    // define the conversion target
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<
        mlir::affine::AffineDialect,
        mlir::BuiltinDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::memref::MemRefDialect>();

    target.addIllegalDialect<mlir::toy::ToyDialect>();
    target.addDynamicallyLegalOp<mlir::toy::PrintOp>([](mlir::toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
          [](mlir::Type type) { return llvm::isa<mlir::TensorType>(type); });
    });

    // setup patterns
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        TransposeOpLowering,
        ConstantOpLowering,
        AddOpLowering,
        MulOpLowering,
        PrintOpLoweringPartial,
        ReturnOpLowering,
        FuncOpLowering
    >(&getContext());

    // do the partial lowering
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

}

namespace toy { // lowering to llvm dialect pass

class PrintOpLoweringFull : public mlir::ConversionPattern {
 public:
  explicit PrintOpLoweringFull(mlir::MLIRContext *context)
      : ConversionPattern(mlir::toy::PrintOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto memRefType = llvm::cast<mlir::MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
      loc, rewriter, "frmt_spec", llvm::StringRef("%f \0", 4), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
      loc, rewriter, "nl", llvm::StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    llvm::SmallVector<mlir::Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      for (mlir::Operation &nested : *loop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1) {
        rewriter.create<mlir::LLVM::CallOp>(loc, getPrintfType(context), printfRef,
            newLineCst);
      }

      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = llvm::cast<mlir::toy::PrintOp>(op);
    auto elementLoad =
        rewriter.create<mlir::memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<mlir::LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        llvm::ArrayRef<mlir::Value>({formatSpecifierCst, elementLoad}));

    // Notifier the rewriter that this operation has been removed
    rewriter.eraseOp(op);
    return mlir::success();
  }
 private:
  static mlir::LLVM::LLVMFunctionType getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty,
        llvmPtrTy, /*isVarArg=*/true);
    return llvmFnType;
  }

  static mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::PatternRewriter &rewriter,
      mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(context, "printf");
    }

    // Insert the printf function into the body of the parent module
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
        getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  static mlir::Value getOrCreateGlobalString(
      mlir::Location loc, mlir::OpBuilder &builder,
      llvm::StringRef name, llvm::StringRef value,
      mlir::ModuleOp module) {
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true,
        mlir::LLVM::Linkage::Internal, name,
        builder.getStringAttr(value),
        /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getIndexAttr(0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

struct ToyToLLVMLoweringPass
    : public mlir::PassWrapper<ToyToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());

    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // lower toy.print
    patterns.add<PrintOpLoweringFull>(&getContext());

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}

}

#undef DEBUG_TYPE
