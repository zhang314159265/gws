#pragma once

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "tritoncc/nvidia_util.h"
// #include "TypeConverter.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/reduce.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/make_range.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/func_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/view.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/spmd_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/load_store.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/elementwise_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/control_flow_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/func_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/convert_layout.h"

namespace tritoncc {

class TritonGPUToLLVMTypeConverter : public mlir::LLVMTypeConverter {
 public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(mlir::MLIRContext *ctx, mlir::LowerToLLVMOptions &option, const mlir::DataLayoutAnalysis *analysis = nullptr) : LLVMTypeConverter(ctx, option, analysis) {
    addConversion([&](mlir::_tritoncc::PointerType type) -> std::optional<mlir::Type> {
      return convertTritonPointerType(type);
    });
    addConversion([&](mlir::RankedTensorType type) -> std::optional<mlir::Type> {
      return convertTritonTensorType(type);
    });
    // Internally store float8 as int8
    addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<mlir::Type> {
      assert(false && "fp8");
    });
    addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), 8);
    });
    addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), 8);
    });
    addConversion([&](mlir::Float8E5M2Type type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), 8);
    });
    // Internally store bfloat16 as int16
    addConversion([&](mlir::BFloat16Type type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), 16);
    });
  }

  mlir::Type convertTritonPointerType(mlir::_tritoncc::PointerType type) {
    auto ctx = type.getContext();
    auto pointeeType = type.getPointeeType();
    if (pointeeType.isa<mlir::RankedTensorType>()) {
      assert(false && "convertTritonPointerType");
    }
    return mlir::LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
  }

  mlir::Type getElementTypeForStruct(
      mlir::RankedTensorType type) {
    auto ctx = type.getContext();
    mlir::Attribute layout = type.getEncoding();
    mlir::Type elemTy = convertType(type.getElementType());
    auto dotOpLayout = layout.dyn_cast<mlir::_tritoncc::gpu::DotOperandEncodingAttr>();
    if (!dotOpLayout) {
      return elemTy;
    }
    assert(false && "getElementTypeForStruct");
  }

  mlir::Type convertTritonTensorType(
      mlir::RankedTensorType type) {
    auto ctx = type.getContext();
    mlir::Attribute layout = type.getEncoding();
    llvm::SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
    mlir::Type eltType = getElementTypeForStruct(type);

    if (auto shared_layout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
      assert(false && "shared_layout");
    }

    unsigned numElementsPerThread = getTotalElemsPerThread(type);
    llvm::SmallVector<mlir::Type, 4> types(numElementsPerThread, eltType);
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, types);
  }
};

class TritonLLVMConversionTarget : public mlir::ConversionTarget {
 public:
  explicit TritonLLVMConversionTarget(mlir::MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::LLVM::LLVMDialect>();
    addLegalDialect<mlir::NVVM::NVVMDialect>();
    addLegalDialect<mlir::_tritoncc::nvgpu::NVGPUDialect>();
    addIllegalDialect<mlir::_tritoncc::TritonDialect>();
    addIllegalDialect<mlir::_tritoncc::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::_tritoncc::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public mlir::ConversionTarget {
 public:
  explicit TritonLLVMFunctionConversionTarget(mlir::MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::index::IndexDialect>();
    addLegalDialect<mlir::LLVM::LLVMDialect>();
    addLegalDialect<mlir::NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPUToLLVM : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<ConvertTritonGPUToLLVM>()) {
    this->computeCapability = computeCapability;
  }

  llvm::StringRef getName() const override {
    return "ConvertTritonGPUToLLVM";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertTritonGPUToLLVM>(*this);
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);

    // the following line is super critical to make sure the output
    // llir does not contains tt.reduce but contains llir shuffle instrs.
    option.overrideIndexBitwidth(32);

    TritonLLVMConversionTarget convTarget(*context);
    tritoncc::TritonGPUToLLVMTypeConverter typeConverter(context, option);
    int benefit = 10;
    int numWarps = mlir::_tritoncc::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = mlir::_tritoncc::gpu::TritonGPUDialect::getNumCTAs(mod);

    // lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      tritoncc::TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      mlir::RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, funcPatterns);
      if (failed(applyPartialConversion(mod, funcTarget, std::move(funcPatterns)))) {
        return signalPassFailure();
      }
    }

    initSharedMemory(typeConverter);

    mlir::RewritePatternSet patterns(context);
    tritoncc::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    tritoncc::populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
    // TODO dot op to llvm
    tritoncc::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, computeCapability, benefit);
    tritoncc::populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, benefit);
    patterns.add<tritoncc::ReduceOpConversion>(typeConverter);

    tritoncc::populateControlFlowOpToLLVMPattern(typeConverter, patterns, benefit);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    tritoncc::populateSPMDOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateMakeRangeOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
 private:
  void initSharedMemory(mlir::LLVMTypeConverter &typeConverter) {
    mlir::ModuleOp mod = getOperation();
    mlir::OpBuilder B(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(B.getIntegerType(8));

    auto arrayTy = mlir::LLVM::LLVMArrayType::get(elemTy, 0);
    B.create<mlir::LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/false, mlir::LLVM::Linkage::External,
      "global_smem", /*value=*/mlir::Attribute(), /*alignment=*/16,
      static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }

  int computeCapability;
};

}
