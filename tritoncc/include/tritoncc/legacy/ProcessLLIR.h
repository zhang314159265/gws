#pragma once

#include <iostream>
#include <fstream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "tritoncc/legacy/Util.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Allocation.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "TypeConverter.h"
#include "PatternTritonGPUOpToLLVM.h"

#include "tritoncc/legacy/ReduceOpConversion.h"
#include "tritoncc/legacy/SPMDOpToLLVM.h"
#include "tritoncc/legacy/MakeRangeOpToLLVM.h"
#include "tritoncc/legacy/ViewOpToLLVM.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/load_store.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/elementwise_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/control_flow_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/func_op.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm_pattern/convert_layout.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"

#if 1
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
}
}
#endif


namespace mlir { namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass();

void populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit);
} }

namespace tritoncc {

namespace { // copied from TritonGPUToLLVM.cpp
using namespace mlir;
class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    #if 1
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    #endif
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

#if 1
class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    #if 1
    addLegalDialect<mlir::index::IndexDialect>();
    #endif
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};
#endif

}

// copies from triton
#if 0
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(ptrTy);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new argument to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(ptrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!LLVM::isKernel(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    if (LLVM::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("nvvm.kernel",
                         rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      rewriter.eraseOp(amendedFuncOp);
    }
    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid",
                       rewriter.getDenseI32ArrayAttr(32 * numWarps));

    // required by AxisInfoAnalysis
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};
#endif

#if 1

struct ConvertTritonGPUToLLVM : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  int computeCapability;
  ConvertTritonGPUToLLVM(int32_t computeCapability)
    : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<ConvertTritonGPUToLLVM>()) {
    this->computeCapability = computeCapability;
  }
  ConvertTritonGPUToLLVM(const ConvertTritonGPUToLLVM& other) : mlir::OperationPass<mlir::ModuleOp>(other) { }

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

    #if 1 // this is super critical to make sure
    // the output does not contains tt.reduce and contains shfl
    option.overrideIndexBitwidth(32);
    #endif

    TritonLLVMConversionTarget convTarget(*context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    int benefit = 10;
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

    #if 1
    // lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, funcPatterns);
      if (failed(applyPartialConversion(mod, funcTarget, std::move(funcPatterns)))) {
        return signalPassFailure();
      }
    }
    #endif

    #if 1
    // allocate shared memory and set barrier
    {
      mlir::ModuleAllocation allocation(mod);
      mlir::ModuleMembarAnalysis membarPass(&allocation);
      membarPass.run();
    }
    #endif

    initSharedMemory(typeConverter);

    mlir::RewritePatternSet patterns(context);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);


    tritoncc::populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
    // mlir::triton::NVIDIA::populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);

    tritoncc::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, computeCapability, benefit);
    tritoncc::populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis, benefit);
    // mlir::triton::NVIDIA::populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);

    patterns.add<tritoncc::ReduceOpConversion>(typeConverter);

    tritoncc::populateControlFlowOpToLLVMPattern(typeConverter, patterns, benefit); // this is needed
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns); // this is needed

    tritoncc::populateSPMDOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateMakeRangeOpToLLVMPattern(typeConverter, patterns, benefit);
    tritoncc::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit); 
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  /*
   * Workaround: LLVM ERROR: Loading a dialect (llvm) while in a multi-threaded execution context (maybe the PassManager): this can indicate a missing `dependentDialects` in a pass for example.
   */
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect>();
  }
 private:
  // Copied from triton code
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    mlir::ModuleOp mod = getOperation();
    mlir::OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }
};
#endif

std::string optimizeLLIR(mlir::ModuleOp& M, Option& opt) {
  { // init_targets
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  }
  llvm::LLVMContext ctx;
  std::cerr << "Before mlir::translateModuleToLLVMIR" << std::endl;
  std::unique_ptr<llvm::Module> llvm_mod = mlir::translateModuleToLLVMIR(M, ctx);
  std::cerr << "After mlir::translateModuleToLLVMIR" << std::endl;
  std::cerr << "llvm::Module ptr: " << llvm_mod.get() << std::endl;
  if (!llvm_mod.get()) {
    std::cerr << "Got a null llvm::Module pointer" << std::endl;
    std::cerr << "The mlir::ModuleOp is:" << std::endl;
    M.dump();
    assert(false);
  }

  // TODO: do something similar to llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
  std::string mod_str;
  llvm::raw_string_ostream os(mod_str);
  os << *llvm_mod;
  return os.str();
}

std::string processLLIR(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
  pm.addPass(std::make_unique<tritoncc::ConvertTritonGPUToLLVM>(opt.capability));

  pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());

  bool success = !mlir::failed(pm.run(M.getOperation()));
  if (!success) {
    std::cerr << "processLLIR fail" << std::endl;
    M.dump();
  }
  assert(success);

  auto llirSrc = optimizeLLIR(M, opt);
  { // dump llir to a file
    std::ofstream out_llir;
    out_llir.open("/tmp/tritoncc.llir");
    out_llir << llirSrc;
    out_llir.close();

    std::cerr << "Written llir code to /tmp/tritoncc.llir" << std::endl;
  }
  return llirSrc;
}

}
