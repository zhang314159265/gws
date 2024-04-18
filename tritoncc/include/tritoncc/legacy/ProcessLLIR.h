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
#include "tritoncc/pass/convert_triton_gpu_to_llvm.h"
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
