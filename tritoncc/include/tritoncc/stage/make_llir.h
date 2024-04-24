#pragma once

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "tritoncc/pass/convert_triton_gpu_to_llvm.h"

namespace mlir { namespace triton {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertNVGPUToLLVMPass();
}}

namespace tritoncc {

std::string optimizeLLIR(mlir::ModuleOp &M, Option &opt) {
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
  std::unique_ptr<llvm::Module> llvm_mod = mlir::translateModuleToLLVMIR(M, ctx);
  if(!llvm_mod.get()) {
    std::cerr << "Got a null llvm::Module pointer" << std::endl;
    std::cerr << "The mlir::ModuleOp is:" << std::endl;
    M.dump();
    assert(false);
  }

  // TODO do something similar to llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
  std::string mod_str;
  llvm::raw_string_ostream os(mod_str);
  os << *llvm_mod;
  return os.str();
}

std::string make_llir(mlir::ModuleOp &M, Option &opt) {
  mlir::MLIRContext &ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
  pm.addPass(std::make_unique<tritoncc::ConvertTritonGPUToLLVM>(opt.capability));

  pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());

  bool success = !mlir::failed(pm.run(M.getOperation()));
  if (!success) {
    std::cerr << "make_llir fail" << std::endl;
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
