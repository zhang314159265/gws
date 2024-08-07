#pragma once

#ifdef add
// defined in Utility.h as a macro and in llvm/IR/LegacyPassManager.h as a method
#undef add
#endif

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Support/SourceMgr.h"

namespace tritoncc {

std::string translateLLVMIRToASM(
    llvm::Module &module,
    const std::string &triple,
    const std::string &proc,
    const std::string &features,
    const std::vector<std::string> &flags,
    bool enable_fp_fusion,
    bool isObject)
{
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *flagPtr = static_cast<llvm::cl::opt<bool>*>(options[flag]);
    assert(flagPtr);
    flagPtr->setValue(true);
  }

  // inline everything
  for (llvm::Function &f : module.functions()) {
    if (!f.hasFnAttribute(llvm::Attribute::NoInline)) {
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  if (enable_fp_fusion) {
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  }
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
    module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
    std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions()) {
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    }
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);
  }
  return result;
}

std::string make_ptx(std::string &llvmIR, Option &opt) {
  std::string ptxCode;
  { // follow llvm.translate_to_asm
    std::string triple = "nvptx64-nvidia-cuda";
    std::string proc = "sm_90a"; // hardcoded for h100
    std::string features;
    std::vector<std::string> flags = {"nvptx-short-ptr"};
    bool enable_fp_fusion = true;
    bool isObject = false;

    llvm::LLVMContext context;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
    llvm::SMDiagnostic error;
    std::unique_ptr<llvm::Module> module =
        llvm::parseIR(buffer->getMemBufferRef(), error, context);
    if (!module) {
      llvm::report_fatal_error(
        "failed to parse IR: " + error.getMessage() +
        "lineno: " + std::to_string(error.getLineNo()));
    }
    ptxCode = translateLLVMIRToASM(
        *module, triple, proc, features, flags, enable_fp_fusion, isObject);
  }

  #if 0
  // Need replace ptx version if we are using the ptxas in triton
  //   third_party/nvidia/backend/bin/ptxas
  // No need to do this if we are using the system ptxas
  std::string ptx_version = "8.3"; // hardcode for the ptxas shipped with triton
  {
    // assume that the version stored in ptxcode was 8.0. Avoid hardcode here..
    auto pos = ptxCode.find(".version 8.0\n");
    assert(pos != std::string::npos);
    ptxCode.replace(pos + 9, 3, ptx_version);
  }
  #endif

  return ptxCode;
}

}
