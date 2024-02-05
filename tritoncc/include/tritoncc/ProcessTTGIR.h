#pragma once

#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "tritoncc/Util.h"

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

namespace tritoncc {

void processTTGIR(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
    opt.num_warps, 32, opt.num_ctas, opt.capability
  ));

  assert(!mlir::failed(pm.run(M.getOperation())));

  #if 0
  std::cout << "M after processTTGIR" << std::endl;
  M.dump();
  #endif
}

}
