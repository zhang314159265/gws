#pragma once

#include <iostream>
#include <cstdlib>
#include "mlir/IR/BuiltinOps.h"
#include "tritoncc/Util.h"

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace tritoncc {

void processTTGIR(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
    opt.num_warps, 32, opt.num_ctas, opt.capability
  ));

  // the following passes makes sure the sizePerThread for the the reduction
  // in test_sum.cpp is <1, 4> rather than <1, 1>.
  // Not realy, need revisit. Also make the sizePerThread <1, 4> is not critical.
  // Since test_sum.py works with sizePerThread being <1, 1>
  #if 0
  {
    pm.addPass(mlir::triton::gpu::createCoalescePass());
    pm.addPass(mlir::triton::gpu::createRemoveLayoutConversionsPass());
  }
  #endif

  assert(!mlir::failed(pm.run(M.getOperation())));

  #if 0
  std::cout << "M after processTTGIR" << std::endl;
  M.dump();
  exit(0);
  #endif
}

}
