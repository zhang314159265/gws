#pragma once

#include "tritoncc/pass/coalesce.h"
#include "tritoncc/pass/remove_layout_conversions.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "mlir/IR/BuiltinOps.h"
#include "tritoncc/Util.h"
#include "tritoncc/pass/convert_triton_to_triton_gpu.h"

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace tritoncc {

void processTTGIR(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);

  pm.addPass(tritoncc::createConvertTritonToTritonGPUPass(
    opt.num_warps, 32, opt.num_ctas, opt.capability
  ));

  // This pass is critical to make sure we have mma layout in ttgir.
  // But cause the following error right now:
  //   unsupported layout conversion
  //   UNREACHABLE executed at /home/shunting/ws/triton/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:102!
  //
  // pm.addPass(mlir::triton::gpu::createAccelerateMatmulPass(opt.capability));

  //XXX this pass is the key to avoid error: LLVM ERROR: DotOperandEncodingAttr non-NvidiaMmaEncodingAttr parent not supported yet
  pm.addPass(mlir::triton::gpu::createReduceDataDuplicationPass());

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

  #if 0
  pm.addPass(mlir::triton::gpu::createCoalescePass());
  #else
  pm.addPass(tritoncc::createCoalescePass());
  #endif
  #if 1
  pm.addPass(mlir::triton::gpu::createRemoveLayoutConversionsPass());
  #else
  pm.addPass(tritoncc::createRemoveLayoutConversionsPass());
  #endif
  assert(!mlir::failed(pm.run(M.getOperation())));

  { // dump ttgir to a file

    // follow __str__ in src/ir.cc
    std::string str;
    llvm::raw_string_ostream os(str);
    M.print(os);

    std::ofstream out_file;
    out_file.open("/tmp/tritoncc.ttgir");
    out_file << os.str();
    out_file.close();

    std::cerr << "Written ttgir code to /tmp/tritoncc.ttgir" << std::endl;
  }


  #if 0
  std::cout << "M after processTTGIR" << std::endl;
  M.dump();
  exit(0);
  #endif
}

}
