#pragma once

#include <fstream>
#include <iostream>
#include <cstdlib>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "tritoncc/legacy/Util.h"
#include "tritoncc/pass/coalesce.h"
#include "tritoncc/pass/remove_layout_conversions.h"
#include "tritoncc/pass/convert_triton_to_triton_gpu.h"

namespace tritoncc {

void make_ttgir(mlir::ModuleOp& M, Option& opt) {
  mlir::MLIRContext *ctx = M.getContext();
  mlir::PassManager pm(ctx);

  pm.addPass(tritoncc::createConvertTritonToTritonGPUPass(
    opt.num_warps, 32, opt.num_ctas, opt.capability
  ));

  pm.addPass(tritoncc::createCoalescePass());
  pm.addPass(tritoncc::createRemoveLayoutConversionsPass());
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
}

}
