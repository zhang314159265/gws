#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace tritoncc {

void make_ttir(mlir::ModuleOp& M) {
  mlir::MLIRContext *ctx = M.getContext();
  mlir::PassManager pm(ctx);
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createInlinerPass());
  assert(!mlir::failed(pm.run(M.getOperation())));
}

}
