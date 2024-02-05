#pragma once
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace tritoncc {

void processTTIR(mlir::ModuleOp& M) {
  mlir::MLIRContext& ctx = *M.getContext();
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::triton::createCombineOpsPass());
  pm.addPass(mlir::createCSEPass());
  assert(!mlir::failed(pm.run(M.getOperation())));
}

};
