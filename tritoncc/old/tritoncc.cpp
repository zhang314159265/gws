#include <iostream>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "tritoncc/ProcessPipeline.h"
#include "tritoncc/MLIRUtil.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Require ttir path" << std::endl;
    return -1;
  }
  const char *ttir_path = argv[1];
  mlir::MLIRContext ctx;
  tritoncc::loadDialects(ctx);

  // XXX don't know why calling ->get() does work work but calling ->clone() works.
  mlir::ModuleOp module = mlir::parseSourceFile<mlir::ModuleOp>(ttir_path, &ctx)->clone();

  std::cerr << "Dump the module" << std::endl;
  module.dump();
  std::cerr << "Done dumping" << std::endl;
  tritoncc::Option opt{
     .num_warps=4, // how is this decided?
     .num_ctas=1,
     .capability=90, // H100
  };
  tritoncc::processPipeline(module, opt);

  std::cerr << "bye" << std::endl;
  return 0;
}
