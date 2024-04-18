#pragma once


// TODO remove this
#include "../legacy/ProcessLLIR.h"

namespace tritoncc {

std::string make_llir(mlir::ModuleOp &M, Option &opt) {
  return processLLIR(M, opt);
  // assert(false && "make_llir");
}

}
