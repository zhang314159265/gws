#pragma once

#include "tritoncc/stage/make_ttir.h"
#include "tritoncc/stage/make_ttgir.h"
#include "tritoncc/legacy/ProcessLLIR.h"
#include "tritoncc/legacy/MakePTX.h"
#include "tritoncc/legacy/MakeCubin.h"
#include "tritoncc/legacy/Util.h"

namespace tritoncc {

std::string processPipeline(mlir::ModuleOp& M, Option& opt) {
  make_ttir(M);
  make_ttgir(M, opt);
  std::string src = processLLIR(M, opt);
  std::string ptxCode = makePTX(src, opt);
  std::string cubinBytes = makeCubin(ptxCode, opt);
  return cubinBytes;
}

}
