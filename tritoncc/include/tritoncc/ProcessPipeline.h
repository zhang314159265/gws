#pragma once

#include "tritoncc/stage/make_ttir.h"
#include "tritoncc/ProcessTTGIR.h"
#include "tritoncc/ProcessLLIR.h"
#include "tritoncc/MakePTX.h"
#include "tritoncc/MakeCubin.h"
#include "tritoncc/Util.h"

namespace tritoncc {

std::string processPipeline(mlir::ModuleOp& M, Option& opt) {
  make_ttir(M);
  processTTGIR(M, opt);
  std::string src = processLLIR(M, opt);
  std::string ptxCode = makePTX(src, opt);
  std::string cubinBytes = makeCubin(ptxCode, opt);
  return cubinBytes;
}

}
