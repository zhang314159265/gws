#pragma once

#include "tritoncc/ProcessTTIR.h"
#include "tritoncc/ProcessTTGIR.h"
#include "tritoncc/ProcessLLIR.h"
#include "tritoncc/MakePTX.h"
#include "tritoncc/Util.h"

namespace tritoncc {

void processPipeline(mlir::ModuleOp& M, Option& opt) {
  processTTIR(M);
  processTTGIR(M, opt);
  std::string src = processLLIR(M, opt);
  makePTX(src, opt);
}

}
