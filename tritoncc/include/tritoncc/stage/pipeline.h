#pragma once

namespace tritoncc {
struct Option {
  int num_warps;
  int num_ctas;
  int capability;
};
}

#include "tritoncc/stage/make_ttir.h"
#include "tritoncc/stage/make_ttgir.h"
#include "tritoncc/stage/make_llir.h"
#include "tritoncc/stage/make_ptx.h"
#include "tritoncc/stage/make_cubin.h"

namespace tritoncc {

std::string compile(mlir::ModuleOp &M, Option &opt) {
  make_ttir(M);
  make_ttgir(M, opt);
  std::string llirSrc = make_llir(M, opt);
  std::string ptxCode = make_ptx(llirSrc, opt);
  std::string cubinBytes = make_cubin(ptxCode, opt);
  return cubinBytes;
}

}
