#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace tritoncc {

struct Option {
  int num_warps;
  int num_ctas;
  int capability;
};

mlir::arith::CmpIPredicate cmpIStrToPredicate(const std::string& opstr) {
  if (opstr == "slt") {
    return mlir::arith::CmpIPredicate::slt;
  } else {
    assert(false && "cmpIStrToPredicate unsupported opstr");
  }
}

}
