#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace tritoncc {

mlir::arith::CmpIPredicate cmpIStrToPredicate(const std::string& opstr) {
  if (opstr == "slt") {
    return mlir::arith::CmpIPredicate::slt;
  } else {
    assert(false && "cmpIStrToPredicate unsupported opstr");
  }
}

}
