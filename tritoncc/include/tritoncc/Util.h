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

static void dump2dVector(llvm::SmallVector<llvm::SmallVector<unsigned>>& offset) {
  std::cerr << "Dump 2d vector [" << offset.size() << " x " << offset[0].size() << "]" << std::endl;
  for (auto& row : offset) {
    std::cerr << "  [";
    for (auto& val: row) {
      std::cerr << val << " ";
    }
    std::cerr << "]" << std::endl;
  }
}

}
