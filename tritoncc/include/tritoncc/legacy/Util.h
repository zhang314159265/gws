#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace tritoncc {

struct Option {
  int num_warps;
  int num_ctas;
  int capability;
};

}
