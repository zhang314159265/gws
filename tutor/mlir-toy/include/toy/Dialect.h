#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace toy {
#include "toy/ShapeInferenceOpInterfaces.h.inc"
}
}

#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
