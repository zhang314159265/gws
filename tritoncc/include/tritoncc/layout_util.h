#pragma once

#include "tritoncc/AxisInfo.h"

namespace tritoncc {

unsigned getNumElementsPerThread(mlir::Operation *op, llvm::SmallVector<unsigned> order,
    tritoncc::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  mlir::Value val = getMemAccessPtr(op);
  assert(val.getType().isa<mlir::RankedTensorType>());
  auto ty = val.getType().cast<mlir::RankedTensorType>();
  auto shapePerCTA = tritoncc::getShapePerCTA(ty);
  AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);
  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
  return currPerThread;
}

}
