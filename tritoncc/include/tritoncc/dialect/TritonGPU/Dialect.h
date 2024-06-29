#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#endif

#include <numeric>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "tritoncc/dialect/TritonGPU/Dialect.h.inc"

#include "tritoncc/dialect/TritonGPU/AttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tritoncc/dialect/TritonGPU/AttrDefs.h.inc"

#define GET_OP_CLASSES
#include "tritoncc/dialect/TritonGPU/Ops.h.inc"

#include "tritoncc/util.h"

struct TritonGPUInferLayoutInterface
    : public tritoncc::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  mlir::LogicalResult
  inferTransOpEncoding(mlir::Attribute operandEncoding,
      llvm::ArrayRef<int32_t> order,
      mlir::Attribute &resultEncoding) const override {
    assert(false && "inferTransOpEncoding");
  }

  mlir::LogicalResult
  inferReduceOpEncoding(mlir::Attribute operandEncoding,
      unsigned axis,
      mlir::Attribute &resultEncoding) const override {
    resultEncoding = mlir::_tritoncc::gpu::SliceEncodingAttr::get(
      getDialect()->getContext(),
      axis,
      operandEncoding);
    return mlir::success();
  }

  mlir::LogicalResult
  inferExpandDimsOpEncoding(mlir::Attribute operandEncoding,
      unsigned axis,
      mlir::Attribute &resultEncoding,
      std::optional<mlir::Location> location) const override {
    auto sliceEncoding = operandEncoding.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>();
    if (!sliceEncoding) {
      return emitOptionalError(
        location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    }
    if (sliceEncoding.getDim() != axis) {
      return emitOptionalError(
        location, "Incompatible slice dimension for ExpandDimsOp operand");
    }
    resultEncoding = sliceEncoding.getParent();
    return mlir::success();
  }

  mlir::LogicalResult
  inferDotOpEncoding(mlir::Attribute operandEncoding,
      unsigned opIdx,
      mlir::Attribute retEncoding,
      std::optional<mlir::Location> location) const override {
    assert(false && "inferDotOpEncoding");
  }

  mlir::LogicalResult
  inferReshapeOpNoReorderEncoding(llvm::ArrayRef<int64_t> srcShape,
      mlir::Attribute srcEnc,
      llvm::ArrayRef<int64_t> dstShape,
      mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const override {
    assert(false && "inferReshapeOpNoReorderEncoding");
  }

  mlir::LogicalResult
  inferJoinOpEncoding(mlir::Attribute srcEnc, mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const override {
    assert(false && "inferJoinOpEncoding");
  }

  mlir::LogicalResult
  inferSplitOpEncoding(mlir::Attribute srcEnc, mlir::Attribute &dstEnc,
      std::optional<mlir::Location> loc) const override {
    assert(false && "inferSplitOpEncoding");
  }

  mlir::LogicalResult
  verifyDotOpEncodingCompatibility(mlir::Operation *op,
      mlir::Attribute operandEncodingA,
      mlir::Attribute operandEncodingB) const override {
    assert(false && "verifyDotOpEncodingCompatibility");
  }
};

namespace tritoncc {

static unsigned getTotalElemsPerThread(mlir::Attribute layout, llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<mlir::_tritoncc::TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else {
    llvm::errs() << "layout " << layout << "\n";
    llvm::report_fatal_error("--> getTotalElemsPerThread not implemented");
    return 0;
  }
}

static llvm::SmallVector<unsigned> getElemsPerThread(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<mlir::_tritoncc::TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getElemsPerThread not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

static unsigned getTotalElemsPerThread(mlir::Type type) {
  if (type.isIntOrIndexOrFloat() || type.isa<mlir::_tritoncc::PointerType>()) {
    return 1;
  }
  auto tensorType = type.cast<mlir::RankedTensorType>();
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
      tensorType.getElementType());
}

// 1 element per thread
// order = reverse(arange(rank))
static mlir::_tritoncc::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape,
    int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  return mlir::_tritoncc::gpu::BlockedEncodingAttr::get(
    context, shape, sizePerThread,
    order, numWarps, threadsPerWarp,
    numCTAs);
}

static llvm::SmallVector<unsigned> getCTAsPerCGA(mlir::Attribute layout) {
  llvm::ArrayRef<unsigned> ref;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getCTAsPerCGA();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::gpu::AMDMfmaEncodingAttr>()) {
    return {1, 1};
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
    ref = sharedLayout.getCTALayout().getCTAsPerCGA();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
  }
  return llvm::SmallVector<unsigned>(ref.begin(), ref.end());
}

static llvm::SmallVector<unsigned> getCTASplitNum(mlir::Attribute layout) {
  llvm::SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getCTASplitNum();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::gpu::AMDMfmaEncodingAttr>()) {
    assert(false && "AMDMfmaEncodingAttr");
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
    assert(false && "SharedEncodingAttr");
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

static llvm::SmallVector<unsigned> getCTAOrder(mlir::Attribute layout) {
  llvm::SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    res = distributedLayout.getCTAOrder();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::gpu::AMDMfmaEncodingAttr>()) {
    return {0, 1};
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
    res = llvm::SmallVector<unsigned>(sharedLayout.getCTALayout().getCTAOrder());
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

static mlir::_tritoncc::gpu::CTALayoutAttr getCTALayout(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return mlir::_tritoncc::gpu::CTALayoutAttr::get(
        layout.getContext(), getCTAsPerCGA(distributedLayout),
        getCTASplitNum(distributedLayout),
        getCTAOrder(distributedLayout));
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
    return sharedLayout.getCTALayout();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  }
  return {};
}

static llvm::SmallVector<int64_t> getShapePerCTA(llvm::ArrayRef<unsigned> CTASplitNum,
    llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with emitCTAOffsetForLayout
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    shapePerCTA[i] = shape[i] / splitNum;
  }
  return shapePerCTA;
}

static llvm::SmallVector<int64_t> getShapePerCTA(mlir::Attribute layout, llvm::ArrayRef<int64_t> shape) {
  if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>()) {
    assert(false && "SharedEncodingAttr");
  }
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

static llvm::SmallVector<int64_t> getShapePerCTA(mlir::Type type) {
  auto tensorType = type.cast<mlir::RankedTensorType>();
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

static llvm::SmallVector<unsigned>
getShapePerCTATile(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> tensorShape = llvm::ArrayRef<int64_t>()) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getShapePerCTATile(tensorShape);
  } else {
    llvm::report_fatal_error("getShapePerCTATile not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

static llvm::SmallVector<unsigned> getThreadsPerWarp(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getThreadsPerWarp();
  } else {
    llvm::report_fatal_error("getThreadsPerWarp not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

static unsigned getWarpSize(mlir::Attribute layout) {
  unsigned size = 1;
  auto threadsPerWarp = getThreadsPerWarp(layout);
  for (auto e : threadsPerWarp) {
    size *= e;
  }
  return size;
}

static llvm::SmallVector<unsigned> getOrder(mlir::Attribute layout) {
  if (auto blockedLayout = layout.dyn_cast<mlir::_tritoncc::gpu::BlockedEncodingAttr>()) {
    return llvm::SmallVector<unsigned>(blockedLayout.getOrder().begin(),
        blockedLayout.getOrder().end());
  } else if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>()) {
    llvm::SmallVector<unsigned> parentOrder = getOrder(sliceLayout.getParent());
    unsigned dim = sliceLayout.getDim();
    llvm::SmallVector<unsigned> order;
    for (unsigned d : parentOrder) {
      if (d == dim) {
        continue;
      } else if (d > dim) {
        order.push_back(d - 1);
      } else {
        order.push_back(d);
      }
    }
    return order;
  } else {
    llvm::errs() << "layout " << layout << "\n";
    llvm::report_fatal_error("Unimplemented usage of getOrder");
  }
  return {};
}

static llvm::SmallVector<unsigned> getSizePerThread(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getSizePerThread();
  } else {
    llvm::report_fatal_error("getSizePerThread not implemented");
    return {};
  }
}

static llvm::SmallVector<unsigned> getContigPerThread(mlir::Attribute layout) {
  if (auto mmaLayout = layout.dyn_cast<mlir::_tritoncc::gpu::NvidiaMmaEncodingAttr>()) {
    assert(false && "NvidiaMma");
  } else if (layout.isa<mlir::_tritoncc::gpu::AMDMfmaEncodingAttr>()) {
    return {1, 1};
  } else if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>()) {
    assert(false && "slice");
  } else {
    return getSizePerThread(layout);
  }
}

static llvm::SmallVector<unsigned> getUniqueContigPerThread(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentUniqueContigPerThread =
        getUniqueContigPerThread(parentLayout, parentShape);
    parentUniqueContigPerThread.erase(parentUniqueContigPerThread.begin()
        + sliceLayout.getDim());
    return parentUniqueContigPerThread;
  }
  // Base case
  auto rank = shape.size();
  llvm::SmallVector<unsigned> ret(rank);
  auto contigPerThread = getContigPerThread(layout);
  assert(contigPerThread.size() == rank && "Unexpected contigPerThread size");
  for (int d = 0; d < rank; ++d) {
    ret[d] = std::min<unsigned>(shape[d], contigPerThread[d]);
  }
  return ret;
}

static llvm::SmallVector<unsigned> getWarpsPerCTA(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getWarpsPerCTA();
  }

  llvm::report_fatal_error("getWarpsPerCTA not implemented");
  return llvm::SmallVector<unsigned>();
}

static llvm::SmallVector<unsigned>
getWarpsPerCTAWithUniqueData(mlir::Attribute layout, llvm::ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>()) {
    assert(false && "slice");
  }
  auto warpsPerCTA = getWarpsPerCTA(layout);
  assert(warpsPerCTA.size() == tensorShape.size() &&
      "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < warpsPerCTA.size(); ++i) {
    auto sizePerWarp =
        getSizePerThread(layout)[i] * getThreadsPerWarp(layout)[i];
    auto maxWarpsPerDim = tritoncc::ceil<unsigned>(tensorShape[i], sizePerWarp);
    warpsPerCTA[i] = std::min<unsigned>(warpsPerCTA[i], maxWarpsPerDim);
  }

  return warpsPerCTA;
}

static llvm::SmallVector<unsigned>
getThreadsPerWarpWithUniqueData(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::gpu::SliceEncodingAttr>()) {
    assert(false && "slice");
  }
  auto threadsPerWarp = getThreadsPerWarp(layout);
  assert(threadsPerWarp.size() == tensorShape.size() &&
    "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < threadsPerWarp.size(); ++i) {
    threadsPerWarp[i] = std::min<unsigned>(threadsPerWarp[i], tensorShape[i]);
  }

  return threadsPerWarp;
}

static unsigned getNumCTAs(mlir::Attribute layout) {
  return tritoncc::product<unsigned>(getCTAsPerCGA(layout));
}

static bool shouldUseDistSmem(mlir::Attribute srcLayout, mlir::Attribute dstLayout) {
  // numCTAs here means numCTAsPerCGA rather than numCTAs per grid.
  unsigned numCTAs = tritoncc::getNumCTAs(srcLayout);
  assert(numCTAs == tritoncc::getNumCTAs(dstLayout) &&
    "Invalid layout conversion: the numbers of CTAs of src and dst "
    "layouts are different");

  // case (1): Neber use dsmem when numCTAs == 1
  if (numCTAs == 1) {
    return false;
  }
  assert(false && "shouldUseDistSmem");
}

}

namespace mlir {
namespace _tritoncc {
namespace gpu {

template <class T>
llvm::SmallVector<T> SliceEncodingAttr::paddedShape(llvm::ArrayRef<T> shape) const {
  size_t rank = shape.size();
  unsigned dim = getDim();
  llvm::SmallVector<T> retShape(rank + 1);
  for (unsigned d = 0; d < rank + 1; ++d) {
    if (d < dim) {
      retShape[d] = shape[d];
    } else if (d == dim) {
      retShape[d] = 1;
    } else {
      retShape[d] = shape[d - 1];
    }
  }
  return retShape;
}

} } }
