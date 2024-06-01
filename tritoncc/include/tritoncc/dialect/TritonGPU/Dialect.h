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
#include "tritoncc/dialect/TritonGPU/Dialect.cpp.inc"

#include "tritoncc/dialect/TritonGPU/AttrInterfaces.h.inc"
#include "tritoncc/dialect/TritonGPU/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tritoncc/dialect/TritonGPU/AttrDefs.h.inc"

#define GET_OP_CLASSES
#include "tritoncc/dialect/TritonGPU/Ops.h.inc"

#include "tritoncc/util.h"

struct TritonGPUInferLayoutInterface
    : public mlir::triton::DialectInferLayoutInterface {
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
    resultEncoding = mlir::_tritoncc::SliceEncodingAttr::get(
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
    auto sliceEncoding = operandEncoding.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>();
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

void mlir::_tritoncc::TritonGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tritoncc/dialect/TritonGPU/AttrDefs.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "tritoncc/dialect/TritonGPU/Ops.cpp.inc"
  >();
  addInterfaces<TritonGPUInferLayoutInterface>();
}

#define GET_OP_CLASSES
#include "tritoncc/dialect/TritonGPU/Ops.cpp.inc"

namespace tritoncc {

unsigned getTotalElemsPerThread(mlir::Attribute layout, llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<mlir::_tritoncc::TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else {
    llvm::errs() << "layout " << layout << "\n";
    llvm::report_fatal_error("--> getTotalElemsPerThread not implemented");
    return 0;
  }
}

llvm::SmallVector<unsigned> getElemsPerThread(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<mlir::_tritoncc::TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getElemsPerThread not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

unsigned getTotalElemsPerThread(mlir::Type type) {
  if (type.isIntOrIndexOrFloat() || type.isa<mlir::triton::PointerType>()) {
    return 1;
  }
  auto tensorType = type.cast<mlir::RankedTensorType>();
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
      tensorType.getElementType());
}

// 1 element per thread
// order = reverse(arange(rank))
mlir::_tritoncc::BlockedEncodingAttr
getDefaultBlockedEncoding(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape,
    int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  return mlir::_tritoncc::BlockedEncodingAttr::get(
    context, shape, sizePerThread,
    order, numWarps, threadsPerWarp,
    numCTAs);
}

llvm::SmallVector<unsigned> getCTAsPerCGA(mlir::Attribute layout) {
  llvm::ArrayRef<unsigned> ref;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getCTAsPerCGA();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::AMDMfmaEncodingAttr>()) {
    return {1, 1};
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::SharedEncodingAttr>()) {
    ref = sharedLayout.getCTALayout().getCTAsPerCGA();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
  }
  return llvm::SmallVector<unsigned>(ref.begin(), ref.end());
}

llvm::SmallVector<unsigned> getCTASplitNum(mlir::Attribute layout) {
  llvm::SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getCTASplitNum();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::AMDMfmaEncodingAttr>()) {
    assert(false && "AMDMfmaEncodingAttr");
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::SharedEncodingAttr>()) {
    assert(false && "SharedEncodingAttr");
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

llvm::SmallVector<unsigned> getCTAOrder(mlir::Attribute layout) {
  llvm::SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    res = distributedLayout.getCTAOrder();
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::_tritoncc::AMDMfmaEncodingAttr>()) {
    return {0, 1};
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::SharedEncodingAttr>()) {
    res = llvm::SmallVector<unsigned>(sharedLayout.getCTALayout().getCTAOrder());
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

mlir::_tritoncc::CTALayoutAttr getCTALayout(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return mlir::_tritoncc::CTALayoutAttr::get(
        layout.getContext(), getCTAsPerCGA(distributedLayout),
        getCTASplitNum(distributedLayout),
        getCTAOrder(distributedLayout));
  } else if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::SharedEncodingAttr>()) {
    return sharedLayout.getCTALayout();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  }
  return {};
}

llvm::SmallVector<int64_t> getShapePerCTA(llvm::ArrayRef<unsigned> CTASplitNum,
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

llvm::SmallVector<int64_t> getShapePerCTA(mlir::Attribute layout, llvm::ArrayRef<int64_t> shape) {
  if (auto sharedLayout = layout.dyn_cast<mlir::_tritoncc::SharedEncodingAttr>()) {
    assert(false && "SharedEncodingAttr");
  }
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

llvm::SmallVector<int64_t> getShapePerCTA(mlir::Type type) {
  auto tensorType = type.cast<mlir::RankedTensorType>();
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

llvm::SmallVector<unsigned>
getShapePerCTATile(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> tensorShape = llvm::ArrayRef<int64_t>()) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getShapePerCTATile(tensorShape);
  } else {
    llvm::report_fatal_error("getShapePerCTATile not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

llvm::SmallVector<unsigned> getThreadsPerWarp(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getThreadsPerWarp();
  } else {
    llvm::report_fatal_error("getThreadsPerWarp not implemented");
    return llvm::SmallVector<unsigned>();
  }
}

unsigned getWarpSize(mlir::Attribute layout) {
  unsigned size = 1;
  auto threadsPerWarp = getThreadsPerWarp(layout);
  for (auto e : threadsPerWarp) {
    size *= e;
  }
  return size;
}

llvm::SmallVector<unsigned> getOrder(mlir::Attribute layout) {
  if (auto blockedLayout = layout.dyn_cast<mlir::_tritoncc::BlockedEncodingAttr>()) {
    return llvm::SmallVector<unsigned>(blockedLayout.getOrder().begin(),
        blockedLayout.getOrder().end());
  } else if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
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

llvm::SmallVector<unsigned> getSizePerThread(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getSizePerThread();
  } else {
    llvm::report_fatal_error("getSizePerThread not implemented");
    return {};
  }
}

llvm::SmallVector<unsigned> getContigPerThread(mlir::Attribute layout) {
  if (auto mmaLayout = layout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
    assert(false && "NvidiaMma");
  } else if (layout.isa<mlir::_tritoncc::AMDMfmaEncodingAttr>()) {
    return {1, 1};
  } else if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
    assert(false && "slice");
  } else {
    return getSizePerThread(layout);
  }
}

llvm::SmallVector<unsigned> getUniqueContigPerThread(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
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

llvm::SmallVector<unsigned> getWarpsPerCTA(mlir::Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<mlir::_tritoncc::DistributedEncodingTrait>()) {
    return distributedLayout.getWarpsPerCTA();
  }

  llvm::report_fatal_error("getWarpsPerCTA not implemented");
  return llvm::SmallVector<unsigned>();
}

llvm::SmallVector<unsigned>
getWarpsPerCTAWithUniqueData(mlir::Attribute layout, llvm::ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
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

llvm::SmallVector<unsigned>
getThreadsPerWarpWithUniqueData(mlir::Attribute layout,
    llvm::ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
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

unsigned getNumCTAs(mlir::Attribute layout) {
  return tritoncc::product<unsigned>(getCTAsPerCGA(layout));
}

bool shouldUseDistSmem(mlir::Attribute srcLayout, mlir::Attribute dstLayout) {
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

#define GET_ATTRDEF_CLASSES
#include "tritoncc/dialect/TritonGPU/AttrDefs.cpp.inc"

namespace mlir {
namespace _tritoncc {

template <typename T>
bool hasEncoding(mlir::Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
    auto encoding = tensorType.getEncoding();
    return encoding && encoding.isa<T>();
  }
  return false;
}

bool hasSharedEncoding(mlir::Value value) {
  return hasEncoding<mlir::_tritoncc::SharedEncodingAttr>(value);
}

struct CanonicalizeConvertFromConvert
    : public mlir::OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
      mlir::PatternRewriter &rewriter) const override {
    // Convert to the same layout is redundant.
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return mlir::success();
    }

    // We don't handle conversions to DotOperandEncodingAttr.  This is a
    // heuristic to accomodate fused attention.
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (dstType.getEncoding().isa<mlir::_tritoncc::DotOperandEncodingAttr>() &&
        srcType.getEncoding().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
      return mlir::failure();
    }

    // for hopper MMAv3
    if (dstType.getEncoding().isa<mlir::_tritoncc::SharedEncodingAttr>() && 
        srcType.getEncoding().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>() &&
        llvm::any_of(op.getResult().getUsers(),
            [](mlir::Operation *dot) { return llvm::isa<mlir::triton::DotOp>(dot); })) {
      return mlir::failure();
    }

    mlir::Operation *arg = op.getSrc().getDefiningOp();
    if (!arg) {
      return mlir::failure();
    }

    // cvt(reshape) -> reshape
    if (auto reshape = llvm::dyn_cast<mlir::triton::ReshapeOp>(arg)) {
      assert(false && "cvt(reshape)");
    }

    // cvt(histogram) -> histogram
    if (auto histogram = llvm::dyn_cast<mlir::triton::HistogramOp>(arg)) {
      assert(false && "cvt(histogram)");
    }

    // cvt(cat) -> cat
    if (auto cat = llvm::dyn_cast<mlir::triton::CatOp>(arg)) {
      assert(false && "cvt(cat)");
    }

    // cvt(alloc_tensor(x), type2) -> alloc_tensor(x, type2)
    if (auto alloc_tensor = llvm::dyn_cast<mlir::_tritoncc::AllocTensorOp>(arg)) {
      assert(false && "cvt(alloc_tensor)");
    }

    // cvt(insert_slice)
    if (auto insert_slice = llvm::dyn_cast<mlir::_tritoncc::InsertSliceAsyncOp>(arg)) {
      assert(false && "cvt(insert_slice)");
    }

    // cvt(extract_slice)
    if (auto extract_slice = llvm::dyn_cast<mlir::_tritoncc::ExtractSliceOp>(arg)) {
      assert(false && "cvt(extract_slice)");
    }

    // cvt(cvt)
    if (auto cvt = llvm::dyn_cast<mlir::_tritoncc::ConvertLayoutOp>(arg)) {
      if (cvt.getSrc().getDefiningOp() && !hasSharedEncoding(cvt.getSrc()) &&
          hasSharedEncoding(op.getSrc()) && !hasSharedEncoding(op.getResult())) {
        return mlir::failure();
      }

      if (hasSharedEncoding(op.getSrc()) && hasSharedEncoding(op.getResult())) {
        return mlir::failure();
      }

      auto srcType = op.getSrc().getType();
      auto srcShared =
          srcType.getEncoding().dyn_cast<mlir::_tritoncc::SharedEncodingAttr>();
      if (srcShared && srcShared.getVec() > 1) {
        return mlir::failure();
      }

      rewriter.replaceOpWithNewOp<mlir::_tritoncc::ConvertLayoutOp>(
          op, op->getResultTypes().front(), cvt.getSrc());
      return mlir::success();
    }

    // cvt(splat)
    if (auto splat = llvm::dyn_cast<mlir::triton::SplatOp>(arg)) {
      assert(false && "cvt(splat)");
    }

    // cvt(make_range)
    if (auto range = llvm::dyn_cast<mlir::triton::MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<mlir::triton::MakeRangeOp>(
          op, op->getResultTypes(), range.getStart(), range.getEnd());
      return mlir::success();
    }

    // cvt(constant)
    if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(arg)) {
      assert(false && "cvt(constant)");
    }

    return mlir::failure();
  }
};

// reshape(cvt) -> reshape
struct CanonicalizeConvertFromReshape
    : public mlir::OpRewritePattern<mlir::triton::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ReshapeOp op,
      mlir::PatternRewriter &rewriter) const override {
    assert(false && "matchAndRewrite");
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<mlir::triton::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::HistogramOp op,
      mlir::PatternRewriter &rewriter) const override {
    assert(false && "matchAndRewrite");
  }
};

void ConvertLayoutOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *context) {
  patterns.add<CanonicalizeConvertFromConvert>(context);
  patterns.add<CanonicalizeConvertFromReshape>(context);
  patterns.add<CanonicalizeConvertFromHistogram>(context);
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getWarpsPerCTA() const {
  assert(false && "DotOperandEncodingAttr::getWarpsPerCTA");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getCTASplitNum() const {
  assert(false && "getCTASplitNum");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getCTAOrder() const {
  assert(false && "getCTAOrder");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getCTAsPerCGA() const {
  assert(false && "getCTAsPerCGA");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getSizePerThread() const {
  assert(false && "getSizePerThread");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getThreadsPerWarp() const {
  assert(false && "getThreadsPerWarp");
}

llvm::SmallVector<unsigned> DotOperandEncodingAttr::getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(false && "getShapePerCTATile");
}

unsigned DotOperandEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  assert(false && "getTotalElemsPerThread");
}

llvm::SmallVector<unsigned>
DotOperandEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  assert(false && "getElemsPerThread");
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getWarpsPerCTA() const {
  assert(false && "SliceEncodingAttr::getWarpsPerCTA");
}

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

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

mlir::Attribute SliceEncodingAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  assert(false && "SliceEncodingAttr::parse");
}


llvm::SmallVector<unsigned> SliceEncodingAttr::getCTASplitNum() const {
  llvm::SmallVector<unsigned> res = tritoncc::getCTASplitNum(getParent());
  res.erase(res.begin() + getDim());
  return res;
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getCTAOrder() const {
  assert(false && "getCTAOrder");
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getCTAsPerCGA() const {
  auto parentCTAsPerCGA = tritoncc::getCTAsPerCGA(getParent());
  if (parentCTAsPerCGA[getDim()] == 1) {
    parentCTAsPerCGA.erase(parentCTAsPerCGA.begin() + getDim());
    return parentCTAsPerCGA;
  }
  llvm::report_fatal_error(
      "getCTAsPerCGA for SliceEncodingAttr is not well-defined");
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getSizePerThread() const {
  auto sizePerThread = tritoncc::getSizePerThread(getParent());
  sizePerThread.erase(sizePerThread.begin() + getDim());
  return sizePerThread;
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getThreadsPerWarp() const {
  assert(false && "getThreadsPerWarp");
}

llvm::SmallVector<unsigned> SliceEncodingAttr::getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape) const {
  llvm::SmallVector<unsigned> shape = tritoncc::getShapePerCTATile(getParent(), tensorShape);
  shape.erase(shape.begin() + getDim());
  return shape;
}

unsigned SliceEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  return tritoncc::product<unsigned>(getElemsPerThread(shape, eltTy));
}

llvm::SmallVector<unsigned>
SliceEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  auto parent = getParent();
  auto parentElemsPerThread =
      tritoncc::getElemsPerThread(parent, paddedShape(shape), eltTy);
  parentElemsPerThread.erase(parentElemsPerThread.begin() + getDim());
  return parentElemsPerThread;
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getWarpsPerCTA() const {
  return llvm::SmallVector<unsigned>(getWarpsPerCTA__());
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getThreadsPerWarp() const {
  return llvm::SmallVector<unsigned>(getThreadsPerWarp__());
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getSizePerThread() const {
  return llvm::SmallVector<unsigned>(getSizePerThread__());
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getCTASplitNum() const {
  return llvm::SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getCTAOrder() const {
  return llvm::SmallVector<unsigned>(getCTALayout().getCTAOrder());
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getCTAsPerCGA() const {
  return llvm::SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << llvm::ArrayRef(getSizePerThread()) << "]"
          << ", threadsPerWarp = [" << llvm::ArrayRef(getThreadsPerWarp()) << "]"
          << ", warpsPerCTA = [" << llvm::ArrayRef(getWarpsPerCTA()) << "]"
          << ", order = [" << getOrder() << "]";

  #if 0
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
      /*rank=*/getSizePerThread().size());
  #endif
  printer << "}>";
}

mlir::Attribute BlockedEncodingAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  assert(false && "BlockedEncodingAttr::parse");
}

llvm::SmallVector<unsigned> BlockedEncodingAttr::getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape) const {
  llvm::SmallVector<unsigned> shape;
  for (unsigned d = 0, n = getOrder().size(); d < n; ++d) {
    shape.push_back(getSizePerThread()[d] * getThreadsPerWarp()[d] *
        getWarpsPerCTA()[d]);
  }
  return shape;
}

llvm::SmallVector<unsigned>
BlockedEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerThread = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();
  auto threadsPerWarp = getThreadsPerWarp();
  auto shapePerCTA = tritoncc::getShapePerCTA(*this, shape);
  assert(rank == sizePerThread.size() &&
      "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  llvm::SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
    elemsPerThread[i] = tritoncc::ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
  }
  return elemsPerThread;
}

unsigned BlockedEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  return tritoncc::product<unsigned>(getElemsPerThread(shape, eltTy));
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getWarpsPerCTA() const {
  return llvm::SmallVector<unsigned>(getWarpsPerCTA__());
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getCTASplitNum() const {
  assert(false && "getCTASplitNum");
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAOrder() const {
  assert(false && "getCTAOrder");
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAsPerCGA() const {
  assert(false && "getCTAsPerCGA");
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getSizePerThread() const {
  assert(false && "getSizePerThread");
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getThreadsPerWarp() const {
  assert(false && "getThreadsPerWarp");
}

llvm::SmallVector<unsigned> AMDMfmaEncodingAttr::getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(false && "getShapePerCTATile");
}

unsigned AMDMfmaEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  assert(false && "getTotalElemsPerThread");
}

llvm::SmallVector<unsigned>
AMDMfmaEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  assert(false && "getElemsPerThread");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getWarpsPerCTA() const {
  return llvm::SmallVector<unsigned>(getWarpsPerCTA__());
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTASplitNum() const {
  assert(false && "getCTASplitNum");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAOrder() const {
  assert(false && "getCTAOrder");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAsPerCGA() const {
  assert(false && "getCTAsPerCGA");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getSizePerThread() const {
  assert(false && "getSizePerThread");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getThreadsPerWarp() const {
  assert(false && "getThreadsPerWarp");
}

llvm::SmallVector<unsigned> NvidiaMmaEncodingAttr::getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape) const {
  assert(false && "getShapePerCTATile");
}

unsigned NvidiaMmaEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  assert(false && "getTotalElemsPerThread");
}

llvm::SmallVector<unsigned>
NvidiaMmaEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  assert(false && "getElemsPerThread");
}

bool NvidiaMmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }
bool NvidiaMmaEncodingAttr::isHopper() const { assert(false); }
bool NvidiaMmaEncodingAttr::isAmpere() const { assert(false); }


unsigned SharedEncodingAttr::getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
  assert(false && "getTotalElemsPerThread");
}

llvm::SmallVector<unsigned>
SharedEncodingAttr::getElemsPerThread(llvm::ArrayRef<int64_t> shape,
    mlir::Type eltTy) const {
  assert(false && "getElemsPerThread");
}

} }
