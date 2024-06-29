#include "mlir/IR/BuiltinOps.h"

#include "tritoncc/dialect/Triton/Dialect.h"
#include "tritoncc/dialect/TritonGPU/Dialect.h"
#include "tritoncc/dialect/TritonGPU/Dialect.cpp.inc"
#include "tritoncc/dialect/TritonGPU/AttrInterfaces.cpp.inc"

void mlir::_tritoncc::gpu::TritonGPUDialect::initialize() {
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

#define GET_ATTRDEF_CLASSES
#include "tritoncc/dialect/TritonGPU/AttrDefs.cpp.inc"

namespace mlir {
namespace _tritoncc {
namespace gpu {

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
  return hasEncoding<mlir::_tritoncc::gpu::SharedEncodingAttr>(value);
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
    if (dstType.getEncoding().isa<mlir::_tritoncc::gpu::DotOperandEncodingAttr>() &&
        srcType.getEncoding().isa<mlir::_tritoncc::gpu::NvidiaMmaEncodingAttr>()) {
      return mlir::failure();
    }

    // for hopper MMAv3
    if (dstType.getEncoding().isa<mlir::_tritoncc::gpu::SharedEncodingAttr>() && 
        srcType.getEncoding().isa<mlir::_tritoncc::gpu::NvidiaMmaEncodingAttr>() &&
        llvm::any_of(op.getResult().getUsers(),
            [](mlir::Operation *dot) { return llvm::isa<mlir::_tritoncc::DotOp>(dot); })) {
      return mlir::failure();
    }

    mlir::Operation *arg = op.getSrc().getDefiningOp();
    if (!arg) {
      return mlir::failure();
    }

    // cvt(reshape) -> reshape
    if (auto reshape = llvm::dyn_cast<mlir::_tritoncc::ReshapeOp>(arg)) {
      assert(false && "cvt(reshape)");
    }

    // cvt(histogram) -> histogram
    if (auto histogram = llvm::dyn_cast<mlir::_tritoncc::HistogramOp>(arg)) {
      assert(false && "cvt(histogram)");
    }

    // cvt(cat) -> cat
    if (auto cat = llvm::dyn_cast<mlir::_tritoncc::CatOp>(arg)) {
      assert(false && "cvt(cat)");
    }

    // cvt(alloc_tensor(x), type2) -> alloc_tensor(x, type2)
    if (auto alloc_tensor = llvm::dyn_cast<mlir::_tritoncc::gpu::AllocTensorOp>(arg)) {
      assert(false && "cvt(alloc_tensor)");
    }

    // cvt(insert_slice)
    if (auto insert_slice = llvm::dyn_cast<mlir::_tritoncc::gpu::InsertSliceAsyncOp>(arg)) {
      assert(false && "cvt(insert_slice)");
    }

    // cvt(extract_slice)
    if (auto extract_slice = llvm::dyn_cast<mlir::_tritoncc::gpu::ExtractSliceOp>(arg)) {
      assert(false && "cvt(extract_slice)");
    }

    // cvt(cvt)
    if (auto cvt = llvm::dyn_cast<mlir::_tritoncc::gpu::ConvertLayoutOp>(arg)) {
      if (cvt.getSrc().getDefiningOp() && !hasSharedEncoding(cvt.getSrc()) &&
          hasSharedEncoding(op.getSrc()) && !hasSharedEncoding(op.getResult())) {
        return mlir::failure();
      }

      if (hasSharedEncoding(op.getSrc()) && hasSharedEncoding(op.getResult())) {
        return mlir::failure();
      }

      auto srcType = op.getSrc().getType();
      auto srcShared =
          srcType.getEncoding().dyn_cast<mlir::_tritoncc::gpu::SharedEncodingAttr>();
      if (srcShared && srcShared.getVec() > 1) {
        return mlir::failure();
      }

      rewriter.replaceOpWithNewOp<mlir::_tritoncc::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), cvt.getSrc());
      return mlir::success();
    }

    // cvt(splat)
    if (auto splat = llvm::dyn_cast<mlir::_tritoncc::SplatOp>(arg)) {
      assert(false && "cvt(splat)");
    }

    // cvt(make_range)
    if (auto range = llvm::dyn_cast<mlir::_tritoncc::MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<mlir::_tritoncc::MakeRangeOp>(
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
    : public mlir::OpRewritePattern<mlir::_tritoncc::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::ReshapeOp op,
      mlir::PatternRewriter &rewriter) const override {
    assert(false && "matchAndRewrite");
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<mlir::_tritoncc::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::HistogramOp op,
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

} } }
