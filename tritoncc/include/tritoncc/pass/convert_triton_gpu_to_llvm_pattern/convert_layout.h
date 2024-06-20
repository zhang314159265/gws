#pragma once

#include "llvm/Support/Debug.h"

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#endif

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "convert-layout-op-to-llvm"

namespace tritoncc {

llvm::SmallVector<unsigned> getRepShapeForCvtLayout(mlir::_tritoncc::ConvertLayoutOp op) {
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  mlir::Attribute srcLayout = srcTy.getEncoding();
  mlir::Attribute dstLayout = dstTy.getEncoding();

  if (tritoncc::shouldUseDistSmem(srcLayout, dstLayout)) {
    assert(false && "shouldUseDistSmem");
  }

  if (auto srcMmaLayout = srcLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
    assert(false && "mma layout");
  }

  assert(srcLayout && dstLayout && "Unexpected layout in getRepShape()");

  auto srcShapePerCTA = tritoncc::getShapePerCTA(srcTy);
  auto dstShapePerCTA = tritoncc::getShapePerCTA(dstTy);
  auto srcShapePerCTATile = tritoncc::getShapePerCTATile(srcLayout, srcTy.getShape());
  auto dstShapePerCTATile = tritoncc::getShapePerCTATile(dstLayout, dstTy.getShape());

  unsigned rank = dstTy.getRank();
  llvm::SmallVector<unsigned> repShape(rank);
  for (unsigned d = 0; d < rank; ++d) {
    repShape[d] =
      std::max(std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
               std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
  }
  return repShape;
}

std::pair<llvm::SmallVector<unsigned>, llvm::SmallVector<unsigned>>
getCvtOrder(mlir::Attribute srcLayout, mlir::Attribute dstLayout) {
  auto srcMmaLayout = srcLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>();
  auto srcDotLayout = srcLayout.dyn_cast<mlir::_tritoncc::DotOperandEncodingAttr>();
  auto dstMmaLayout = dstLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>();
  auto dstDotLayout = dstLayout.dyn_cast<mlir::_tritoncc::DotOperandEncodingAttr>();
  assert(!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) &&
    "mma -> mma layout conversion is only supported on Ampere");

  // mma or dot layout does not have an order, so the order depends on the
  // layout of the other operand.
  auto inOrd = (srcMmaLayout || srcDotLayout) ? tritoncc::getOrder(dstLayout)
                                              : tritoncc::getOrder(srcLayout);

  auto outOrd = (dstMmaLayout || dstDotLayout) ? tritoncc::getOrder(srcLayout)
                                               : tritoncc::getOrder(dstLayout);

  return {inOrd, outOrd};
}

llvm::SmallVector<unsigned>
getScratchConfigForCvtLayout(mlir::_tritoncc::ConvertLayoutOp op, unsigned &inVec, unsigned &outVec) {
  auto repShape = getRepShapeForCvtLayout(op);
  if (repShape.empty()) {
    return repShape;
  }
  auto rank = repShape.size();
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  mlir::Attribute srcLayout = srcTy.getEncoding();
  mlir::Attribute dstLayout = dstTy.getEncoding();

  auto [inOrd, outOrd] = getCvtOrder(srcLayout, dstLayout);
  unsigned srcContigPerThread =
      tritoncc::getUniqueContigPerThread(srcLayout, srcTy.getShape())[inOrd[0]];
  unsigned dstContigPerThread =
      tritoncc::getUniqueContigPerThread(dstLayout, dstTy.getShape())[outOrd[0]];
  unsigned innerDim = rank - 1;
  inVec = outOrd[0] != innerDim ? 1
          : inOrd[0] != innerDim ? 1 : srcContigPerThread;
  outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  // For conversions to MmaV1 (Nvidia V100), this inVec is hardcoded in the
  // codegen.
  if (auto mma = srcLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
    if (mma.getVersionMajor() == 1) {
      inVec = srcContigPerThread;
    }
  }

  if (rank <= 1) {
    return repShape;
  }
  // pad the last dimension
  unsigned paddedDim = rank - 1;
  if (auto dstBlockedLayout = dstLayout.dyn_cast<mlir::_tritoncc::BlockedEncodingAttr>()) {
    paddedDim = dstBlockedLayout.getOrder()[0];
  }
  unsigned pad = std::max(inVec, outVec);
  repShape[paddedDim] += pad;
  return repShape;
}

struct ConvertLayoutOpConversion
    : public mlir::ConvertOpToLLVMPattern<mlir::_tritoncc::ConvertLayoutOp> {
 public:
  using ConvertOpToLLVMPattern<mlir::_tritoncc::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      bool stNotRd, mlir::RankedTensorType type,
      llvm::ArrayRef<unsigned> numCTAsEachRep,
      llvm::ArrayRef<unsigned> multiDimRepId, unsigned vec,
      llvm::ArrayRef<unsigned> paddedRepShape,
      llvm::ArrayRef<unsigned> origRepShape,
      llvm::ArrayRef<unsigned> outOrd, llvm::SmallVector<mlir::Value> &vals,
      mlir::Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = tritoncc::getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    llvm::SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = tritoncc::getShapePerCTATile(layout);
    auto shapePerCTA = tritoncc::getShapePerCTA(layout, type.getShape());
    auto order = tritoncc::getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = elemTy.isa<mlir::triton::PointerType>();
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1) {
      elemTy = mlir::IntegerType::get(elemTy.getContext(), 8);
    } else if (isPtr) {
      elemTy = mlir::IntegerType::get(elemTy.getContext(), 64);
    }

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      llvm::SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId = 
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        llvm::SmallVector<mlir::Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                multiDimCTAInRepId, shapePerCTATile);
        llvm::SmallVector<mlir::Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile, shapePerCTA); 
        mlir::Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
            paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        mlir::Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
        if (stNotRd) {
          mlir::Value valVec = macro_undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1) {
              currVal = zext(llvmElemTy, currVal);
            } else if (isPtr) {
              currVal = ptrtoint(llvmElemTy, currVal);
            }
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          mlir::Value valVec = macro_load(vecTy, ptr);
          for (unsigned v = 0; v < vec; ++v) {
            mlir::Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1) {
              currVal = icmp_ne(currVal,
                  rewriter.create<mlir::LLVM::ConstantOp>(
                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            } else if (isPtr) {
              currVal = inttoptr(llvmElemTyOrig, currVal);
            }
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  bool isStMatrixCompatible(mlir::RankedTensorType tensorTy) const {
    auto mmaLayout = tensorTy.getEncoding().dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>();
    if (!mmaLayout || !mmaLayout.isHopper()) {
      return false;
    }
    assert(false && "isStMatrixCompatible");
  }

  mlir::LogicalResult
  lowerDistributedToDistributed(mlir::_tritoncc::ConvertLayoutOp op,
      OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    mlir::RankedTensorType srcTy = op.getSrc().getType();
    mlir::RankedTensorType dstTy = op.getType();
    mlir::Attribute srcLayout = srcTy.getEncoding();
    mlir::Attribute dstLayout = dstTy.getEncoding();

    if (tritoncc::shouldUseDistSmem(srcLayout, dstLayout)) {
      assert(false && "lower dist to dist with dist smem");
    }
    mlir::Value smemBase = tritoncc::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    llvm::SmallVector<unsigned> numReplicates(rank);
    llvm::SmallVector<unsigned> inNumCTAsEachRep(rank);
    llvm::SmallVector<unsigned> outNumCTAsEachRep(rank);
    llvm::SmallVector<unsigned> inNumCTAs(rank);
    llvm::SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = tritoncc::getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = tritoncc::getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = tritoncc::getShapePerCTA(srcLayout, shape);

    // For Volta, all the coords for a CTA are calculated.
    bool isSrcMmaV1{}, isDstMmaV1{};
    if (auto mmaLayout = srcLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
      isSrcMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = srcLayout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
      isSrcMmaV1 =
          sliceLayout.getParent().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>() &&
          sliceLayout.getParent().cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>().isVolta();
    }
    if (auto mmaLayout = dstLayout.dyn_cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
      isDstMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dstLayout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
      isDstMmaV1 =
          sliceLayout.getParent().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>() &&
          sliceLayout.getParent().cast<mlir::_tritoncc::NvidiaMmaEncodingAttr>().isVolta();
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA = std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA = std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = tritoncc::unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto origRepShape = getRepShapeForCvtLayout(op);
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);
    if (mlir::getElementTypeOrSelf(op.getType())
        .isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>()) {
      assert(false && "fp8");
    }

    unsigned outElems = tritoncc::getTotalElemsPerThread(dstTy);
    auto outOrd = tritoncc::getOrder(dstLayout);
    llvm::SmallVector<mlir::Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      if (srcLayout.isa<mlir::_tritoncc::BlockedEncodingAttr>() ||
          srcLayout.isa<mlir::_tritoncc::SliceEncodingAttr>() ||
          srcLayout.isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
        if (isSrcMmaV1) {
          assert(false && "isSrcMmaV1");
        } else if (isStMatrixCompatible(srcTy) && accumNumReplicates == 1 &&
            outOrd[0] == 1 && paddedRepShape[1] % 8 == 0) {
          assert(false && "stMatrix");
        } else {
          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
              inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
              origRepShape, outOrd, vals, smemBase);
        }
      } else {
        llvm::report_fatal_error(
          "ConvertLayout with input layout not implemented");
        return mlir::failure();
      }

      barrier();
      if (dstLayout.isa<mlir::_tritoncc::BlockedEncodingAttr>() ||
          dstLayout.isa<mlir::_tritoncc::SliceEncodingAttr>() ||
          dstLayout.isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
        if (isDstMmaV1) {
          assert(false && "isDstMmaV1");
        } else {
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
              outNumCTAsEachRep, multiDimRepId, outVec,
              paddedRepShape, origRepShape, outOrd, outVals,
              smemBase);
        }
      } else {
        llvm::report_fatal_error(
          "ConvertLayout with output layout not implemented");
        return mlir::failure();
      }
    }
    
    mlir::Value result = tritoncc::packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::_tritoncc::ConvertLayoutOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType srcTy = op.getSrc().getType();
    mlir::RankedTensorType dstTy = op.getType();
    mlir::Attribute srcLayout = srcTy.getEncoding();
    mlir::Attribute dstLayout = dstTy.getEncoding();

    LLVM_DEBUG({
      llvm::dbgs() << "ConvertLayoutOpConversion src type " << srcTy
        << ", dst type " << dstTy << "\n";
    });

    if (tritoncc::isaDistributedLayout(srcLayout) && dstLayout.isa<mlir::_tritoncc::SharedEncodingAttr>()) {
      assert(false && "distributed to shared");
    }
    if (srcLayout.isa<mlir::_tritoncc::SharedEncodingAttr>() &&
        dstLayout.isa<mlir::_tritoncc::DotOperandEncodingAttr>()) {
      assert(false && "shared to dot");
    }
    if (srcLayout.isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>()) {
      assert(false && "mma to mma");
    }
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }

    assert(false && "matchAndRewrite");
  }
 private:
  llvm::SmallVector<mlir::Value> getWrappedMultiDimOffset(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, llvm::ArrayRef<mlir::Value> multiDimOffset, llvm::ArrayRef<unsigned> shape, llvm::SmallVector<unsigned> shapePerCTATile, llvm::SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    llvm::SmallVector<mlir::Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d]) {
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      } else {
        multiDimOffsetWrapped[d] = multiDimOffset[d];
      }
    }
    return multiDimOffsetWrapped;
  }

  llvm::SmallVector<mlir::Value>
  getMultiDimOffset(mlir::Attribute layout, mlir::Location loc,
      mlir::ConversionPatternRewriter &rewriter, unsigned elemId,
      mlir::RankedTensorType type,
      llvm::ArrayRef<unsigned> multiDimCTAInRepId,
      llvm::ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<mlir::_tritoncc::BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem = 
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
      llvm::SmallVector<mlir::Value> multiDimOffset(rank);
      llvm::SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, tritoncc::getSizePerThread(layout), tritoncc::getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            macro_add(multiDimOffsetFirstElem[d],
              i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = tritoncc::getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = mlir::RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
      auto offsets = tritoncc::emitOffsetForLayout(layout, type);
      auto parentOffset = tritoncc::emitOffsetForLayout(parentEncoding, parentTy);
      llvm::SmallVector<int> idxs;
      for (llvm::SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
        parentEncoding, loc, rewriter, idxs[elemId], parentTy,
        sliceLayout.paddedShape(multiDimCTAInRepId),
        sliceLayout.paddedShape(shapePerCTATile));
      llvm::SmallVector<mlir::Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim) {
          continue;
        }
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }
};

#if USE_TRITON
void populateConvertLayoutOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  mlir::PatternBenefit benefit
) {
  mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
}
#else
void populateConvertLayoutOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  mlir::PatternBenefit benefit
) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
}
#endif
}

#undef DEBUG_TYPE
