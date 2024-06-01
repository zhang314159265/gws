#pragma once

namespace tritoncc {

mlir::Value getMask(mlir::Type valueTy, mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
  auto tensorTy = valueTy.dyn_cast<mlir::RankedTensorType>();
  mlir::Value mask = int_val(1, 1);  // true
  auto tid = tid_val();  // thread id
  auto clusterCTAId = getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    mlir::Attribute layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = tritoncc::getSizePerThread(layout);
    auto threadsPerWarp = tritoncc::getThreadsPerWarp(layout);
    auto warpsPerCTA = tritoncc::getWarpsPerCTA(layout);
    auto order = tritoncc::getOrder(layout);
    auto shapePerCTATile = tritoncc::getShapePerCTATile(layout, shape);
    mlir::Value warpSize = i32_val(32);
    mlir::Value laneId = urem(tid, warpSize);
    mlir::Value warpId = udiv(tid, warpSize);

    llvm::SmallVector<mlir::Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    llvm::SmallVector<mlir::Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    for (unsigned dim = 0; dim < rank; ++dim) {
      if (shape[dim] >= shapePerCTATile[dim]) {
        continue;
      }
      mlir::Value threadDim =
        macro_add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
            multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
          i32_val(shape[dim])));
    }
    if (tritoncc::getNumCTAs(layout) > 1) {
      assert("NumCTAs > 1");
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
  }
  return mask;
}

// Contains some helper functions for both load and store conversions
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getVectorSize(mlir::Value ptr) const;
  unsigned getMaskAlignment(mlir::Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  unsigned getContiguity(mlir::Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorTy) {
      return 1;
    }
    return axisAnalysisPass.getPtrContiguity(ptr);
  }
protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

unsigned LoadStoreConversionBase::getVectorSize(mlir::Value ptr) const {
  mlir::RankedTensorType tensorTy = ptr.getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorTy) {
    return 1;
  }
  // unit is number of items
  unsigned contiguity = getContiguity(ptr);
  unsigned pointeeBitWidth = tritoncc::getPointeeBitWidth(tensorTy);
  // The maximum vector size is 128 bits on NVIDIA GPUs.
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

struct LoadOpConversion : public ConvertOpToLLVMPattern<mlir::triton::LoadOp>,
                          public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<mlir::triton::LoadOp>::ConvertOpToLLVMPattern;

  LoadOpConversion(LLVMTypeConverter &converter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::LoadOp op,
      OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

mlir::LogicalResult LoadOpConversion::matchAndRewrite(
    mlir::triton::LoadOp op,
    OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();
  const mlir::LLVMTypeConverter *typeConverter = getTypeConverter();

  // original values
  mlir::Value ptr = op.getPtr();
  mlir::Value mask = op.getMask();
  mlir::Value other = op.getOther();

  // adaptor values
  assert(!tritoncc::isTensorPointerType(ptr.getType()) &&
    "Cannot convert load with a tensor pointer into LLVM; "
    "this case should be transformed to normal load before lowering");

  mlir::Value llPtr = adaptor.getPtr();
  mlir::Value llMask = adaptor.getMask();
  mlir::Value llOther = adaptor.getOther();

  // Determine the vectorization size
  mlir::Type valueTy = op.getResult().getType();
  mlir::Type valueElemTy =
    typeConverter->convertType(mlir::getElementTypeOrSelf(valueTy));
  unsigned vec = getVectorSize(ptr);
  unsigned numElems = tritoncc::getTotalElemsPerThread(ptr.getType());
  if (llMask) {
    vec = std::min<size_t>(vec, getMaskAlignment(mask));
  }

  // Get the LLVM values for pointers
  auto ptrElems = tritoncc::unpackLLElements(loc, llPtr, rewriter);
  assert(ptrElems.size() == numElems);

  // Get the LLVM values for mask
  llvm::SmallVector<mlir::Value> maskElems;
  if (llMask) {
    maskElems = tritoncc::unpackLLElements(loc, llMask, rewriter);
    assert(maskElems.size() == numElems);
  }

  // Get the LLVM values for `other`
  bool otherIsSplatConstInt = false;
  mlir::DenseElementsAttr constAttr;
  int64_t splatVal = 0;
  if (other && valueElemTy.isa<mlir::IntegerType>() &&
      matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
      constAttr.getElementType().isa<mlir::IntegerType>()) {
    otherIsSplatConstInt = true;
    splatVal = constAttr.getSplatValue<llvm::APInt>().getSExtValue();
  }
  llvm::SmallVector<mlir::Value> otherElems;
  if (other) {
    otherElems = tritoncc::unpackLLElements(loc, llOther, rewriter);
  }

  // vectorized iteration through all the pointer/mask/other elements
  const int valueElemNBits =
    std::max(8u, valueElemTy.getIntOrFloatBitWidth());
  const int numVecs = numElems / vec;

  llvm::SmallVector<mlir::Value> loadedVals;
  for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
    size_t in_off = 0;

    const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
    const size_t totalWidth = valueElemNBits * vec;
    const size_t width = std::min(totalWidth, maxWordWidth);
    const size_t nWords = std::max<size_t>(1, totalWidth / width);
    const size_t wordNElems = width / valueElemNBits;
    const size_t movWidth = width < 16 ? 16 : width;
    assert(wordNElems * nWords * numVecs == numElems);

    const bool hasL2EvictPolicy = false;

    PTXBuilder ptxBuilder;

    mlir::Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

    const std::string readConstraint =
        (width == 64) ? "l" : ((width == 32) ? "r" : "c");
    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

    // prepare asm operands
    auto *dstsOpr = ptxBuilder.newListOperand();
    for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
      auto *opr = ptxBuilder.newOperand(writeConstraint, /*init=*/true);
      dstsOpr->listAppend(opr);
    }

    auto *addrOpr =
        ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

    // Define the instruction opcode
    auto &ld = ptxBuilder.create<>("ld")
        ->o("volatile", op.getIsVolatile())
        .global()
        .o("ca", op.getCache() == triton::CacheModifier::CA)
        .o("cg", op.getCache() == triton::CacheModifier::CG)
        .o("L1::evict_first",
            op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
        .o("L1::evict_last",
            op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
        .o("L1::cache_hint", hasL2EvictPolicy)
        .v(nWords)
        .b(width);

    ld(dstsOpr, addrOpr).predicate(pred, "b");

    if (other) {
      for (size_t ii = 0; ii < nWords; ++ii) {
        // PTX doesn't support mov.u8, so we need to use mov.u16
        PTXInstr &mov =
            ptxBuilder.create<>("mov")->o("u" + std::to_string(movWidth));
        
        size_t size = width / valueElemNBits;

        auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
        mlir::Value v = macro_undef(vecTy);
        for (size_t s = 0; s < size; ++s) {
          mlir::Value falseVal = otherElems[vecStart + ii * size + s];
          mlir::Value sVal = createIndexAttrConstant(
            rewriter, loc, typeConverter->getIndexType(), s);
          v = insert_element(vecTy, v, falseVal, sVal);
        }
        v = bitcast(v, mlir::IntegerType::get(getContext(), width));

        PTXInstr::Operand *opr{};

        if (otherIsSplatConstInt) {
          for (size_t s = 0; s < 32; s += valueElemNBits) {
            splatVal |= splatVal << valueElemNBits;
          }
          opr = ptxBuilder.newConstantOperand(splatVal);
        } else {
          opr = ptxBuilder.newOperand(v, readConstraint);
        }

        mov(dstsOpr->listGet(ii), opr).predicateNot(pred, "b");
      }
    }

    // Create inline ASM signature
    llvm::SmallVector<mlir::Type> retTys(nWords, mlir::IntegerType::get(getContext(), width));
    mlir::Type retTy = retTys.size() > 1
        ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
        : retTys[0];

    mlir::Value ret = ptxBuilder.launch(rewriter, loc, retTy);

    // Extract and store return values
    llvm::SmallVector<mlir::Value> rets;
    for (unsigned int ii = 0; ii < nWords; ++ii) {
      mlir::Value curr;
      if (retTy.isa<LLVM::LLVMStructType>()) {
        curr = extract_val(IntegerType::get(getContext(), width), ret, ii);
      } else {
        curr = ret;
      }
      curr = bitcast(curr, LLVM::getFixedVectorType(valueElemTy,
          width / valueElemNBits));
      rets.push_back(curr);
    }

    int tmp = width / valueElemNBits;
    for (size_t ii = 0; ii < vec; ++ii) {
      mlir::Value vecIdx = createIndexAttrConstant(
          rewriter, loc, typeConverter->getIndexType(), ii % tmp);
      mlir::Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
      loadedVals.push_back(loaded);
    }
  }

  mlir::Type llvmResultStructTy = typeConverter->convertType(valueTy);
  mlir::Value resultStruct = tritoncc::packLLElements(
      loc, typeConverter, loadedVals,
      rewriter, llvmResultStructTy);
  rewriter.replaceOp(op, {resultStruct});
  return mlir::success();
}

struct StoreOpConversion : public ConvertOpToLLVMPattern<mlir::triton::StoreOp>,
    public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<mlir::triton::StoreOp>::ConvertOpToLLVMPattern;

  StoreOpConversion(LLVMTypeConverter &converter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit) : ConvertOpToLLVMPattern<mlir::triton::StoreOp>(converter, benefit), LoadStoreConversionBase(axisAnalysisPass) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::StoreOp op,
      OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

mlir::LogicalResult StoreOpConversion::matchAndRewrite(
    mlir::triton::StoreOp op,
    OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value ptr = op.getPtr();
  mlir::Value value = op.getValue();

  mlir::Value llPtr = adaptor.getPtr();
  mlir::Value llMask = adaptor.getMask();
  mlir::Value llValue = adaptor.getValue();

  auto loc = op->getLoc();
  mlir::MLIRContext *ctx = rewriter.getContext();

  mlir::Type valueTy = value.getType();
  mlir::Type valueElemTy = typeConverter->convertType(getElementTypeOrSelf(valueTy));

  unsigned vec = getVectorSize(ptr);
  unsigned elemsPerThread = tritoncc::getTotalElemsPerThread(ptr.getType());

  auto ptrElems = tritoncc::unpackLLElements(loc, llPtr, rewriter);
  auto valueElems = tritoncc::unpackLLElements(loc, llValue, rewriter);
  assert(ptrElems.size() == valueElems.size());


  // Determine the vectorization size
  llvm::SmallVector<mlir::Value> maskElems;
  if (llMask) {
    mlir::Value mask = op.getMask();
    maskElems = tritoncc::unpackLLElements(loc, llMask, rewriter);
    assert(valueElems.size() == maskElems.size());

    unsigned maskAlign = getMaskAlignment(mask);
    vec = std::min(vec, maskAlign);
  }

  mlir::Value mask = getMask(valueTy, rewriter, loc);
  const size_t dtsize =
    std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
  const size_t valueElemNBits = dtsize * 8;

  const int numVecs = elemsPerThread / vec;
  for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
    size_t in_off = 0;

    const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
    const size_t totalWidth = valueElemNBits * vec;
    const size_t width = std::min(totalWidth, maxWordWidth);
    const size_t nWords = std::max<size_t>(1, totalWidth / width);
    const size_t wordNElems = width / valueElemNBits;
    assert(wordNElems * nWords * numVecs == elemsPerThread);

    mlir::Type valArgTy = mlir::IntegerType::get(ctx, width);
    auto wordTy = vec_ty(valueElemTy, wordNElems);

    mlir::SmallVector<std::pair<mlir::Value, std::string>> asmArgs;
    for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
      mlir::Value llWord = macro_undef(wordTy);
      // Insert each value element to the composition
      for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
        const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
        assert(elemOffset < valueElems.size());
        mlir::Value elem = valueElems[elemOffset];
        if (elem.getType().isInteger(1)) {
          elem = sext(i8_ty, elem);
        }
        elem = bitcast(elem, valueElemTy);

        llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
      }
      llWord = bitcast(llWord, valArgTy);
      std::string constraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      asmArgs.emplace_back(llWord, constraint);
    }

    // Prepare the PTX inline asm
    PTXBuilder ptxBuilder;
    auto *asmArgList = ptxBuilder.newListOperand(asmArgs);

    mlir::Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;

    auto *asmAddr =
        ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

    auto &ptxStoreInstr =
        ptxBuilder.create<>("st")
          ->global()
          .o("wb", op.getCache() == mlir::triton::CacheModifier::WB)
          .o("cg", op.getCache() == mlir::triton::CacheModifier::CG)
          .o("cs", op.getCache() == mlir::triton::CacheModifier::CS)
          .o("wt", op.getCache() == mlir::triton::CacheModifier::WT)
          .o("L1::evict_first",
            op.getEvict() == mlir::triton::EvictionPolicy::EVICT_FIRST)
          .o("L1::evict_last",
            op.getEvict() == mlir::triton::EvictionPolicy::EVICT_LAST)
          .v(nWords)
          .b(width);

    ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

    mlir::Type boolTy = getTypeConverter()->convertType(rewriter.getIntegerType(1));
    auto asmReturnTy = void_ty(ctx);

    ptxBuilder.launch(rewriter, loc, asmReturnTy);
  }
  rewriter.eraseOp(op);
  return mlir::success();
}

void populateLoadStoreOpToLLVMPatterns(
  mlir::LLVMTypeConverter &typeConverter,
  mlir::RewritePatternSet &patterns,
  ModuleAxisInfoAnalysis &axisInfoAnalysis,
  mlir::PatternBenefit benefit
) {
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<StoreOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}

}
