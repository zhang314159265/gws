#pragma once

namespace tritoncc {

class ReduceOpHelper {
 public:
  explicit ReduceOpHelper(mlir::triton::ReduceOp op) {
    this->op = mlir::triton::ReduceOp(op.getOperation());
    this->axis = op.getAxis();
    mlir::RankedTensorType firstTy = op.getOperands()[0].getType().cast<mlir::RankedTensorType>();
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();
  }

  mlir::triton::ReduceOp getOperation() { return op; }

  llvm::SmallVector<unsigned> getOrderWithAxisAtBeginning() {
    auto srcLayout = getSrcLayout();
    auto order = mlir::triton::gpu::getOrder(srcLayout);
    auto it = std::find(order.begin(), order.end(), axis);
    order.erase(it);
    order.insert(order.begin(), axis);
    return order;
  }

  unsigned getInterWarpSizeWithUniqueData() {
    auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
    unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
    return std::min(srcReduceDimSize / sizeIntraWarps,
        mlir::triton::gpu::getWarpsPerCTAWithUniqueData(
          getSrcLayout(), getSrcShape())[axis]);
  }

  llvm::SmallVector<unsigned> getScratchConfig() {
    llvm::SmallVector<unsigned> smemShape;
    if (isWarpSynchronous()) {
      return {0, 0};
    }
    smemShape = convertType<unsigned>(getSrcShape());
    smemShape[axis] = getInterWarpSizeWithUniqueData();

    return smemShape;
  }

  unsigned getIntraWarpSizeWithUniqueData() {
    auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
    unsigned elementPerThreads = mlir::triton::gpu::getUniqueContigPerThread(
        getSrcLayout(), getSrcShape())[axis];
    return std::min(srcReduceDimSize / elementPerThreads,
        mlir::triton::gpu::getThreadsPerWarpWithUniqueData(
            getSrcLayout(), getSrcShape())[axis]);
  }

  bool isSupportedLayout() {
    if (!isReduceWithinCTA()) {
      return false;
    }
    auto srcLayout = getSrcLayout();
    if (srcLayout.isa<mlir::triton::gpu::BlockedEncodingAttr>()) {
      return true;
    }
    assert(false && "isSupportedLayout");
  }

  bool isReduceWithinCTA() {
    int axis = getAxis();
    auto srcLayout = getSrcLayout();
    auto CTASplitNum = mlir::triton::gpu::getCTASplitNum(srcLayout);
    assert(axis < CTASplitNum.size());
    return CTASplitNum[axis] == 1;
  }

  llvm::ArrayRef<int64_t> getSrcShape() { return srcShape; }

  bool isWarpSynchronous() {
    auto srcLayout = getSrcLayout();
    auto srcShape = getSrcShape();
    return mlir::triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis] == 1;
  }

  mlir::Attribute getSrcLayout() { return srcEncoding; }
  int getAxis() { return axis; }
 private:
  mlir::triton::ReduceOp op;
  llvm::ArrayRef<int64_t> srcShape;
  mlir::Attribute srcEncoding;
  llvm::SmallVector<mlir::Type> srcElementTypes;
  int axis;
};

// TODO move to util.h
llvm::SmallVector<llvm::SmallVector<unsigned>>
emitOffsetForLayout(mlir::Attribute layout, mlir::RankedTensorType type) {
  if (auto blockedLayout = layout.dyn_cast<mlir::triton::gpu::BlockedEncodingAttr>()) {
    return emitOffsetForBlockedLayout(blockedLayout, type);
  }
  if (auto sliceLayout = layout.dyn_cast<mlir::triton::gpu::SliceEncodingAttr>()) {
    return emitOffsetForSliceLayout(sliceLayout, type);
  }
  assert(false && "emitOffsetForLayout");
}

class ReduceOpConversion : public mlir::ConvertOpToLLVMPattern<mlir::triton::ReduceOp> {
 public:
  explicit ReduceOpConversion(mlir::LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::ReduceOp>(typeConverter, 10) {
  }

  llvm::SmallVector<mlir::Value> getMultiDimWarpId(
    ReduceOpHelper &helper, mlir::Value &warpId, mlir::Location &loc,
    mlir::ConversionPatternRewriter &rewriter) const;

  void loadReductionAndPackResult(
      ReduceOpHelper &helper,
      llvm::SmallVector<unsigned> smemShape,
      llvm::SmallVector<mlir::Value> &smemBases,
      mlir::ConversionPatternRewriter &rewriter) const;

  void accumulatePartialReductions(ReduceOpHelper &helper, llvm::SmallVector<mlir::Value> &smemBases, mlir::ConversionPatternRewriter &rewriter) const;

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
      std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &indices,
      llvm::SmallVector<mlir::Value> &smemBases,
      mlir::ConversionPatternRewriter &rewriter) const;

  llvm::SmallVector<mlir::Value> getSmemBases(
      mlir::triton::ReduceOp op, unsigned elems, mlir::ConversionPatternRewriter &rewriter) const;

  void warpReduce(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      llvm::SmallVector<mlir::Value> &acc, mlir::triton::ReduceOp op,
      unsigned numLaneToReduce, unsigned interleave) const;

  void reduceWithinWarps(ReduceOpHelper &helper,
      std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
      mlir::ConversionPatternRewriter &rewriter) const;

  void accumulate(mlir::ConversionPatternRewriter &rewriter,
      mlir::Region &combineOp, llvm::SmallVector<mlir::Value> &acc,
      mlir::ValueRange cur, bool isFirst) const;

  void reduceWithinThreads(
      ReduceOpHelper &helper, llvm::SmallVector<llvm::SmallVector<mlir::Value>> &srcValues,
      std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
      std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &indices,
      mlir::ConversionPatternRewriter &rewriter) const;

  llvm::SmallVector<llvm::SmallVector<mlir::Value>>
  unpackInputs(
      mlir::Location loc, mlir::triton::ReduceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::ReduceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Type getElementType(mlir::triton::ReduceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  void sync(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::triton::ReduceOp op) const {
    rewriter.create<mlir::gpu::BarrierOp>(loc);
  }
};

llvm::SmallVector<mlir::Value> ReduceOpConversion::getMultiDimWarpId(
  ReduceOpHelper &helper, mlir::Value &warpId, mlir::Location &loc,
  mlir::ConversionPatternRewriter &rewriter
) const {
  auto srcLayout = helper.getSrcLayout();
  auto srcShape = helper.getSrcShape();
  auto order = mlir::triton::gpu::getOrder(srcLayout);
  llvm::SmallVector<mlir::Value> multiDimWarpId;
  if (auto sliceLayout = srcLayout.dyn_cast<mlir::triton::gpu::SliceEncodingAttr>()) {
    assert(false && "SliceEncodingAttr");
  } else {
    auto warpsPerCTA =
        mlir::triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape);
    multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
  }
  return multiDimWarpId;
}

void ReduceOpConversion::loadReductionAndPackResult(
    ReduceOpHelper &helper,
    llvm::SmallVector<unsigned> smemShape,
    llvm::SmallVector<mlir::Value> &smemBases,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::triton::ReduceOp op = helper.getOperation();
  mlir::Location loc = op.getLoc();
  auto srcLayout = helper.getSrcLayout();
  auto axis = op.getAxis();
  auto smemOrder = helper.getOrderWithAxisAtBeginning();
  llvm::SmallVector<mlir::Value> results(op.getNumOperands());
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto elemTy = getElementType(op, i);
    if (auto resultTy =
        op.getResult()[i].getType().dyn_cast<mlir::RankedTensorType>()) {
      // nd-tensor where n >= 1
      auto resultLayout = resultTy.getEncoding().cast<mlir::triton::gpu::SliceEncodingAttr>();
      unsigned resultElems = mlir::triton::gpu::getTotalElemsPerThread(resultTy);
      auto resultIndices =
          emitIndices(loc, rewriter, resultLayout, resultTy, true);
      assert(resultIndices.size() == resultElems);

      llvm::SmallVector<mlir::Value> resultVals(resultElems);
      for (size_t j = 0; j < resultElems; ++j) {
        llvm::SmallVector<mlir::Value> readIdx = resultIndices[j];
        readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
        mlir::Value readOffset =
            mlir::LLVM::linearize(rewriter, loc, readIdx, smemShape, smemOrder);
        mlir::Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
            smemBases[i], readOffset);
        resultVals[j] = load(elemTy, readPtr);
      }

      results[i] = tritoncc::packLLElements(loc, getTypeConverter(), resultVals,
          rewriter, resultTy);
    } else {
      // 0d-tensor -> scalar
      results[i] = load(elemTy, smemBases[i]);
    }
  }
  rewriter.replaceOp(op, results);
}

void ReduceOpConversion::accumulatePartialReductions(ReduceOpHelper &helper, llvm::SmallVector<mlir::Value> &smemBases, mlir::ConversionPatternRewriter &rewriter) const {
  mlir::triton::ReduceOp op = helper.getOperation();
  auto srcLayout = helper.getSrcLayout();
  auto smemShape = helper.getScratchConfig();
  unsigned elems = product<unsigned>(smemShape);
  unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
  mlir::Location loc = op.getLoc();

  mlir::Value threadId = getThreadId(rewriter, loc);
  mlir::Value warpSize = i32_val(32);
  mlir::Value laneId = urem(threadId, warpSize);
  mlir::Value zero = i32_val(0);

  auto mod = op.getOperation()->getParentOfType<mlir::ModuleOp>();
  unsigned numThreads =
      product<unsigned>(mlir::triton::gpu::getWarpsPerCTA(srcLayout)) *
      mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
  mlir::Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
  mlir::Value readOffset = threadId;
  for (unsigned round = 0; round < elemsPerThread; ++round) {
    llvm::SmallVector<mlir::Value> acc(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      mlir::Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
          smemBases[i], readOffset);
      acc[i] = mlir::LLVM::NVIDIA::loadShared(rewriter, loc, readPtr, elemTy, threadIsNeeded);
    }
    warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */);
    // only the first thread in each sizeInterWarps is writing
    mlir::Value writeOffset = readOffset;
    llvm::SmallVector<mlir::Value> writePtrs(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
          smemBases[i], writeOffset);
    }

    mlir::Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
    mlir::Value laneIdModSizeInterWarpsIsZero =
        icmp_eq(laneIdModSizeInterWarps, zero);
    mlir::Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      mlir::LLVM::NVIDIA::storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
    }

    if (round != elemsPerThread - 1) {
      readOffset = add(readOffset, i32_val(numThreads));
    }
  }
}

void ReduceOpConversion::storeWarpReduceToSharedMemory(
    ReduceOpHelper &helper,
    std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
    std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &indices,
    llvm::SmallVector<mlir::Value> &smemBases,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::triton::ReduceOp op = helper.getOperation();
  mlir::Location loc = op.getLoc();
  mlir::Value threadId = getThreadId(rewriter, loc);
  mlir::Value warpSize = i32_val(32);
  mlir::Value warpId = udiv(threadId, warpSize);
  mlir::Value laneId = urem(threadId, warpSize);
  auto srcLayout = helper.getSrcLayout();
  auto srcShape = helper.getSrcShape();
  unsigned axis = op.getAxis();
  auto smemShape = helper.getScratchConfig();

  auto threadsPerWarp =
      mlir::triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
  auto order = mlir::triton::gpu::getOrder(srcLayout);
  llvm::SmallVector<mlir::Value> multiDimLaneId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  mlir::Value laneIdAxis = multiDimLaneId[axis];
  mlir::Value zero = i32_val(0);
  mlir::Value laneZero = icmp_eq(laneIdAxis, zero);

  llvm::SmallVector<mlir::Value> multiDimWarpId
      = getMultiDimWarpId(helper, warpId, loc, rewriter);
  mlir::Value warpIdAxis = multiDimWarpId[axis];

  auto smemOrder = helper.getOrderWithAxisAtBeginning();
  for (auto it : accs) {
    const llvm::SmallVector<unsigned> &key = it.first;
    llvm::SmallVector<mlir::Value> &acc = it.second;

    llvm::SmallVector<mlir::Value> writeIdx = indices[key];
    writeIdx[axis] = warpIdAxis;
    mlir::Value writeOffset =
        mlir::LLVM::linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      mlir::Value writePtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
          smemBases[i], writeOffset);
      mlir::LLVM::NVIDIA::storeShared(rewriter, loc, writePtr, acc[i], laneZero);
    }
  }
}

llvm::SmallVector<mlir::Value> ReduceOpConversion::getSmemBases(
    mlir::triton::ReduceOp op, unsigned elems, mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  // indices will store the index of the op operands in descending order
  // of their bitwidths
  std::vector<unsigned> indices(op.getNumOperands());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
    return op.getElementTypes()[i].getIntOrFloatBitWidth() >
        op.getElementTypes()[j].getIntOrFloatBitWidth();
  });
  // Assign base index to each operand in their order in indices
  std::map<unsigned, mlir::Value> indexToBase;
  indexToBase[indices[0]] =
      mlir::LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());

  for (unsigned i = 1; i < op.getNumOperands(); ++i) {
    indexToBase[indices[i]] = gep(
      ptr_ty(rewriter.getContext(), 3),
      getElementType(op, indices[i - 1]),
      indexToBase[indices[i - 1]],
      i32_val(elems)
    );
  }
  // smemBases[k] is the base pointer for the k-th operand
  llvm::SmallVector<mlir::Value> smemBases(op.getNumOperands());
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    smemBases[i] = indexToBase[i];
  }
  return smemBases;
}

void ReduceOpConversion::warpReduce(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::SmallVector<mlir::Value> &acc, mlir::triton::ReduceOp op,
    unsigned numLaneToReduce, unsigned interleave) const {
  for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
    llvm::SmallVector<mlir::Value> shfl(acc.size());
    for (int i = 0; i < acc.size(); ++i) {
      shfl[i] = mlir::LLVM::shflSync(loc, rewriter, acc[i], N * interleave);
    }
    accumulate(rewriter, op.getCombineOp(), acc, shfl, false);
  }
}

void ReduceOpConversion::reduceWithinWarps(ReduceOpHelper &helper,
    std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::triton::ReduceOp op = helper.getOperation();
  unsigned sizeIntraWarps = 32;
  unsigned threadOffsetOnReductionAxis = 1;
  for (auto &it : accs) {
    llvm::SmallVector<mlir::Value> &acc = it.second;
    warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps, threadOffsetOnReductionAxis);
  }
}

void ReduceOpConversion::accumulate(mlir::ConversionPatternRewriter &rewriter,
    mlir::Region &combineOp, llvm::SmallVector<mlir::Value> &acc,
    mlir::ValueRange cur, bool isFirst) const {
  if (isFirst) {
    acc = llvm::SmallVector<mlir::Value>(cur.begin(), cur.end());
    return;
  }

  // Create a new copy of the reduce block, and inline it
  mlir::Block *currentBlock = rewriter.getBlock();
  mlir::Region &parent = *currentBlock->getParent();
  rewriter.cloneRegionBefore(combineOp, &parent.front());
  auto &newReduce = parent.front();
  auto returnOp = llvm::dyn_cast<mlir::triton::ReduceReturnOp>(newReduce.getTerminator());

  llvm::SmallVector<mlir::Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
      combineArgs);
  auto results = returnOp.getResult();
  for (unsigned i = 0; i < acc.size(); ++i) {
    acc[i] = results[i];
  }

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
}

void ReduceOpConversion::reduceWithinThreads(
    ReduceOpHelper &helper, llvm::SmallVector<llvm::SmallVector<mlir::Value>> &srcValues,
    std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &accs,
    std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> &indices,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::triton::ReduceOp op = helper.getOperation();
  mlir::RankedTensorType operandType = op.getInputTypes()[0];
  llvm::SmallVector<llvm::SmallVector<unsigned>> offset =
      tritoncc::emitOffsetForLayout(helper.getSrcLayout(), operandType);

  unsigned srcElems = mlir::triton::gpu::getTotalElemsPerThread(operandType);
  auto *combineOp = &op.getCombineOp();
  auto srcIndices = emitIndices(op.getLoc(), rewriter, helper.getSrcLayout(),
      operandType, true);

  // reduce within threads
  for (unsigned i = 0; i < srcElems; ++i) {
    llvm::SmallVector<unsigned> key = offset[i];
    key[op.getAxis()] = 0;
    bool isFirst = accs.find(key) == accs.end();
    accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
    if (isFirst) {
      indices[key] = srcIndices[i];
    }
  }
}

llvm::SmallVector<llvm::SmallVector<mlir::Value>>
ReduceOpConversion::unpackInputs(
    mlir::Location loc, mlir::triton::ReduceOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto types = op.getInputTypes();
  auto operands = adaptor.getOperands();
  unsigned srcElems = mlir::triton::gpu::getTotalElemsPerThread(types[0]);
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> srcValues(srcElems);
  for (int i = 0; i < op.getNumOperands(); ++i) {
    auto values = tritoncc::unpackLLElements(loc, operands[i], rewriter);
    assert(values.size() == srcValues.size());
    for (int j = 0; j < srcValues.size(); ++j) {
      srcValues[j].push_back(values[j]);
    }
  }
  return srcValues;
}

mlir::LogicalResult ReduceOpConversion::matchAndRewrite(
    mlir::triton::ReduceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {
  ReduceOpHelper helper(op);
  assert(helper.isSupportedLayout());
  mlir::Location loc = op->getLoc();

  auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
  std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> accs;
  std::map<llvm::SmallVector<unsigned>, llvm::SmallVector<mlir::Value>> indices;
  reduceWithinThreads(helper, srcValues, accs, indices, rewriter);
  reduceWithinWarps(helper, accs, rewriter);

  if (helper.isWarpSynchronous()) {
    assert(false && "single warp");
    return mlir::success();
  }

  auto smemShape = helper.getScratchConfig();

  llvm::SmallVector<mlir::Value> smemBases = getSmemBases(
    op, product<unsigned>(smemShape), rewriter);
  storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

  sync(rewriter, loc, op);

  accumulatePartialReductions(helper, smemBases, rewriter);

  sync(rewriter, loc, op);

  loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);
  return mlir::success();
}

}