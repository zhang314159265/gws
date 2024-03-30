#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "Utility.h"
#ifdef call
#undef call
#endif

#include "tritoncc/legacy/DebugListener.h"
#include "mlir/Transforms/DialectConversion.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"

#if 0
using mlir::triton::gpu::getTotalElemsPerThread;
using mlir::unpackLLElements;
using mlir::LLVM::linearize;
using mlir::LLVM::loadShared;
using mlir::triton::gpu::getOrder;
using mlir::LLVM::storeShared;
using mlir::emitOffsetForLayout;
using mlir::LLVM::shflSync;
#endif

namespace tritoncc {

#if 0
static SmallVector<SmallVector<unsigned>>
emitOffsetForBlockedLayout(const BlockedEncodingAttr &blockedLayout,
    RankedTensorType type) {
  auto shape = type.getShape();
  auto sizePerThread = blockedLayout.getSizePerThread(0;
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto waprsPerCTA = blockedLayout.getWaprsPerCTA();
  auto order = blockedLayout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(blockedLayout);
  auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);

  assert(false && "ABORT");
}
#endif

static SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type) {
  if (auto blockedLayout = layout.dyn_cast<mlir::triton::gpu::BlockedEncodingAttr>()) {
    return emitOffsetForBlockedLayout(blockedLayout, type);
  }
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    return emitOffsetForSliceLayout(sliceLayout, type);
  }
  assert(false && "NYI");
}

#if 1
class ReduceOpHelper {
 public:
  explicit ReduceOpHelper(triton::ReduceOp op) : op(op.getOperation()), axis(op.getAxis()) {
    RankedTensorType firstTy = op.getOperands()[0].getType().cast<RankedTensorType>();
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    std::cerr << "ReduceOpHelper: srcElementTypes cnt " << srcElementTypes.size() << std::endl;
    for (Type& ty : srcElementTypes) {
      ty.dump();
    }
  }

  triton::ReduceOp getOperation() { return op; }

  SmallVector<unsigned> getOrderWithAxisAtBeginning() {
    auto srcLayout = getSrcLayout();
    auto order = triton::gpu::getOrder(srcLayout);
    auto it = std::find(order.begin(), order.end(), axis);
    order.erase(it);
    order.insert(order.begin(), axis);
    return order;
  }

  #if 1 // copied from triton
  unsigned getInterWarpSizeWithUniqueData() {
    auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
    unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
    return std::min(srcReduceDimSize / sizeIntraWarps,
                    triton::gpu::getWarpsPerCTAWithUniqueData(
                        getSrcLayout(), getSrcShape())[axis]);
  }
  
  unsigned getIntraWarpSizeWithUniqueData() {
    auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
    unsigned elementPerThreads = triton::gpu::getUniqueContigPerThread(
        getSrcLayout(), getSrcShape())[axis];
    return std::min(srcReduceDimSize / elementPerThreads,
                    triton::gpu::getThreadsPerWarpWithUniqueData(
                        getSrcLayout(), getSrcShape())[axis]);
  }
  #endif

  // copied from triton
  SmallVector<unsigned> getScratchConfig() {
    SmallVector<unsigned> smemShape;
    // that case doesn't need inter-warp communication
    if (isWarpSynchronous())
      return {0, 0};
  
    smemShape = convertType<unsigned>(getSrcShape());
    smemShape[axis] = getInterWarpSizeWithUniqueData();
  
    return smemShape;
  }

  bool isReduceWithinCTA() {
    int axis = getAxis();
    auto srcLayout = getSrcLayout();
    auto CTASplitNum = mlir::triton::gpu::getCTASplitNum(srcLayout);
    assert(axis < CTASplitNum.size());
    return CTASplitNum[axis] == 1;
  }

  bool isSupportedLayout() {
    if (!isReduceWithinCTA()) {
      return false;
    }
    auto srcLayout = getSrcLayout();
    if (srcLayout.isa<triton::gpu::BlockedEncodingAttr>()) {
      return true;
    }
    assert(false && "cases not handled yet");
  }

  ArrayRef<int64_t> getSrcShape() { return srcShape; }

  bool isWarpSynchronous() {
    auto srcLayout = getSrcLayout();
    auto srcShape = getSrcShape();
    return triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis] == 1;
  }

  Attribute getSrcLayout() { return srcEncoding; }
  int getAxis() { return axis; }
 private:
  triton::ReduceOp op;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;
};
#endif

class ReduceOpConversion : public mlir::ConvertOpToLLVMPattern<triton::ReduceOp> {
 public:
  using SourceOp = triton::ReduceOp;
  ReduceOpConversion(mlir::LLVMTypeConverter &typeConverter) : mlir::ConvertOpToLLVMPattern<SourceOp>(typeConverter, 10) {
  }

  // TODO this can be dedep with ElementwiseOpConversion
  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = mlir::triton::gpu::getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto values = tritoncc::unpackLLElements(loc, operands[i], rewriter);
      assert(values.size() == srcValues.size());
      for (int j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  // copied literally from triton
  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
      SmallVector<Value> &acc, ValueRange cur, bool isFirst) const {
    if (isFirst) {
      acc = SmallVector<Value>(cur.begin(), cur.end());
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(combineOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp = dyn_cast<triton::ReduceReturnOp>(newReduce.getTerminator());

    llvm::SmallVector<Value> combineArgs(2 * acc.size());
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

  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>>& accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>>& indices,
      ConversionPatternRewriter& rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    SmallVector<SmallVector<unsigned>> offset = 
      tritoncc::emitOffsetForLayout(helper.getSrcLayout(), operandType);

    dump2dVector(offset);
    unsigned srcElems = mlir::triton::gpu::getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, helper.getSrcLayout(),
        operandType, true);
    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst) {
        indices[key] = srcIndices[i];
      }
    }
  }

  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
      SmallVector<Value>& acc, triton::ReduceOp op,
      unsigned numLaneToReduce, unsigned interleave) const {
    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (int i = 0; i < acc.size(); ++i) {
        shfl[i] = mlir::LLVM::shflSync(loc, rewriter, acc[i], N * interleave);
      }
      accumulate(rewriter, op.getCombineOp(), acc, shfl, false);
    }
  }

  void reduceWithinWarps(ReduceOpHelper& helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = 32;
    unsigned threadOffsetOnReductionAxis = 1;
    for (auto& it : accs) {
      SmallVector<Value>& acc = it.second;
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps, threadOffsetOnReductionAxis);
    }
  }

  #if 1 // copied literally from triton
  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = mlir::triton::gpu::getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            tritoncc::emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = tritoncc::packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }
  #endif

  // Return the pointee type of the shared memory pointer of operand i.
  Type getElementType(SourceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // copied from triton
  #if 1
  SmallVector<Value> getSmemBases(SourceOp op, unsigned elems, ConversionPatternRewriter& rewriter) const {
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
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 3), getElementType(op, indices[i - 1]),
          indexToBase[indices[i - 1]], i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }
  #endif

  #if 1 // copied from triton
  SmallVector<Value>
  getMultiDimWarpId(ReduceOpHelper &helper, Value &warpId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = mlir::triton::gpu::getOrder(srcLayout);
    SmallVector<Value> multiDimWarpId;

    // 2x2 warps with slice dim = 0, warpId = 2 ends up writing at the same
    // address as warpId = 0 since the warpsPerCTA is [1, 2], need to figure out
    // a way to properly delinearize warpId in the slice case
    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      auto parentLayout = sliceLayout.getParent();
      auto parentWarpsPerCTA = triton::gpu::getWarpsPerCTA(parentLayout);
      auto parentOrder = triton::gpu::getOrder(parentLayout);
      multiDimWarpId =
          delinearize(rewriter, loc, warpId, parentWarpsPerCTA, parentOrder);
      multiDimWarpId.erase(multiDimWarpId.begin() + sliceLayout.getDim());
    } else {
      auto warpsPerCTA =
          triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape);
      multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    }
    return multiDimWarpId;
  }


  #endif

  #if 1 // copied from triton
  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchConfig();

    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
    auto order = ::mlir::triton::gpu::getOrder(srcLayout);
    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    Value laneIdAxis = multiDimLaneId[axis];
    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    SmallVector<Value> multiDimWarpId =
        getMultiDimWarpId(helper, warpId, loc, rewriter);
    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          mlir::LLVM::linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                             smemBases[i], writeOffset);
        mlir::LLVM::NVIDIA::storeShared(rewriter, loc, writePtr, acc[i], laneZero);
      }
    }
  }
  #endif

  void sync(ConversionPatternRewriter &rewriter, Location loc, triton::ReduceOp op) const {
    rewriter.create<mlir::gpu::BarrierOp>(loc);
  }

  void accumulatePartialReductions(ReduceOpHelper &helper, SmallVector<Value> &smemBases, ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto srcLayout = helper.getSrcLayout();
    auto smemShape = helper.getScratchConfig();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value zero = i32_val(0);

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    unsigned numThreads =
      product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) *
      triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
            smemBases[i], readOffset);
        acc[i] = mlir::LLVM::NVIDIA::loadShared(rewriter, loc,readPtr, elemTy, threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
            smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
      
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        mlir::LLVM::NVIDIA::storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }
  }

  #if 1 // copied from trition
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = mlir::triton::gpu::getTotalElemsPerThread(resultTy);
        std::cerr << "loadReductionAndPackResult: resultElems " << resultElems << std::endl;
        auto resultIndices =
            emitIndices(loc, rewriter, resultLayout, resultTy, true);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          Value readOffset =
              mlir::LLVM::linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
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
  #endif

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    ReduceOpHelper helper(op);
    assert(helper.isSupportedLayout());
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      packResults(helper, accs, rewriter);
      return success();
    }

    auto smemShape = helper.getScratchConfig();

    SmallVector<Value> smemBases = getSmemBases(op, product<unsigned>(smemShape), rewriter);
    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    accumulatePartialReductions(helper, smemBases, rewriter);

    sync(rewriter, loc, op);

    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

    #if 0 // debug
    std::cerr << "smemShape ";
    for (auto& v : smemShape) {
      std::cerr << v << " ";
    }
    std::cerr << std::endl;
    std::cerr << "sizeInterWarps: " << helper.getInterWarpSizeWithUniqueData() << std::endl;
    auto srcLayout = helper.getSrcLayout();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) *
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    std::cerr << "triton::gpu::TritonGPUDialect::getThreadsPerWarp " << triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod) << std::endl;
    std::cerr << "numThreads " << numThreads << std::endl;
    #endif

    #if 0
    op.getOperation()->getParentOfType<ModuleOp>().dump(); // TODO

    std::cerr << "ReduceOp: ";
    op.dump();
    #endif

    return success();
  }
 private:
};

}
