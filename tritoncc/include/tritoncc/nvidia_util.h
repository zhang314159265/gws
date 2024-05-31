#pragma once

#include <set>
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#else

#include "tritoncc/dialect/NVGPU/Dialect.h"

#include "tritoncc/dialect/TritonNvidiaGPU/Dialect.h"
#include "tritoncc/PTXAsmFormat.h"

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define gep(...) rewriter.create<mlir::LLVM::GEPOp>(loc, __VA_ARGS__)
#define macro_load(...) rewriter.create<mlir::LLVM::LoadOp>(loc, __VA_ARGS__)
#define store(val, ptr) rewriter.create<mlir::LLVM::StoreOp>(loc, val, ptr)
#define ptr_ty(...) mlir::LLVM::LLVMPointerType::get(__VA_ARGS__)
#define urem(...) rewriter.create<mlir::LLVM::URemOp>(loc, __VA_ARGS__)
#define macro_add(...) rewriter.create<mlir::LLVM::AddOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define bitcast(val, type) rewriter.create<mlir::LLVM::BitcastOp>(loc, type, val)
#define zext(...) rewriter.create<mlir::LLVM::ZExtOp>(loc, __VA_ARGS__)
#define extract_element(...) rewriter.create<mlir::LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define insert_element(...) rewriter.create<mlir::LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<mlir::LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define inttoptr(...) rewriter.create<mlir::LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<mlir::LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define icmp_ne(...) rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, __VA_ARGS__)
#define icmp_eq(...) rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, __VA_ARGS__)
#define icmp_slt(...) rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, __VA_ARGS__)

#define macro_undef(...) rewriter.create<mlir::LLVM::UndefOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<mlir::LLVM::AndOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<mlir::LLVM::SExtOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<mlir::LLVM::MulOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<mlir::LLVM::UDivOp>(loc, __VA_ARGS__)
#define trunc(...) rewriter.create<mlir::LLVM::TruncOp>(loc, __VA_ARGS__)

// Types
#define int_ty(width) rewriter.getIntegerType(width)
#define macro_i32_ty rewriter.getIntegerType(32)
#define i8_ty rewriter.getIntegerType(8)
#define void_ty(ctx) mlir::LLVM::LLVMVoidType::get(ctx)

// Constants
#define i32_val(...) tritoncc::createConstantI32(loc, rewriter, __VA_ARGS__)
#define vec_ty(type, num) mlir::VectorType::get(num, type)
#define int_val(width, val) tritoncc::createLLVMIntegerConstant(rewriter, loc, width, val)
#define tid_val() getThreadId(rewriter, loc)

namespace tritoncc {

mlir::Value createLLVMIntegerConstant(mlir::OpBuilder &builder, mlir::Location loc, short width, int64_t value) {
  mlir::Type ty = builder.getIntegerType(width);
  return builder.create<mlir::LLVM::ConstantOp>(loc, ty,
      builder.getIntegerAttr(ty, value));
}

mlir::Value createConstantI32(mlir::Location loc, mlir::OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32ty,
      mlir::IntegerAttr::get(i32ty, v));
}

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
    llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d} -> accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
    llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

mlir::Value getThreadIdInCTA(mlir::RewriterBase &rewriter, mlir::Location loc) {
  mlir::Value tid =
      rewriter.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  return rewriter.create<mlir::arith::IndexCastOp>(loc, macro_i32_ty, tid);
}

mlir::Value getThreadId(mlir::RewriterBase &rewriter, mlir::Location loc) {
  return getThreadIdInCTA(rewriter, loc);
}

mlir::Value getClusterCTAId(mlir::RewriterBase &rewriter, mlir::Location loc) {
  return rewriter.create<mlir::_tritoncc::ClusterCTAIdOp>(
    loc, rewriter.getI32Type()
  );
}

llvm::SmallVector<mlir::Value> delinearize(
    mlir::OpBuilder &b, mlir::Location loc, mlir::Value linear,
    llvm::ArrayRef<unsigned> shape,
    llvm::ArrayRef<unsigned> order);

// Get an index-base for each dimension for a \param blockedLayout.
llvm::SmallVector<mlir::Value>
emitBaseIndexWithinCTAForBlockedLayout(
    mlir::Location loc, mlir::RewriterBase &rewriter,
    const mlir::triton::gpu::BlockedEncodingAttr &blockedLayout,
    mlir::RankedTensorType type) {
  auto shape = type.getShape();
  mlir::Value threadId = getThreadId(rewriter, loc);
  mlir::Value warpSize = i32_val(mlir::triton::gpu::getWarpSize(blockedLayout));
  mlir::Value laneId = urem(threadId, warpSize);
  mlir::Value warpId = udiv(threadId, warpSize);
  auto sizePerThread = blockedLayout.getSizePerThread();
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
  auto order = blockedLayout.getOrder();
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(blockedLayout, shape);
  unsigned rank = shape.size();

  // delinearize threadId to get the base index
  llvm::SmallVector<mlir::Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);
  llvm::SmallVector<mlir::Value> multiDimThreadId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);

  llvm::SmallVector<mlir::Value> multiDimBase(rank);
  for (unsigned k = 0; k < rank; ++k) {
    // Wrap around multiDimWarpId/multiDimThreadId in case
    // shapePerCTATile[k] > shapePerCTA[k]
    auto maxWarps =
        ceil<unsigned>(shapePerCTA[k], sizePerThread[k] * threadsPerWarp[k]);
    auto maxThreads = ceil<unsigned>(shapePerCTA[k], sizePerThread[k]);
    multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
    multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));

    mlir::Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
    mlir::Value sizePerThreadK = i32_val(sizePerThread[k]);
    multiDimBase[k] =
      mul(sizePerThreadK,
        macro_add(multiDimThreadId[k], mul(multiDimWarpId[k], threadsPerWarpK)));
  }
  return multiDimBase;
}

llvm::SmallVector<mlir::Value> emitCTAOffsetForLayout(mlir::Location loc,
    mlir::RewriterBase &rewriter,
    mlir::Attribute layout,
    llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::SmallVector<unsigned> CTAsPerCGA = mlir::triton::gpu::getCTAsPerCGA(layout);
  llvm::SmallVector<unsigned> CTASplitNum = mlir::triton::gpu::getCTASplitNum(layout);
  llvm::SmallVector<unsigned> CTAOrder = mlir::triton::gpu::getCTAOrder(layout);
  llvm::SmallVector<int64_t> shapePerCTA =
      mlir::triton::gpu::getShapePerCTA(CTASplitNum, shape);

  // Delinearize clusterCTAId
  mlir::Value clusterCTAId = getClusterCTAId(rewriter, loc);
  llvm::SmallVector<mlir::Value> multiDimClusterCTAId =
      delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

  // CTA Wrapping
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistant with getShapePerCTA
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    multiDimClusterCTAId[i] = urem(multiDimClusterCTAId[i], i32_val(splitNum));
  }

  llvm::SmallVector<mlir::Value> CTAOffset(rank);
  for (unsigned i = 0; i < rank; ++i) {
    CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));
  }
  return CTAOffset;
}

llvm::SmallVector<mlir::Value>
emitBaseIndexForLayout(mlir::Location loc, mlir::RewriterBase &rewriter,
    mlir::Attribute layout, mlir::RankedTensorType type, bool withCTAOffset) {
  auto shape = type.getShape();

  llvm::SmallVector<mlir::Value> baseIndex;
  mlir::RewriterBase::InsertionGuard guard(rewriter);
  llvm::SmallVector<mlir::Value> result;
  if (auto blockedLayout = layout.dyn_cast<mlir::triton::gpu::BlockedEncodingAttr>()) {
    result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
        blockedLayout, type);
  } else if (auto mmaLayout = layout.dyn_cast<mlir::triton::gpu::NvidiaMmaEncodingAttr>()) {
    assert(false && "mma");
  } else if (auto mfmaLayout = layout.dyn_cast<mlir::triton::gpu::AMDMfmaEncodingAttr>()) {
    assert(false && "mfma");
  } else if (auto sliceLayout = layout.dyn_cast<mlir::triton::gpu::SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    mlir::RankedTensorType parentTy =
        mlir::RankedTensorType::get(parentShape, type.getElementType(), parentLayout);
    result = emitBaseIndexForLayout(loc, rewriter, parentLayout, parentTy,
        withCTAOffset);
    result.erase(result.begin() + sliceLayout.getDim());
    // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
    return result;
  } else {
    llvm_unreachable("unsupported emitBaseIndexForLayout");
  }
  if (withCTAOffset) {
    auto CTAOffset = emitCTAOffsetForLayout(loc, rewriter, layout, shape);
    assert(CTAOffset.size() == result.size() && "Rank mismatch");
    for (unsigned k = 0; k < result.size(); ++k) {
      result[k] = macro_add(result[k], CTAOffset[k]);
    }
  }
  return result;
}

llvm::SmallVector<llvm::SmallVector<unsigned>>
emitOffsetForBlockedLayout(const mlir::triton::gpu::BlockedEncodingAttr &blockedLayout,
    mlir::RankedTensorType type) {
  auto shape = type.getShape();
  auto sizePerThread = blockedLayout.getSizePerThread();
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
  auto order = blockedLayout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(blockedLayout);
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(blockedLayout, shape);

  unsigned rank = shape.size();
  llvm::SmallVector<unsigned> tilesPerDim(rank);
  for (unsigned k = 0; k < rank; ++k) {
    tilesPerDim[k] = ceil<unsigned>(shapePerCTA[k], shapePerCTATile[k]);
  }

  unsigned elemsPerThread = mlir::triton::gpu::getTotalElemsPerThread(type);
  unsigned totalSizePerThread = product<unsigned>(sizePerThread);
  llvm::SmallVector<llvm::SmallVector<unsigned>> reorderedOffset(elemsPerThread);
  for (unsigned n = 0; n < elemsPerThread; ++n) {
    unsigned linearNanoTileId = n / totalSizePerThread;
    unsigned linearNanoTileElemId = n % totalSizePerThread;
    llvm::SmallVector<unsigned> multiDimNanoTileId =
        getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim, order);
    llvm::SmallVector<unsigned> multiDimNanoTileElemId =
        getMultiDimIndex<unsigned>(linearNanoTileElemId, sizePerThread, order);
    for (unsigned k = 0; k < rank; ++k) {
      unsigned reorderedMultiDimId =
        multiDimNanoTileId[k] *
            (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
        multiDimNanoTileElemId[k];
      reorderedOffset[n].push_back(reorderedMultiDimId);
    }
  }
  return reorderedOffset;
}

llvm::SmallVector<llvm::SmallVector<unsigned>>
emitOffsetForLayout(mlir::Attribute layout, mlir::RankedTensorType type);

llvm::SmallVector<mlir::SmallVector<unsigned>>
emitOffsetForSliceLayout(const mlir::triton::gpu::SliceEncodingAttr &sliceLayout,
    mlir::RankedTensorType type) {
  auto parentEncoding = sliceLayout.getParent();
  unsigned dim = sliceLayout.getDim();
  auto parentShape = sliceLayout.paddedShape(type.getShape());
  mlir::RankedTensorType parentTy =
      mlir::RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
  auto parentOffsets = emitOffsetForLayout(parentEncoding, parentTy);

  unsigned numOffsets = parentOffsets.size();
  llvm::SmallVector<llvm::SmallVector<unsigned>> resultOffsets;
  std::set<llvm::SmallVector<unsigned>> uniqueOffsets;

  for (unsigned i = 0; i < numOffsets; ++i) {
    llvm::SmallVector<unsigned> offsets = parentOffsets[i];
    offsets.erase(offsets.begin() + dim);
    if (uniqueOffsets.find(offsets) == uniqueOffsets.end()) {
      resultOffsets.push_back(offsets);
      uniqueOffsets.insert(offsets);
    }
  }
  return resultOffsets;
}

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

bool isKernel(mlir::FunctionOpInterface funcOp);

mlir::Value getStackPointer(mlir::PatternRewriter &rewriter,
    mlir::FunctionOpInterface funcOp) {
  auto mod = funcOp->getParentOfType<mlir::ModuleOp>();
  mlir::LLVM::GlobalOp globalBase = nullptr;
  mod.walk([&](mlir::LLVM::GlobalOp op) {
    if (op.getSymName() == "global_smem") {
      globalBase = op;
    }
  });
  assert(globalBase);
  if (isKernel(funcOp)) {
    return rewriter.create<mlir::LLVM::AddressOfOp>(funcOp.getLoc(), globalBase);
  } else {
    return funcOp.getArgument(funcOp.getNumArguments() - 1);
  }
}

mlir::Value getSharedMemoryBase(mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::Operation *op) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
  mlir::FunctionOpInterface func =
      op->template getParentOfType<mlir::FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = op->getAttr("allocation.offset")
      .cast<mlir::IntegerAttr>()
      .getValue()
      .getZExtValue();
  mlir::Value offVal = i32_val(offset);
  mlir::Value base = gep(ptrTy, i8_ty, tritoncc::getStackPointer(rewriter, func), offVal);
  return base;
}

mlir::Value storeShared(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value ptr, mlir::Value val, mlir::Value pred) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  unsigned bits = std::max(8u, val.getType().getIntOrFloatBitWidth());
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  mlir::triton::PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
}

mlir::Value loadShared(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value ptr, mlir::Type elemTy, mlir::Value pred) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = ptr.getType().cast<mlir::LLVM::LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  mlir::triton::PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d} -> accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
    llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(applyPermutation(multiDimIndex, order),
      applyPermutation(shape, order));
}

mlir::Value getSRegValue(mlir::OpBuilder &b, mlir::Location loc, const std::string &sRegStr) {
  mlir::triton::PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  mlir::Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
}

mlir::Value llGetPid(int axis, mlir::Location loc, mlir::ModuleOp moduleOp,
    mlir::ConversionPatternRewriter &rewriter) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = mlir::triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
  sreg.append(1, 'x' + axis);
  return getSRegValue(rewriter, loc, sreg);
}

bool isKernel(mlir::FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public;
}

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
llvm::SmallVector<llvm::SmallVector<mlir::Value>>
emitIndices(mlir::Location loc, mlir::RewriterBase &rewriter,
      mlir::Attribute layout, mlir::RankedTensorType type, bool withCTAOffset) {
  // step 1, delinearize threadId to get the base index
  auto multiDimBase =
      emitBaseIndexForLayout(loc, rewriter, layout, type, withCTAOffset);
  // step 2, get offset of each element
  auto offset = emitOffsetForLayout(layout, type);
  // step 3, add offset to base, and reorder the sequence
  // of indices to guarantee that elems in the same
  // sizePerThread are adjacent in order
  auto shape = type.getShape();
  unsigned rank = shape.size();
  unsigned elemsPerThread = offset.size();
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> multiDimIdx(
      elemsPerThread, llvm::SmallVector<mlir::Value>(rank));

  for (unsigned n = 0; n < elemsPerThread; ++n) {
    for (unsigned k = 0; k < rank; ++k) {
      multiDimIdx[n][k] = macro_add(multiDimBase[k], i32_val(offset[n][k]));
    }
  }
  return multiDimIdx;
}

mlir::Value commonShflSync(
    mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::Value val, mlir::Value i, mlir::NVVM::ShflKind mode,
    mlir::Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    assert(false && "64 bit");
  }
  mlir::Type type = val.getType();
  if (type != macro_i32_ty) {
    val = bitcast(val, int_ty(bits));
    if (bits < 32) {
      val = zext(macro_i32_ty, val);
    }
  }
  mlir::Value mask = i32_val(0xFFFFFFFF);
  mlir::Value result = rewriter.create<mlir::NVVM::ShflOp>(
    loc, macro_i32_ty, mask, val, i, clamp, mode, mlir::UnitAttr());
  if (type != macro_i32_ty) {
    if (bits < 32) {
      result = trunc(int_ty(bits), result);
    }
    result = bitcast(result, type);
  }
  return result;
}

mlir::Value shflSync(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::Value val, int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), mlir::NVVM::ShflKind::bfly,
      i32_val(0x1f));
}

llvm::SmallVector<mlir::Value> delinearize(mlir::OpBuilder &b, mlir::Location loc, mlir::Value linear,
    llvm::ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  llvm::SmallVector<mlir::Value> multiDim(rank);
  if (rank == 1) {
    multiDim[0] = linear;
  } else {
    mlir::Value remained = linear;
    for (auto &&en : llvm::enumerate(shape.drop_back())) {
      auto dimSize = b.create<mlir::arith::ConstantIntOp>(loc, en.value(), 32);
      multiDim[en.index()] = b.create<mlir::arith::RemSIOp>(loc, remained, dimSize);
      remained = b.create<mlir::arith::DivSIOp>(loc, remained, dimSize);
    }
    multiDim[rank - 1] = remained;
  }
  return multiDim;
}

llvm::SmallVector<mlir::Value> delinearize(
    mlir::OpBuilder &b, mlir::Location loc, mlir::Value linear,
    llvm::ArrayRef<unsigned> shape,
    llvm::ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  auto reorderedMultiDim = delinearize(b, loc, linear, reordered);
  llvm::SmallVector<mlir::Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

mlir::Value linearize(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, llvm::ArrayRef<mlir::Value> multiDim, llvm::ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  mlir::Value linear = i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      mlir::Value dimSize = i32_val(dimShape);
      linear = macro_add(mul(linear, dimSize), dim);
    }
  }
  return linear;
}

mlir::Value linearize(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, llvm::ArrayRef<mlir::Value> multiDim, llvm::ArrayRef<unsigned> shape, llvm::ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
      applyPermutation(shape, order));
}

// Return the operand used to access the memory in the operation.
mlir::Value getMemAccessPtr(mlir::Operation* op) {
  if (mlir::triton::LoadOp ld = llvm::dyn_cast<mlir::triton::LoadOp>(op)) {
    return ld.getPtr();
  }
  if (mlir::triton::StoreOp st = llvm::dyn_cast<mlir::triton::StoreOp>(op)) {
    return st.getPtr();
  }
  return nullptr;
}

unsigned getElementBitWidth(mlir::RankedTensorType type) {
  auto typeForMem =
      type.getElementType().isa<mlir::triton::PointerType>()
          ? type.getElementType().cast<mlir::triton::PointerType>().getPointeeType()
          : type.getElementType();
  return typeForMem.getIntOrFloatBitWidth();
}

namespace {

// Detect dead arguments in scf.for op by assuming all the values are dead and
// propagate liveness property.
struct ForOpDeadArgElimination : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::scf::ForOp forOp,
      mlir::PatternRewriter &rewriter) const final {
    assert(false && "matchAndRewrite");
  }
};

}

// Populate pattern to remove dead cycles in ForOp.
void populateForOpDeadArgumentElimination(mlir::RewritePatternSet &patterns) {
  patterns.add<ForOpDeadArgElimination>(patterns.getContext());
}

}

#endif
