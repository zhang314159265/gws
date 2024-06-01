#pragma once

#include <numeric>

#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"

namespace tritoncc {

bool isTensorPointerType(mlir::Type type) {
  if (mlir::triton::PointerType ptrType = type.dyn_cast<mlir::triton::PointerType>()) {
    return ptrType.getPointeeType().isa<mlir::RankedTensorType>();
  }
  return false;
}

bool isSingleValue(mlir::Value value) {
  if (mlir::RankedTensorType tensorTy = value.getType().dyn_cast<mlir::RankedTensorType>()) {
    return tensorTy.getNumElements() == 1;
  }
  return true;
}

int tritonGPUGetNumWarps(mlir::ModuleOp mod) {
  if (!mod->hasAttr("triton_gpu.num-warps")) {
    llvm::report_fatal_error("TritonGPU module should contain a triton_gpu.num-warps attribute");
  }
  return mod->getAttr("triton_gpu.num-warps").cast<mlir::IntegerAttr>().getInt();
}

int tritonGPUGetThreadsPerWarp(mlir::ModuleOp mod) {
  mlir::Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu.threads-per-warp");
  if (!threadsPerWarp) {
    return 32;
  }
  return threadsPerWarp.cast<mlir::IntegerAttr>().getInt();
}

bool isExpensiveLoadOrStore(mlir::Operation *op) {
  mlir::Type operandType = op->getOperand(0).getType();
  if (isTensorPointerType(operandType)) {
    return true;
  }
  if (tritoncc::isSingleValue(op->getOperand(0))) {
    return false;
  }
  mlir::RankedTensorType tensorType = op->getOperand(0).getType().cast<mlir::RankedTensorType>();
  mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
  int numWarps = tritonGPUGetNumWarps(mod);
  int threadsPerWarp = tritonGPUGetThreadsPerWarp(mod);
  return tensorType.getNumElements() >= numWarps * threadsPerWarp;
}

bool canFoldIntoConversion(mlir::Operation *op, mlir::Attribute targetEncoding) {
  if (llvm::isa<mlir::triton::CatOp>(op)) {
    assert(false && "CatOp");
  }
  if (auto convert = llvm::dyn_cast<mlir::_tritoncc::ConvertLayoutOp>(op)) {
    assert(false && "ConvertLayoutOp");
  }
  if (auto reshape = llvm::dyn_cast<mlir::triton::ReshapeOp>(op)) {
    assert(false && "ReshapeOp");
  }
  return llvm::isa<mlir::arith::ConstantOp,
    mlir::triton::MakeRangeOp,
    mlir::triton::SplatOp,
    mlir::triton::HistogramOp>(op);
}

template <typename T>
bool hasEncoding(mlir::Value value) {
  mlir::Type type = value.getType();
  if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
    mlir::Attribute encoding = tensorType.getEncoding();
    return encoding && encoding.isa<T>();
  }
  return false;
}

bool hasSharedEncoding(mlir::Value value) {
  return hasEncoding<mlir::_tritoncc::SharedEncodingAttr>(value);
}

mlir::Attribute inferDstEncoding(mlir::triton::ReduceOp op,
    mlir::Attribute encoding) {
  return mlir::_tritoncc::SliceEncodingAttr::get(op->getContext(), op.getAxis(), encoding);
}

mlir::Attribute inferDstEncoding(mlir::Operation *op, mlir::Attribute encoding) {
  if (llvm::isa<mlir::triton::ScanOp>(op)) {
    assert(false && "ScanOp");
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      llvm::isa<mlir::scf::WhileOp, mlir::scf::ForOp, mlir::scf::YieldOp, mlir::scf::ConditionOp>(op)) {
    return encoding;
  }
  if (auto reduceOp = llvm::dyn_cast<mlir::triton::ReduceOp>(op)) {
    return inferDstEncoding(reduceOp, encoding);
  }
  llvm::errs() << "inferDstEncoding for " << *op << "\n";
  assert(false && "infertDstEncoding");
}

std::optional<mlir::Attribute> inferSrcEncoding(mlir::triton::ReduceOp op,
    mlir::Attribute encoding) {
  auto sliceEncoding = encoding.dyn_cast<mlir::_tritoncc::SliceEncodingAttr>();
  if (!sliceEncoding) {
    return std::nullopt;
  }
  if (op.getAxis() != sliceEncoding.getDim()) {
    return std::nullopt;
  }
  return sliceEncoding.getParent();
}

std::optional<mlir::Attribute> inferSrcEncoding(mlir::triton::ExpandDimsOp op,
    mlir::Attribute encoding) {
  return mlir::_tritoncc::SliceEncodingAttr::get(
    op->getContext(), op.getAxis(), encoding);
}

std::optional<mlir::Attribute> inferSrcEncoding(mlir::Operation *op, mlir::Attribute encoding) {
  if (llvm::isa<mlir::triton::ScanOp>(op)) {
    assert(false && "ScanOp");
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      llvm::isa<mlir::scf::WhileOp, mlir::scf::YieldOp, mlir::scf::ConditionOp>(op)) {
    return encoding;
  }
  if (auto reduceOp = llvm::dyn_cast<mlir::triton::ReduceOp>(op)) {
    return inferSrcEncoding(reduceOp, encoding);
  }
  if (auto expand = llvm::dyn_cast<mlir::triton::ExpandDimsOp>(op)) {
    return inferSrcEncoding(expand, encoding);
  }
  llvm::errs() << "inferSrcEncoding for " << *op << "\n";
  assert(false && "infertSrcEncoding");
}

llvm::SmallVector<unsigned, 4> argSortDesc(const llvm::SmallVector<int64_t>& arr) {
  llvm::SmallVector<unsigned, 4> ret(arr.size());
  std::iota(ret.begin(), ret.end(), 0);
  std::stable_sort(ret.begin(), ret.end(),
    [&](unsigned x, unsigned y) { return arr[x] > arr[y]; });
  return ret;
}

template <typename T>
T product(llvm::ArrayRef<T> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies());
}

mlir::LogicalResult
getConvertBackwardSlice(mlir::Value root, llvm::SetVector<mlir::Value> &slice,
    mlir::Attribute rootEncoding,
    llvm::DenseMap<mlir::Value, mlir::Attribute> &layout,
    std::function<bool(mlir::Operation *)> stopPropagation) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::Attribute>> queue = {{root, rootEncoding}};
  while (!queue.empty()) {
    auto [currentValue, encoding] = queue.back();
    queue.pop_back();
    if (!currentValue.getType().isa<mlir::RankedTensorType>()) {
      continue;
    }
    if (currentValue.getDefiningOp<mlir::scf::ForOp>()) {
      return mlir::failure();
    }
    slice.insert(currentValue);
    if (layout.find(currentValue) != layout.end()) {
      if (layout[currentValue] != encoding) {
        return mlir::failure();
      }
    }
    layout[currentValue] = encoding;
    if (auto ifOp = currentValue.getDefiningOp<mlir::scf::IfOp>()) {
      assert(false && "IfOp");
    }
    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (mlir::Value result : definingOp->getResults()) {
        if (result == currentValue || !result.getType().isa<mlir::RankedTensorType>()) {
          continue;
        }
        if (layout.find(result) != layout.end()) {
          if (layout[result] != encoding) {
            return mlir::failure();
          }
          continue;
        }
        layout[result] = encoding;
      }
      if (tritoncc::canFoldIntoConversion(definingOp, encoding)) {
        continue;
      }
      if (stopPropagation && stopPropagation(definingOp)) {
        continue;
      }
      if (llvm::isa<mlir::triton::CatOp>(definingOp)) {
        return mlir::failure();
      }
      for (mlir::Value operand : definingOp->getOperands()) {
        std::optional<mlir::Attribute> srcEncoding = tritoncc::inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding) {
          return mlir::failure();
        }
        if (slice.count(operand) == 0) {
          queue.push_back({operand, *srcEncoding});
        }
      }
      continue;
    }
    assert(false && "getConvertBackwardSlice");
  }
  return mlir::success();
}

mlir::Type getPointeeType(mlir::Type type) {
  if (auto tensorTy = type.dyn_cast<mlir::RankedTensorType>()) {
    // Tensor of pointers
    auto shape = tensorTy.getShape();
    auto ptrType = tensorTy.getElementType().dyn_cast<mlir::triton::PointerType>();
    mlir::Type pointeeType = ptrType.getPointeeType();
    return mlir::RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
  } else if (auto ptrType = type.dyn_cast<mlir::triton::PointerType>()) {
    // scalar pointer
    return ptrType.getPointeeType();
  }
  return type;
}

unsigned getPointeeBitWidth(mlir::Type type) {
  mlir::Type pointeeType = getPointeeType(type);
  if (mlir::RankedTensorType tensorTy = pointeeType.dyn_cast<mlir::RankedTensorType>()) {
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  }
  return pointeeType.getIntOrFloatBitWidth();
}

mlir::Value packLLElements(mlir::Location loc, const mlir::LLVMTypeConverter *typeConverter, mlir::ValueRange resultVals, mlir::ConversionPatternRewriter &rewriter, mlir::Type type) {
  mlir::LLVM::LLVMStructType structType = typeConverter->convertType(type).dyn_cast<mlir::LLVM::LLVMStructType>();
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }
  mlir::Value llvmStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, structType);
  auto elementTypes = structType.getBody();
  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value());
    assert(v.value().getType() == elementTypes[v.index()]);
    llvmStruct = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

llvm::SmallVector<mlir::Value> unpackLLElements(mlir::Location loc, mlir::Value llvmStruct, mlir::ConversionPatternRewriter &rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      llvmStruct.getType().isa<mlir::triton::PointerType>() ||
      llvmStruct.getType().isa<mlir::LLVM::LLVMPointerType>()) {
    return {llvmStruct};
  }
  llvm::ArrayRef<mlir::Type> types = llvmStruct.getType().cast<mlir::LLVM::LLVMStructType>().getBody();
  llvm::SmallVector<mlir::Value> results(types.size());
  for (int i = 0; i < types.size(); ++i) {
    results[i] = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, types[i], llvmStruct, i);
  }
  return results;
}

llvm::SmallVector<mlir::Value> unpackI32(const llvm::SmallVector<mlir::Value> &inValues, mlir::Type srcTy, mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, const mlir::LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<mlir::RankedTensorType>();
  if (!tensorTy) {
    return inValues;
  }
  auto encoding = tensorTy.getEncoding().dyn_cast<mlir::_tritoncc::DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>())) {
    return inValues;
  }
  assert(false && "unpackI32");
}

llvm::SmallVector<mlir::Value> packI32(const llvm::SmallVector<mlir::Value> &inValues, mlir::Type srcTy, mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, const mlir::LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<mlir::RankedTensorType>();
  if (!tensorTy) {
    return inValues;
  }
  auto encoding = tensorTy.getEncoding().dyn_cast<mlir::_tritoncc::DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<mlir::_tritoncc::NvidiaMmaEncodingAttr>())) {
    return inValues;
  }
  assert(false && "packI32");
}

bool isaDistributedLayout(mlir::Attribute layout) {
  return layout.isa<mlir::_tritoncc::BlockedEncodingAttr>()
    || layout.isa<mlir::_tritoncc::MmaEncodingTrait>()
    || layout.isa<mlir::_tritoncc::SliceEncodingAttr>();
}

template <typename T>
T nextPowOf2(T n) {
  assert(false && "nextPowOf2");
}

// Get the highest power of 2 divisor of an integer.
template <typename T>
T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & -n);
}

template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

template <typename T, typename U>
llvm::SmallVector<T> convertType(llvm::ArrayRef<U> in) {
  llvm::SmallVector<T> out;
  for (const auto &i : in) {
    out.push_back(T(i));
  }
  return out;
}

template <typename T, typename VecU>
llvm::SmallVector<T> convertType(const VecU &in) {
  return convertType<T>(llvm::ArrayRef(in));
}

template <typename T, typename U>
llvm::SmallVector<T> applyPermutation(llvm::ArrayRef<T> vec, llvm::ArrayRef<U> permutation) {
  static_assert(std::is_integral_v<U>);
  assert(vec.size() == permutation.size());

  // Check that `permutation` is actually a permutation.
#ifndef NDEBUG
  llvm::SmallVector<U> sortedPerm(permutation);
  llvm::sort(sortedPerm);
  for (U i = 0; i < static_cast<U>(sortedPerm.size()); i++) {
    assert(sortedPerm[i] == i);
  }
#endif

  llvm::SmallVector<T> ret;
  ret.reserve(vec.size());
  for (const U &i : permutation) {
    ret.push_back(vec[i]);
  }
  return ret;
}

template <typename VecT, typename PermT>
auto applyPermutation(const VecT &vec, const PermT &permutation) {
  return applyPermutation(llvm::ArrayRef(vec), llvm::ArrayRef(permutation));
}

}
