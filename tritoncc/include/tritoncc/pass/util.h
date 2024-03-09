#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"

namespace tritoncc {
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

}
