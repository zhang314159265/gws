#pragma once

#include "mlir/Analysis/Liveness.h"
#include "tritoncc/analysis_util.h"

#ifdef USE_TRITON
#undef USE_TRITON
#endif

#define USE_TRITON 0

#if USE_TRITON
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"

namespace tritoncc {
using mlir::triton::gpu::createAllocateSharedMemoryPass;
}

#else

namespace tritoncc {

// Bitwidth of pointers
constexpr int kPtrBitWidth = 64;

template <typename T> class Interval {
 public:
  Interval() {}
  Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }
  bool intersects(const Interval &R) const {
    return Start < R.End && R.Start < End;
  }
  T start() const { return Start; }
  T end() const { return End; }
  T size() const { return End - Start; }
 private:
  T Start = std::numeric_limits<T>::min();
  T End = std::numeric_limits<T>::max();
};

class Allocation {
 public:
  // A unique identifier for shared memory buffers
  using BufferId = size_t;

  using FuncAllocMapT = CallGraph<Allocation>::FuncDataMapT;

  static constexpr BufferId InvalidBufferId =
      std::numeric_limits<BufferId>::max();

  Allocation() = default;
  explicit Allocation(mlir::Operation *operation) : operation(operation) {}

  // runs allocation analysis on the given top-level operation.
  void run(FuncAllocMapT &funcAllocMap);
  mlir::Operation *getOperation() const { return operation; }

  size_t getSharedMemorySize() const { return sharedMemorySize; }

  BufferId getBufferId(mlir::Value value) const {
    if (valueBuffer.count(value)) {
      return valueBuffer.lookup(value)->id;
    } else {
      return InvalidBufferId;
    }
  }

  BufferId getBufferId(mlir::Operation *operation) const {
    if (opScratch.count(operation)) {
      return opScratch.lookup(operation)->id;
    } else if (opVirtual.count(operation)) {
      return opVirtual.lookup(operation)->id;
    } else {
      return InvalidBufferId;
    }
  }

  size_t getOffset(BufferId bufferId) const {
    return bufferSet.at(bufferId).offset;
  }

 private:
  // A class that represents a shared memory buffer
  struct BufferT {
    // Explicit: triton_gpu.alloc_tensor
    // Scratch: triton_gpu.convert_layout
    // Virtual: triton.call
    enum class BufferKind { Explicit, Scratch, Virtual };

    inline static std::atomic<BufferId> nextId = 0;
    
    BufferKind kind;
    BufferId id;
    size_t size, alignment, offset;

    BufferT() : BufferT(BufferKind::Explicit, 0) { }
    BufferT(BufferKind kind, size_t size, size_t alignment = 4,
        size_t offset = 0)
        : kind(kind), id(nextId++), size(size), alignment(alignment),
          offset(offset) {}
  };

  using OpScratchMapT = llvm::DenseMap<mlir::Operation *, BufferT *>;
  using BufferSetT = std::map<BufferId, BufferT>;
  using AliasBufferMapT = llvm::MapVector<mlir::Value, llvm::SetVector<BufferT *>>;
  using ValueBufferMapT = llvm::MapVector<mlir::Value, BufferT *>;

  void addAlias(mlir::Value value, mlir::Value alloc) {
    assert(false && "addAlias"); 
  }

  template <BufferT::BufferKind Kind, typename KeyType, typename... Args>
  void addBuffer(KeyType &key, Args &&...args) {
    auto buffer = BufferT(Kind, std::forward<Args>(args)...);
    bufferSet[buffer.id] = std::move(buffer);
    if constexpr (Kind == BufferT::BufferKind::Explicit) {
      assert(false && "Explicit");
    } else if constexpr (Kind == BufferT::BufferKind::Virtual) {
      assert(false && "Virtual");
    } else {
      opScratch[key] = &bufferSet[buffer.id];
    }
  }

  mlir::Operation *operation = nullptr;
  OpScratchMapT opScratch;
  OpScratchMapT opVirtual;
  BufferSetT bufferSet;
  AliasBufferMapT aliasBuffer;
  ValueBufferMapT valueBuffer;
  size_t sharedMemorySize = 0;

  friend class AllocationAnalysis;
};

class AllocationAnalysis {
 public:
  AllocationAnalysis(mlir::Operation *operation,
      Allocation::FuncAllocMapT *funcAllocMap,
      Allocation *allocation)
    : operation(operation), funcAllocMap(funcAllocMap),
      allocation(allocation) {
    run();
  }
 private:
  using BufferT = Allocation::BufferT;
  // Value -> Liveness Range
  // Use MapVector to ensure determinism
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  // Nodes -> Nodes
  using GraphT = llvm::DenseMap<BufferT *, llvm::DenseSet<BufferT *>>;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  void getExplicitValueSize(mlir::Operation *op) {
    if (!maybeSharedAllocationOp(op) || maybeAliasOp(op)) {
      return;
    }
    size_t kAlignment = 0;
    for (mlir::Value result : op->getResults()) {
      if (tritoncc::hasSharedEncoding(result)) {
        assert(false && "getExplicitValueSize");
      }
    }
  }

  template <BufferT::BufferKind T>
  void maybeAddScratchBuffer(mlir::Operation *op, unsigned bytes,
      unsigned alignment) {
    if (bytes > 0) {
      allocation->addBuffer<T>(op, bytes, alignment);
    }
  }

  template <BufferT::BufferKind T>
  void maybeAddScratchBuffer(mlir::Operation *op, unsigned bytes) {
    if (bytes > 0) {
      allocation->addBuffer<T>(op, bytes);
    }
  }

  // initializes temporary shared memory for a given operation.
  void getScratchValueSize(mlir::Operation *op) {
    const size_t scratchAlignment = 128;
    if (auto reduceOp = llvm::dyn_cast<mlir::triton::ReduceOp>(op)) {
      ReduceOpHelper helper(reduceOp);
      unsigned bytes = helper.getScratchSizeInBytes();
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes,
          scratchAlignment);
    } else if (auto scanOp = llvm::dyn_cast<mlir::triton::ScanOp>(op)) {
      assert(false && "ScanOp");
    } else if (auto histogram = llvm::dyn_cast<mlir::triton::HistogramOp>(op)) {
      assert(false && "HistogramOp");
    } else if (auto cvtLayout = llvm::dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtLayout.getSrc().getType();
      auto dstTy = cvtLayout.getType();
      auto srcEncoding = srcTy.getEncoding();
      auto dstEncoding = dstTy.getEncoding();
      if (srcEncoding.isa<mlir::triton::gpu::SharedEncodingAttr>() ||
          dstEncoding.isa<mlir::triton::gpu::SharedEncodingAttr>()) {
        // Conversions from/to shared memory do not need scratch memory.
        return;
      }

      // ConvertLayoutOp with both input/output non-shared_layout
      // TODO: Besides of implementing ConvertLayoutOp via shared memory, it's
      //       also possible to realize it with other approaches in restricted
      //       conditions, such as warp-shuffle
      unsigned inVec = 0;
      unsigned outVec = 0;
      auto smemShape = getScratchConfigForCvtLayout(cvtLayout, inVec, outVec);
      unsigned elems = std::accumulate(smemShape.begin(), smemShape.end(), 1,
          std::multiplies{});

      auto bytes =
          srcTy.getElementType().isa<mlir::triton::PointerType>()
            ? elems * kPtrBitWidth / 8
            : elems * std::max<int>(8, srcTy.getElementTypeBitWidth()) / 8;
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes, scratchAlignment);
    } else if (auto atomicRMWOp = llvm::dyn_cast<mlir::triton::AtomicRMWOp>(op)) {
      assert(false && "AtomicRMWOp");
    } else if (auto atomicCASOp = llvm::dyn_cast<mlir::triton::AtomicCASOp>(op)) {
      assert(false && "AtomicCASOp");
    } else if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
      assert(false && "CallOpInterface");
    }
  }

  void getValueAlias(mlir::Value value, SharedMemoryAliasAnalysis &analysis) {
    mlir::dataflow::Lattice<AliasInfo> *latticeElement =
        analysis.getLatticeElement(value);
    if (latticeElement) {
      AliasInfo &info = latticeElement->getValue();
      if (!info.getAllocs().empty()) {
        for (auto alloc : info.getAllocs()) {
          allocation->addAlias(value, alloc);
        }
      }
    }
  }

  // Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      getExplicitValueSize(op);
      getScratchValueSize(op);
    });
    // Get the alias values
    std::unique_ptr<mlir::DataFlowSolver> solver = tritoncc::createDataFlowSolver();
    SharedMemoryAliasAnalysis *aliasAnalysis =
        solver->load<SharedMemoryAliasAnalysis>();
    if (failed(solver->initializeAndRun(operation))) {
      // TODO: return error intead of bailing out..
      llvm_unreachable("failed to run SharedMemoryAliasAnalysis");
    }
    operation->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        getValueAlias(operand, *aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, *aliasAnalysis);
      }
    });
  }

  void resolveExplicitBufferLiveness(
      llvm::function_ref<Interval<size_t>(mlir::Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      assert(false && "resolveExplicitBufferLiveness");
    }
  }

  void resolveAliasBufferLiveness(
      llvm::function_ref<Interval<size_t>(mlir::Value value)> getLiveness) {
    for (auto aliasBufferIter : allocation->aliasBuffer) {
      assert(false && "resolveAliasBufferLiveness");
    }
  }

  void resolveScratchBufferLiveness(
      const llvm::DenseMap<mlir::Operation *, size_t> &operationId) {
    // Analyze liveness of scratch buffers and virtual buffers
    auto processScratchMemory = [&](const auto &container) {
      for (auto opScratchIter : container) {
        auto *op = opScratchIter.first;
        auto *buffer = opScratchIter.second;
        bufferRange.insert({buffer, Interval(operationId.lookup(op),
          operationId.lookup(op) + 1)});
      }
    };
    processScratchMemory(allocation->opScratch);
    processScratchMemory(allocation->opVirtual);
  }

  void resolveLiveness() {
    llvm::DenseMap<mlir::Operation *, size_t> operationId;
    operation->walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::Operation *op) { operationId[op] = operationId.size(); });

    // Analyze liveness of explicit buffers
    mlir::Liveness liveness(operation);
    auto getValueLivenessRange = [&](mlir::Value value) -> Interval<size_t> {
      assert(false && "getValueLivenessRange");
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
    resolveAliasBufferLiveness(getValueLivenessRange);
    resolveScratchBufferLiveness(operationId);
  }

  // Computes the initial shared memory offsets
  void calculateStarts(const llvm::SmallVector<BufferT *> &buffers,
      llvm::DenseMap<BufferT *, size_t> &bufferStart) {
    using TripleMapT = std::multimap<size_t, Interval<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Interval<size_t>()));
    llvm::SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto size = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt = 
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (auto val : tripleMap) {
              res = res && !val.second.intersects(xRange); // only one buffer intersect
            }
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        size_t alignment = buffer->alignment;
        size_t alignSize = ((size + alignment - 1) / alignment) * alignment;
        bufferStart[buffer] = alignSize;
        tripleMap.insert({alignSize + xSize,
            Interval{std::max(range.start(), xRange.start()),
              std::min(range.end(), xRange.end())}});
        if (range.start() < xRange.start()) {
          tripleMap.insert({size, Interval{range.start(), xRange.end()}});
        }
        if (xRange.end() < range.end()) {
          tripleMap.insert({size, Interval{xRange.start(), range.end()}});
        }
        xBuffers.erase(bufferIt);
      }
    }
  }

  void buildInterferenceGraph(const llvm::SmallVector<BufferT *> &buffers,
      const llvm::DenseMap<BufferT *, size_t> &bufferStart,
      GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y) {
          continue;
        }
        auto xStart = bufferStart.lookup(x);
        auto yStart = bufferStart.lookup(y);
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }
      }
    }
  }

  // Finalizes shared memory offsets considering interference
  void allocate(const llvm::SmallVector<BufferT *> &buffers,
      const GraphT &interference,
      llvm::DenseMap<BufferT *, size_t> &bufferStart) {
    // Reset shared memory size
    allocation->sharedMemorySize = 0;
    llvm::DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    llvm::SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
    }

    // Finalize allocation
    for (auto x : buffers) {
      size_t adj = 0;
      for (auto y : interference.lookup(x)) {
        adj = std::max(adj, bufferStart.lookup(y) + y->size);
      }
      x->offset = bufferStart.lookup(x) + colors.lookup(x) * adj;
      bufferStart[x] = x->offset;
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
  }

  // Computes the shared memory offsets for all related values.
  // Paper: Algorithms for Compile-Time Memory Optimization
  void computeOffsets() {
    llvm::SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    llvm::DenseMap<BufferT *, size_t> bufferStart;
    calculateStarts(buffers, bufferStart);
    
    GraphT interference;
    buildInterferenceGraph(buffers, bufferStart, interference);
    do {
      allocate(buffers, interference, bufferStart);
      buildInterferenceGraph(buffers, bufferStart, interference);
    } while (!interference.empty());
  }

  mlir::Operation *operation;
  Allocation::FuncAllocMapT *funcAllocMap;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
};

void Allocation::run(FuncAllocMapT &funcAllocMap) {
  AllocationAnalysis(getOperation(), &funcAllocMap, this);
}

class ModuleAllocation : public tritoncc::CallGraph<Allocation> {
 public:
  explicit ModuleAllocation(mlir::ModuleOp moduleOp) : CallGraph<Allocation>(moduleOp) {
    walk<mlir::WalkOrder::PreOrder, mlir::WalkOrder::PostOrder>(
      // pre-order edge walk callback
      [](mlir::CallOpInterface callOp, mlir::FunctionOpInterface funcOp) {},
      // post-order node walk callback
      [&](mlir::FunctionOpInterface funcOp) {
        auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
        if (inserted) {
          iter->second.run(funcMap);
        }
      }
    );
  }

  size_t getSharedMemorySize() {
    size_t size = 0;
    for (auto funcOp : getRoots()) {
      auto *alloc = getFuncData(funcOp);
      size = std::max(size, alloc->getSharedMemorySize());
    }
    return size;
  }
};

struct AllocateSharedMemory : public mlir::OperationPass<mlir::ModuleOp> {
  explicit AllocateSharedMemory() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<AllocateSharedMemory>()) { }

  llvm::StringRef getName() const override {
    return "AllocateSharedMemory";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod);

    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      funcOp.walk([&](mlir::Operation *op) {
        auto *funcAllocation = allocation.getFuncData(funcOp);
        auto oBufferId = funcAllocation->getBufferId(op);
        int offset = -1;
        if (oBufferId != Allocation::InvalidBufferId) {
          offset = funcAllocation->getOffset(oBufferId);
        } else if (op->getNumResults() == 1) {
          mlir::Value value = op->getResult(0);
          auto vBufferId = funcAllocation->getBufferId(value);
          if (vBufferId != Allocation::InvalidBufferId) {
            offset = funcAllocation->getOffset(vBufferId);
          }
        }
        if (offset == -1) {
          return;
        }
        op->setAttr("allocation.offset",
            IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      });
    });
    mod->setAttr("triton_gpu.shared",
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
            allocation.getSharedMemorySize()));
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemory>();
}

}
#endif
