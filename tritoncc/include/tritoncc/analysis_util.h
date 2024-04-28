#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Analysis/Utility.h"

namespace tritoncc {
using mlir::CallGraph;
using mlir::multiRootGetSlice;
using mlir::createDataFlowSolver;
using mlir::multiRootTopologicalSort;
}
#else

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tritoncc {

// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public mlir::DataFlowAnalysis {
 public:
  using DataFlowAnalysis::DataFlowAnalysis;

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    mlir::WalkResult result = top->walk([&](mlir::Operation *op) {
      if (failed(visit(op))) {
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return mlir::success(!result.wasInterrupted());
  }

  mlir::LogicalResult visit(mlir::ProgramPoint point) override {
    mlir::Operation *op = point.get<mlir::Operation *>();
    mlir::Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(op->getResult(0));
      propagateIfChanged(constant, constant->join(mlir::dataflow::ConstantValue(
          value, op->getDialect())));
      return mlir::success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (mlir::Region &region : op->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        setAllToUnknownConstants(block.getArguments());
      }
    }
    return mlir::success();
  }
 private:
  // set all given values as not constants.
  void setAllToUnknownConstants(mlir::ValueRange values) {
    mlir::dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (mlir::Value value : values) {
      auto *constant =
          getOrCreate<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};

// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<mlir::DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<mlir::DataFlowSolver>();
  solver->load<mlir::dataflow::DeadCodeAnalysis>();
  solver->load<tritoncc::ConstantAnalysis>();
  return solver;
}

struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  llvm::DenseSet<mlir::Operation *> set;
  std::deque<mlir::Operation *> deque;

  bool push_back(mlir::Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  mlir::Operation *pop_front() {
    mlir::Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

struct DFSState {
  DFSState(const llvm::SetVector<mlir::Operation *> &set) : toSort(set), seen() {}
  const llvm::SetVector<mlir::Operation *> &toSort;
  llvm::SmallVector<mlir::Operation *, 16> topologicalCounts;
  llvm::DenseSet<mlir::Operation *> seen;

  void addToReadyQueue(mlir::Operation *op, DFSSubgraphState &subGraph,
      llvm::SmallVector<mlir::Operation *, 4> &readyQueue) {
    bool ready = true;
    for (mlir::Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    mlir::Operation *parent = op->getParentOp();
    while (parent) {
      if (!seen.count(parent)) {
        subGraph.push_back(parent);
        ready = false;
      } 
      parent = parent->getParentOp();
    }
    if (ready) {
      readyQueue.push_back(op);
    }
  }
};

void dfsPostorder(mlir::Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  llvm::SmallVector<mlir::Operation *> ops;
  while (!subGraph.empty()) {
    llvm::SmallVector<mlir::Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      mlir::Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second) {
        continue;
      }
      ops.push_back(current);
      for (mlir::Value result : current->getResults()) {
        for (mlir::Operation *op : result.getUsers()) {
          state->addToReadyQueue(op, subGraph, readyQueue);
        }
      }
      for (mlir::Region &region : current->getRegions()) {
        for (mlir::Operation &op : region.getOps()) {
          state->addToReadyQueue(&op, subGraph, readyQueue);
        }
      }
    }
  }

  for (mlir::Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0) {
      state->topologicalCounts.push_back(op);
    }
  }
}

// Multi-root DAG topological sort.
// Performs a topological sort of the Operation in the `toSort` SetVector.
// Returns a topologically sorted SetVector.
// It is faster than mlir::topologicalSort because it prunes nodes that have
// been visited before.
llvm::SetVector<mlir::Operation *>
multiRootTopologicalSort(const llvm::SetVector<mlir::Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  llvm::SetVector<mlir::Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

// This uses the topologicalSort above
llvm::SetVector<mlir::Operation *>
multiRootGetSlice(mlir::Operation *op,
    mlir::TransitiveFilter backwardFilter = nullptr,
    mlir::TransitiveFilter forwardFilter = nullptr) {
  llvm::SetVector<mlir::Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  llvm::SetVector<mlir::Operation *> backwardSlice;
  llvm::SetVector<mlir::Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = slice[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    mlir::getBackwardSlice(currentOp, &backwardSlice, opt);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    mlir::getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

// This class represents a call graph for a given ModuleOp and holds
// data of type T associated with each FunctionOpInterface.
template <typename T>
class CallGraph {
 public:
  using FuncDataMapT = llvm::DenseMap<mlir::FunctionOpInterface, T>;

  // Constructor that builds the call graph for the given moduleOp. 
  explicit CallGraph(mlir::ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  llvm::SmallVector<mlir::FunctionOpInterface> getRoots() const {
    return roots;
  }

  // Retrieves the data associated with a function
  T *getFuncData(mlir::FunctionOpInterface funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  // walks the call graph and applies the provided update functions
  // to the edges and nodes.
  template <mlir::WalkOrder UpdateEdgeOrder = mlir::WalkOrder::PreOrder,
            mlir::WalkOrder UpdateNodeOrder = mlir::WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void walk(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    llvm::DenseSet<mlir::FunctionOpInterface> visited;
    for (auto root : roots) {
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
          updateNodeFn);
    }
  }

 private:
  void build() {
    mlir::SymbolTableCollection symbolTable;
    llvm::DenseSet<mlir::FunctionOpInterface> visited;
    // Build graph
    moduleOp.walk([&](mlir::Operation *op) {
      auto caller = op->getParentOfType<mlir::FunctionOpInterface>();
      if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
        auto *callee = callOp.resolveCallable(&symbolTable);
        auto funcOp = llvm::dyn_cast_or_null<mlir::FunctionOpInterface>(callee);
        if (funcOp) {
          graph[caller].emplace_back(
            std::pair<mlir::CallOpInterface, mlir::FunctionOpInterface>(callOp, funcOp));
          visited.insert(funcOp);
        }
      }
    });
    // Find roots
    moduleOp.walk([&](mlir::FunctionOpInterface funcOp) {
      if (!visited.count(funcOp)) {
        roots.push_back(funcOp);
      }
    });
  }

  template <mlir::WalkOrder UpdateEdgeOrder = mlir::WalkOrder::PreOrder,
            mlir::WalkOrder UpdateNodeOrder = mlir::WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doWalk(mlir::FunctionOpInterface funcOp,
              llvm::DenseSet<mlir::FunctionOpInterface> &visited,
              UpdateEdgeFn updateEdgeFn,
              UpdateNodeFn updateNodeFn) {
    if (visited.count(funcOp)) {
      llvm::report_fatal_error("Cycle detected in call graph");
    }
    if constexpr (UpdateNodeOrder == mlir::WalkOrder::PreOrder) {
      updateNodeFn(funcOp);
    }
    for (auto [callOp, callee] : graph[funcOp]) {
      if constexpr (UpdateEdgeOrder == mlir::WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(callee, visited, updateEdgeFn, updateNodeFn);
      if constexpr (UpdateEdgeOrder == mlir::WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
    }
    if constexpr (UpdateNodeOrder == mlir::WalkOrder::PostOrder) {
      updateNodeFn(funcOp);
    }
    visited.erase(funcOp);
  }
 protected:
  mlir::ModuleOp moduleOp;
  FuncDataMapT funcMap;
  llvm::DenseMap<mlir::FunctionOpInterface,
      llvm::SmallVector<std::pair<mlir::CallOpInterface, mlir::FunctionOpInterface>>> graph;
  llvm::SmallVector<mlir::FunctionOpInterface> roots;
};

}

#endif
