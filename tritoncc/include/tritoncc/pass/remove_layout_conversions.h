#pragma once

#include "llvm/Support/Debug.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0

#if USE_TRITON
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#endif

#include "tritoncc/util.h"

#define DEBUG_TYPE "remove-layout-conversions"

namespace tritoncc {

// return true if the op is an op with a layout we don't want to change. We
// will propagate the layout starting from anchor ops.
bool isLayoutAnchor(mlir::Operation *op) {
  if (llvm::isa<mlir::_tritoncc::LoadOp, mlir::_tritoncc::StoreOp>(op)) {
    return tritoncc::isExpensiveLoadOrStore(op); 
  }

  if (llvm::isa<mlir::_tritoncc::DotOp, mlir::_tritoncc::AtomicRMWOp, mlir::_tritoncc::AtomicCASOp>(op)) {
    return true;
  }

  // a heuristic
  if (mlir::_tritoncc::ReshapeOp reshape = llvm::dyn_cast<mlir::_tritoncc::ReshapeOp>(op)) {
    return reshape.getAllowReorder();
  }
  return false;
}

class LayoutPropagation {
 public:
  struct LayoutInfo {
    LayoutInfo(mlir::Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() { }
    llvm::SmallSetVector<mlir::Attribute, 8> encodings;
  };
  explicit LayoutPropagation(mlir::_tritoncc::FuncOp F) : funcOp(F) { }

  void initAnchorLayout() {
    auto maybeAddAnchor = [&](mlir::Value v) {
      if (mlir::RankedTensorType tensorType = v.getType().dyn_cast<mlir::RankedTensorType>()) {
        assert(!tensorType.getEncoding().isa<mlir::_tritoncc::gpu::NvidiaMmaEncodingAttr>() && "mma layout not supported yet");
        layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
      }
    };

    // add function args as anchors to ease writing tests.
    for (mlir::Value arg : funcOp.getArguments()) {
      maybeAddAnchor(arg);
    }

    funcOp.walk([&](mlir::Operation *op) {
      if (isLayoutAnchor(op)) {
        for (mlir::Value result : op->getResults()) {
          maybeAddAnchor(result);
        }
      }
    });
  }

  void setEncoding(mlir::ValueRange values, LayoutInfo &info, llvm::SmallVector<mlir::Value> &changed, mlir::Operation *op) {
    for (mlir::Value value : values) {
      if (!value.getType().isa<mlir::RankedTensorType>()) {
        continue;
      }
      bool hasChanged = false;
      for (mlir::Attribute encoding : info.encodings) {
        std::optional<mlir::Attribute> dstEncoding;
        if (llvm::isa<mlir::_tritoncc::gpu::ConvertLayoutOp>(op)) {
          dstEncoding = encoding;
        } else {
          dstEncoding = tritoncc::inferDstEncoding(op, encoding);
        }
        if (dstEncoding) {
          hasChanged |= layouts[value].encodings.insert(*dstEncoding);
        }
      }
      if (hasChanged) {
        changed.push_back(value);
      }
    }
  }

  // Add layouts given in `Info` to the uses of `value`. This is not transitive.
  // Return the affected users.
  llvm::SmallVector<mlir::Value> propagateToUsers(mlir::Value value, LayoutInfo &info) {
    llvm::SmallVector<mlir::Value> changed;
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();

      LLVM_DEBUG({
        llvm::dbgs() << "maybe propagateToUsers " << value << " ==> " << *user << "\n";
      });
      if (mlir::scf::ForOp forOp = llvm::dyn_cast<mlir::scf::ForOp>(user)) {
        assert(false && "ForOp not supported yet");
      }
      if (mlir::scf::WhileOp whileOp = llvm::dyn_cast<mlir::scf::WhileOp>(user)) {
        assert(false && "WhileOp not supported yet");
      }
      if (mlir::scf::YieldOp yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(user)) {
        assert(false && "YieldOp not supported yet");
      }
      if (mlir::scf::ConditionOp conditionOp = llvm::dyn_cast<mlir::scf::ConditionOp>(user)) {
        assert(false && "ConditionOp not supported yet");
      }

      if (user->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>()
        || user->hasTrait<mlir::OpTrait::Elementwise>()
        || llvm::isa<
            mlir::_tritoncc::ReduceOp,
            mlir::_tritoncc::ExpandDimsOp,
            mlir::_tritoncc::ReshapeOp,
            mlir::_tritoncc::JoinOp,
            mlir::_tritoncc::gpu::ConvertLayoutOp,
            mlir::_tritoncc::SplitOp
          >(user)
      ) {
        setEncoding(user->getResults(), info, changed, user);
        continue;
      }
    }
    return changed;
  }

  void propagateLayout() {
    llvm::SmallVector<mlir::Value> queue; // a stack more specifically
    for (auto it : layouts) {
      queue.push_back(it.first);
    }
    while (!queue.empty()) {
      mlir::Value currentValue = queue.back();
      queue.pop_back();
      LayoutInfo info = layouts[currentValue];

      LLVM_DEBUG({
        llvm::dbgs() << "propagateLayout considering " << currentValue << ", which has "
          << info.encodings.size() << " candidate encoding(s):\n";
        for (mlir::Attribute encoding : info.encodings) {
          llvm::dbgs() << "  " << encoding << "\n";
        }
      });

      llvm::SmallVector<mlir::Value> changed = propagateToUsers(currentValue, info);
      queue.insert(queue.end(), changed.begin(), changed.end());
    }
  }

  void resolveConflicts() {
    for (auto &it : layouts) {
      mlir::Operation *op = it.first.getDefiningOp();
      LayoutInfo &info = it.second;
      if (info.encodings.size() <= 1) {
        continue;
      }

      mlir::Attribute encoding = info.encodings.front();
      bool isLoadOrStore = op && llvm::isa<
        mlir::_tritoncc::LoadOp,
        mlir::_tritoncc::StoreOp,
        mlir::_tritoncc::AtomicRMWOp,
        mlir::_tritoncc::AtomicCASOp
      >(op);
      for (mlir::Attribute e : info.encodings) {
        if ((isLoadOrStore && e.isa<mlir::_tritoncc::gpu::BlockedEncodingAttr>()) ||
          (!isLoadOrStore && e.isa<mlir::_tritoncc::gpu::NvidiaMmaEncodingAttr>())) {
           encoding = e;
           break;
        }
      }
      info.encodings.clear();
      info.encodings.insert(encoding);
    }
  }

  void map(mlir::Value old, mlir::Value newV) {
    rewriteMapping[{old, newV.getType().cast<mlir::RankedTensorType>().getEncoding()}] = newV;
  }

  mlir::Operation *cloneElementwise(mlir::OpBuilder &rewriter, mlir::Operation *op, mlir::Attribute encoding) {
    mlir::Operation *newOp = rewriter.clone(*op);

    std::optional<mlir::Attribute> operandEnc;
    if (op->getNumOperands() > 0) {
      operandEnc = tritoncc::inferSrcEncoding(op, encoding);
      assert(operandEnc.has_value());
    }

    for (mlir::OpOperand &operand : op->getOpOperands()) {
      newOp->setOperand(operand.getOperandNumber(),
        getValueAs(operand.get(), *operandEnc));
    }
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      auto origType = op->getResult(i).getType().dyn_cast<mlir::RankedTensorType>();
      if (!origType) {
        continue;
      }
      auto newType = mlir::RankedTensorType::get(origType.getShape(),
        origType.getElementType(), encoding);
      newOp->getResult(i).setType(newType);
    }
    return newOp;
  }

  mlir::Operation *rewriteOp(mlir::Operation *op) {
    opToDelete.insert(op);
    if (mlir::scf::ForOp forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
      assert(false && "ForOp");
    }
    if (mlir::scf::WhileOp whileOp = llvm::dyn_cast<mlir::scf::WhileOp>(op)) {
      assert(false && "WhileOp");
    }
    if (mlir::scf::IfOp ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
      assert(false && "IfOp");
    }
    mlir::OpBuilder rewriter(op);
    mlir::Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
    if (mlir::_tritoncc::gpu::ConvertLayoutOp convertOp = llvm::dyn_cast<mlir::_tritoncc::gpu::ConvertLayoutOp>(op)) {
      mlir::Attribute srcEncoding = convertOp.getSrc().getType().getEncoding();
      auto it = layouts.find(convertOp.getSrc());
      if (it != layouts.end()) {
        srcEncoding = *it->second.encodings.begin();
      }
      mlir::Value src = getValueAs(convertOp.getSrc(), srcEncoding);
      mlir::RankedTensorType tensorType = op->getResult(0).getType().cast<mlir::RankedTensorType>();
      mlir::RankedTensorType newType = mlir::RankedTensorType::get(
        tensorType.getShape(),
        tensorType.getElementType(),
        encoding);
      mlir::_tritoncc::gpu::ConvertLayoutOp cvt = rewriter.create<mlir::_tritoncc::gpu::ConvertLayoutOp>(op->getLoc(), newType, src);
      map(op->getResult(0), cvt.getResult());
      return cvt.getOperation();
    }
    if (tritoncc::canFoldIntoConversion(op, encoding)) {
      assert(false && "canFoldIntoConversion true");
    }
    if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
        op->hasTrait<mlir::OpTrait::Elementwise>() ||
        llvm::isa<mlir::_tritoncc::ReduceOp, mlir::_tritoncc::ExpandDimsOp, mlir::_tritoncc::ReshapeOp, mlir::_tritoncc::JoinOp, mlir::_tritoncc::SplitOp, mlir::_tritoncc::gpu::ConvertLayoutOp>(op)) {
      mlir::Operation *newOp = cloneElementwise(rewriter, op, encoding);
      for (auto [oldResult, newResult] : llvm::zip(op->getResults(), newOp->getResults())) {
        map(oldResult, newResult);
      }
      return newOp;
    }
    llvm::report_fatal_error("unexpected op in rewrite");
    return nullptr;
  }

  bool reduceToScalar(mlir::Operation *op) {
    // For reductions returning a scalar we can change the src encoding without
    // affecting the output.
    return llvm::isa<mlir::_tritoncc::ReduceOp>(op) && !op->getResultTypes()[0].isa<mlir::RankedTensorType>();
  }

  mlir::Value getValueAs(mlir::Value value, mlir::Attribute encoding) {
    if (mlir::RankedTensorType tensorType = value.getType().dyn_cast<mlir::RankedTensorType>()) {
      mlir::Value rewrittenValue;
      auto layoutIt = layouts.find(value);
      if (layoutIt == layouts.end()) {
        rewrittenValue = value;
      } else {
        assert(layoutIt->second.encodings.size() == 1 && "we should have resolved to a single encoding");
        mlir::Attribute encodingPicked = *layoutIt->second.encodings.begin();
        if (encodingPicked == tensorType.getEncoding()) {
          rewrittenValue = value;
        } else {
          rewrittenValue = rewriteMapping[{value, encodingPicked}];
        }
      }
      assert(rewrittenValue);
      if (rewrittenValue.getType().cast<mlir::RankedTensorType>().getEncoding() == encoding) {
        return rewrittenValue;
      }
      mlir::OpBuilder rewriter(value.getContext());
      rewriter.setInsertionPointAfterValue(rewrittenValue);
      mlir::RankedTensorType tmpType = mlir::RankedTensorType::get(
        tensorType.getShape(),
        tensorType.getElementType(),
        encoding
      );
      mlir::Value converted = rewriter.create<mlir::_tritoncc::gpu::ConvertLayoutOp>(
        value.getLoc(), tmpType, rewrittenValue);
      return converted;
    }
    return value;
  }

  void rewriteRegion(mlir::Region &region) {
    llvm::SmallVector<mlir::Region *> queue = {&region};
    while (!queue.empty()) {
      mlir::Region *currentRegion = queue.back();
      queue.pop_back();

      for (mlir::Operation &op : currentRegion->getOps()) {
        bool needRewrite = false;
        llvm::SmallVector<mlir::Value> results = op.getResults();
        for (mlir::Value result : results) {
          auto it = layouts.find(result);
          if (it == layouts.end()) {
            continue;
          }
          LayoutInfo &info = it->second;
          assert(info.encodings.size() == 1 && "we should have resolved to a single encoding");
          mlir::Attribute encoding = result.getType().cast<mlir::RankedTensorType>().getEncoding();
          if (encoding == *info.encodings.begin()) {
            continue;
          }
          needRewrite = true;
        }
        if (needRewrite) {
          mlir::Operation *newOp = rewriteOp(&op);
          for (mlir::Region &R : newOp->getRegions()) {
            queue.push_back(&R);
          }
        } else if (mlir::scf::YieldOp yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(&op)) {
          assert(false && "YieldOp");
        } else if (mlir::scf::ConditionOp conditonOp = llvm::dyn_cast<mlir::scf::ConditionOp>(&op)) {
          assert(false && "ConditionOp");
        } else if (reduceToScalar(&op)) {
          assert(false && "rewriteReduceToScalar");
        } else if (mlir::_tritoncc::AssertOp assertOp = llvm::dyn_cast<mlir::_tritoncc::AssertOp>(&op)) {
          assert(false && "AssertOp");
        } else {
          // If we don't need to rewrite the op, we still need to remap the
          // operands.
          for (mlir::OpOperand &operand : op.getOpOperands()) {
            auto it = layouts.find(operand.get());
            if (it == layouts.end()) {
              continue;
            }
            mlir::Attribute encoding =
                operand.get().getType().cast<mlir::RankedTensorType>().getEncoding();
            mlir::Value newOperand = getValueAs(operand.get(), encoding);
            op.setOperand(operand.getOperandNumber(), newOperand);
          }
          for (mlir::Region &R : op.getRegions()) {
            queue.push_back(&R);
          }
        }
      }
    }

    for (mlir::Operation *op : llvm::reverse(opToDelete)) {
      op->erase();
    }
  }

  void rewrite() {
    rewriteRegion(funcOp->getRegion(0));
  }

 private:
  llvm::MapVector<mlir::Value, LayoutInfo> layouts;
  llvm::DenseMap<std::pair<mlir::Value, mlir::Attribute>, mlir::Value> rewriteMapping;
  llvm::SetVector<mlir::Operation *> opToDelete;
  mlir::_tritoncc::FuncOp funcOp;
};

bool canBeRemat(mlir::Operation *op) {
  if (llvm::isa<mlir::_tritoncc::LoadOp, mlir::_tritoncc::StoreOp>(op)) {
    return !tritoncc::isExpensiveLoadOrStore(op);
  }
  if (llvm::isa<mlir::tensor::ExtractSliceOp, mlir::_tritoncc::gpu::AllocTensorOp,
      mlir::_tritoncc::gpu::InsertSliceAsyncOp,
      mlir::_tritoncc::AtomicRMWOp,
      mlir::_tritoncc::AtomicCASOp,
      mlir::_tritoncc::DotOp>(op)) {
    return false;
  }
  if (llvm::isa<mlir::scf::WhileOp, mlir::scf::ConditionOp>(op)) {
    return false;
  }
  return true;
}

mlir::LogicalResult getRematerializableSlice(
    mlir::Value root, mlir::Attribute rootEncoding,
    llvm::SetVector<mlir::Value> &slice,
    llvm::DenseMap<mlir::Value, mlir::Attribute> &layout,
    std::function<bool(mlir::Operation *)> stopPropagation = nullptr) {
  mlir::LogicalResult result = tritoncc::getConvertBackwardSlice(
      root, slice, rootEncoding,
      layout, stopPropagation);
  if (result.failed() || slice.empty()) {
    return mlir::failure();
  }

  // Check if all the operations in the slice can be rematerialized.
  for (mlir::Value v : slice) {
    if (mlir::Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op)) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

void rewriteSlice(llvm::SetVector<mlir::Value> &slice, llvm::DenseMap<mlir::Value, mlir::Attribute> &layout, mlir::_tritoncc::gpu::ConvertLayoutOp convertOp, mlir::IRMapping &mapping) {
  llvm::SetVector<mlir::Operation *> opsToRewrite;
  for (mlir::Value v : slice) {
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<mlir::scf::IfOp>()) {
        assert(false && "IfOp");
      }
    } else {
      assert(false && "BlockArgument");
    }
  }
  opsToRewrite = multiRootTopologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite
  // This is required for slicecount(value) to work mid rewrite.
  llvm::SmallVector<std::tuple<mlir::Value, mlir::Value>> replacements;

  llvm::SmallVector<mlir::Operation *> deadOps;
  mlir::IRRewriter builder(slice.begin()->getContext());
  for (mlir::Operation *op : opsToRewrite) {
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
      assert(false && "ForOp");
    }
    if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
      assert(false && "IfOp");
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
      assert(false && "YieldOp");
    }
    if (llvm::isa<mlir::arith::ConstantOp>(op)) {
      assert(false && "ConstantOp");
    }
    mlir::Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end()) {
        continue;
      }
      auto newType = mlir::RankedTensorType::get(
        old.getType().cast<mlir::RankedTensorType>().getShape(),
        old.getType().cast<mlir::RankedTensorType>().getElementType(),
        it->second);
      newV.setType(newType);
    }
  }
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));
  convertOp.erase();

  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  for (mlir::Operation *op : deadOps) {
    op->erase();
  }
}

void rewriteSlice(llvm::SetVector<mlir::Value> &slice, llvm::DenseMap<mlir::Value, mlir::Attribute> &layout, mlir::_tritoncc::gpu::ConvertLayoutOp convertOp) {
  mlir::IRMapping mapping;
  rewriteSlice(slice, layout, convertOp, mapping);
}

void backwardRematerialization(mlir::_tritoncc::gpu::ConvertLayoutOp convertOp) {
  if (tritoncc::hasSharedEncoding(convertOp.getResult()) ||
      tritoncc::hasSharedEncoding(convertOp.getSrc())) {
    return;
  }
  mlir::RankedTensorType targetType = convertOp.getType();
  if (targetType.getEncoding().isa<mlir::_tritoncc::gpu::DotOperandEncodingAttr>()) {
    return;
  }

  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  llvm::SetVector<mlir::Value> slice;
  llvm::DenseMap<mlir::Value, mlir::Attribute> layout;
  mlir::LogicalResult result = getRematerializableSlice(
    convertOp.getSrc(), targetType.getEncoding(), slice, layout);
  if (result.failed()) {
    return;
  }

  // 2. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp);
}

void backwardRematerialization(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::_tritoncc::gpu::ConvertLayoutOp> convertOps;
  module.walk(
    [&](mlir::_tritoncc::gpu::ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (auto convertOp : convertOps) {
    backwardRematerialization(convertOp);
  }
}

void hoistConvertOnTopOfExtOrBroadcast(mlir::_tritoncc::gpu::ConvertLayoutOp convertOp) {
  // we don't want to rematerialize any convertion to/from shared
  if (hasSharedEncoding(convertOp.getResult()) ||
      hasSharedEncoding(convertOp.getSrc())) {
    return;
  }
  mlir::RankedTensorType targetType = convertOp.getType();
  if (targetType.getEncoding().isa<mlir::_tritoncc::gpu::DotOperandEncodingAttr>()) {
    return;
  }

  auto isExtOrBroadcastOp = [](mlir::Operation *op) {
    return llvm::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
        mlir::arith::ExtFOp, mlir::_tritoncc::BroadcastOp,
        mlir::_tritoncc::ExpandDimsOp>(op);
  };
  // 1. Take a backward slice of all the tensor dependencies.
  llvm::SetVector<mlir::Value> slice;
  llvm::DenseMap<mlir::Value, mlir::Attribute> layout;
  mlir::LogicalResult result
      = tritoncc::getRematerializableSlice(convertOp.getSrc(), targetType.getEncoding(), slice, layout, isExtOrBroadcastOp);
  if (result.failed()) {
    return;
  }

  mlir::Operation *extOrBroadcastOp = nullptr;
  unsigned sliceSize = slice.size();

  for (unsigned i = 0; i < sliceSize; ++i) {
    mlir::Value v = slice[i];
    mlir::Operation *op = v.getDefiningOp();
    if (!op) {
      continue;
    }
    if (isExtOrBroadcastOp(op)) {
      assert(false && "isExtOrBroadcastOp");
    }
  }
  if (extOrBroadcastOp == nullptr) {
    return;
  }
  assert(false && "hoistConvertOnTopOfExtOrBroadcast");
}

void hoistConvert(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::_tritoncc::gpu::ConvertLayoutOp> convertOps;
  module.walk(
    [&](mlir::_tritoncc::gpu::ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (mlir::_tritoncc::gpu::ConvertLayoutOp convertOp : convertOps) {
    hoistConvertOnTopOfExtOrBroadcast(convertOp);
  }
}

// dot(a, b, load(ptr)) -> add(load(ptr), dot(a, b, 0))
class ConvertDotConvert : public mlir::RewritePattern {
 public:
  explicit ConvertDotConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::_tritoncc::gpu::ConvertLayoutOp::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
      mlir::PatternRewriter &rewriter) const override {
    auto dstOp = llvm::cast<mlir::_tritoncc::gpu::ConvertLayoutOp>(op);
    auto dotOp = dstOp.getSrc().getDefiningOp<mlir::_tritoncc::DotOp>();
    if (!dotOp) {
      return mlir::failure();
    }
    assert(false && "ConvertDotConvert matchAndRewrite");
  }
};

class RemoveLayoutConversionsPass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit RemoveLayoutConversionsPass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<RemoveLayoutConversionsPass>()) { }

  llvm::StringRef getName() const override {
    return "RemoveLayoutConversionsPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    assert(false && "clonePass nyi");
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();

    LLVM_DEBUG({
      llvm::dbgs() << "RemoveLayoutConversionsPass got input module " << moduleOp << "\n";
    });

    // 1
    moduleOp.walk([](mlir::_tritoncc::FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    LLVM_DEBUG({
      llvm::dbgs() << "Module after propagating layouts forward:\n";
      moduleOp.dump();
    });

    mlir::RewritePatternSet cleanUpPatterns(context);
    mlir::_tritoncc::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (mlir::applyPatternsAndFoldGreedily(moduleOp, std::move(cleanUpPatterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Module after canonicalizing:\n";
      moduleOp.dump();
    });

    // 2
    backwardRematerialization(moduleOp);
    LLVM_DEBUG({
      llvm::dbgs() << "Module after backward remat:\n";
      moduleOp.dump();
    });

    // 3. For remaining converts, try to hoist them above cast generating larger
    // size types in order to reduce the cost of the convert op.
    hoistConvert(moduleOp);
    LLVM_DEBUG({
      llvm::dbgs() << "Module after hoisting converts:\n";
      moduleOp.dump();
    });

    mlir::RewritePatternSet decomposePatterns(context);
    decomposePatterns.add<ConvertDotConvert>(context);
    if (mlir::applyPatternsAndFoldGreedily(moduleOp, std::move(decomposePatterns)).failed()) {
      signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Module after decomposing dot-converts:\n";
      moduleOp.dump();
    });

    // 4. Apply clean up patterns to remove dead convert and dead code
    // generated by the previous transformations.
    mlir::RewritePatternSet cleanUpPatterns2(context);
    populateForOpDeadArgumentElimination(cleanUpPatterns2);
    mlir::scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    mlir::scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    mlir::_tritoncc::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    if (mlir::applyPatternsAndFoldGreedily(moduleOp, std::move(cleanUpPatterns2)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Module after final cleanups:\n";
      moduleOp.dump();
    });
  }
};

#if USE_TRITON
static std::unique_ptr<mlir::Pass> createRemoveLayoutConversionsPass() {
  return mlir::triton::gpu::createRemoveLayoutConversionsPass();
}
#else
static std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createRemoveLayoutConversionsPass() {
  return std::make_unique<RemoveLayoutConversionsPass>();
}
#endif

}

#undef DEBUG_TYPE
