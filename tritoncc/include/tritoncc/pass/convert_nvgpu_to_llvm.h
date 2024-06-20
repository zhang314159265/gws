#pragma once

namespace tritoncc {

const std::string Cluster_Cta_Id_Op =
  "{\n"
  ".reg .u32 a<5>;\n"
  "mov.u32 a0, %cluster_ctaid.x;\n" // x
  "mov.u32 a1, %cluster_ctaid.y;\n" // y
  "mov.u32 a2, %cluster_ctaid.z;\n" // z
  "mov.u32 a3, %cluster_nctaid.x;\n" // nx
  "mov.u32 a4, %cluster_nctaid.y;\n" // ny
  "mad.lo.u32 a1, a2, a4, a1;\n"
  "mad.lo.u32 $0, a1, a3, a0;\n"
  "}";

using OperandsAndConstraints = std::vector<std::pair<mlir::Value, std::string>>;
typedef std::vector<std::string> Constraints;

template <typename SourceOp, typename ConcreteT>
class NVGPUOpPatternBase : public mlir::RewritePattern {
 public:
  explicit NVGPUOpPatternBase(mlir::MLIRContext *context)
      : mlir::RewritePattern(SourceOp::getOperationName(), 1, context) {}

  llvm::SmallVector<mlir::triton::PTXBuilder::Operand *>
  getPtxOutputs(std::vector<std::string> &outputConstraints,
      mlir::triton::PTXBuilder &ptxBuilder) const {
    llvm::SmallVector<mlir::triton::PTXBuilder::Operand *> ptxOutputs;
    for (unsigned i = 0; i < outputConstraints.size(); ++i) {
      auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
      ptxOutputs.push_back(ptxOutput);
    }
    return ptxOutputs;
  }

  OperandsAndConstraints
  unpackOperands(OperandsAndConstraints &operandsAndConstraints,
      mlir::triton::PTXBuilder &ptxBuilder, mlir::Location &loc,
      mlir::PatternRewriter &rewriter) const {
    OperandsAndConstraints unpackedOperands;
    for (auto &[operand, constraint] : operandsAndConstraints) {
      auto llvmStruct = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(operand.getType());
      if (llvmStruct) {
        assert(false && "unpackOperands");
      } else {
        unpackedOperands.push_back({operand, constraint});
      }
    }
    return unpackedOperands;
  }

  llvm::SmallVector<mlir::triton::PTXBuilder::Operand *>
  getPtxOperands(OperandsAndConstraints &operandsAndConstraints,
      mlir::triton::PTXBuilder &ptxBuilder, mlir::Location &loc,
      mlir::PatternRewriter &rewriter) const {
    llvm::SmallVector<mlir::triton::PTXBuilder::Operand *> ptxOperands;
    auto unpackedOperandsAndConstraints =
        unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
    for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
      assert(false && "getPtxOperands");
    }
    return ptxOperands;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
      mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto sourceOp = llvm::dyn_cast<SourceOp>(op);
    if (!sourceOp) {
      return mlir::failure();
    }
    auto concrete = static_cast<const ConcreteT *>(this);
    auto ptxAsm = concrete->getPtxAsm(sourceOp);
    auto ptxAsmPatched = patchPtxAsm(sourceOp, ptxAsm);
    auto hasSideEffects = !mlir::isMemoryEffectFree(sourceOp);
    auto operandsAndConstraints = concrete->getOperandsAndConstraints(sourceOp);
    auto outputConstraints = concrete->getOutputConstraints(sourceOp);

    mlir::triton::PTXBuilder ptxBuilder;
    auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
    auto ptxOperands =
        getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
    llvm::SmallVector<mlir::triton::PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
    outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
    auto &ptxInstr = *ptxBuilder.create<mlir::triton::PTXInstr>(ptxAsmPatched);
    ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
    auto retTy =
        op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
    auto res = ptxBuilder.launch(rewriter, loc, retTy,
        /*hasSideEffects*/hasSideEffects);
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, res);
    }

    return mlir::success();
  }

  std::string patchPtxAsm(mlir::Operation *op, std::string ptxAsm) const {
    std::vector<std::pair<int, int>> patchLocations;
    std::vector<std::string> patchValues;
    auto start = ptxAsm.find("#", 0);
    while (start != std::string::npos) {
      assert(false && "found '#'");
    }
    assert(patchLocations.size() == patchValues.size());
    if (patchLocations.size() == 0) {
      return ptxAsm;
    }
    assert(false && "patchPtxAsm");
  }
};

template <typename SourceOp>
class NVGPUOpGenericPattern
    : public NVGPUOpPatternBase<SourceOp, NVGPUOpGenericPattern<SourceOp>> {
 public:
  explicit NVGPUOpGenericPattern(mlir::MLIRContext *context, std::string ptxAsm,
      std::vector<std::string> outputConstraints,
      std::vector<std::string> inputConstraints)
    : NVGPUOpPatternBase<SourceOp, NVGPUOpGenericPattern<SourceOp>>(context),
      ptxAsm(ptxAsm), outputConstraints(outputConstraints),
      inputConstraints(inputConstraints) {}

  std::string getPtxAsm(SourceOp op) const { return ptxAsm; }

  std::vector<std::string> getOutputConstraints(SourceOp op) const {
    return outputConstraints;
  }

  OperandsAndConstraints getOperandsAndConstraints(SourceOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); ++i) {
      operandsAndConstraints.push_back(
        {op->getOperand(i), inputConstraints[i]});
    }
    return operandsAndConstraints;
  }

 private:
  std::string ptxAsm;
  std::vector<std::string> outputConstraints;
  std::vector<std::string> inputConstraints;
};

class ConvertNVGPUToLLVM : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  ConvertNVGPUToLLVM() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<ConvertNVGPUToLLVM>()) {}

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);

    patterns.add<NVGPUOpGenericPattern<mlir::_tritoncc::nvgpu::ClusterCTAIdOp>>(
      context, Cluster_Cta_Id_Op, Constraints({"=r"}), Constraints());

    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }

  llvm::StringRef getName() const override { return "ConvertNVGPUToLLVM"; }
  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertNVGPUToLLVM>(*this);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertNVGPUToLLVMPass() {
  return std::make_unique<ConvertNVGPUToLLVM>();
}

}
