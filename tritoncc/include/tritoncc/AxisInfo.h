#pragma once

#define USE_TRITON 0

#if USE_TRITON
#include "triton/Analysis/AxisInfo.h"
namespace tritoncc {
using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::AxisInfo;
}
#else

#include "triton/Analysis/Utility.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "tritoncc/util.h"

namespace tritoncc {

int64_t gcd(int64_t a, int64_t b) {
  int64_t r;
  while (b) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

// if lhs * rhs overflows, return max possible value for the type
int64_t multiplyDivisor(int64_t lhs, int64_t rhs) {
  int64_t maxDivisor = highestPowOf2Divisor<int64_t>(0);
  if (lhs > maxDivisor / rhs) {
    return maxDivisor;
  }
  return lhs * rhs;
}

class AxisInfo {
 public:
  typedef llvm::SmallVector<int64_t> DimVectorT;

  AxisInfo() : AxisInfo({}, {}, {}) {}

  AxisInfo(DimVectorT contiguity, DimVectorT divisibility, DimVectorT constancy)
      : AxisInfo(contiguity, divisibility, constancy, std::nullopt) {}

  AxisInfo(DimVectorT contiguity, DimVectorT divisibility, DimVectorT constancy,
          std::optional<int64_t> constantValue)
      : contiguity(contiguity), divisibility(divisibility),
        constancy(constancy), constantValue(constantValue) {
    assert(divisibility.size() == contiguity.size());
    assert(constancy.size() == contiguity.size());
  }

  int64_t getContiguity(size_t dim) const {
    return contiguity[dim];
  }

  const DimVectorT &getContiguity() const {
    return contiguity;
  }

  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int64_t getConstancy(size_t dim) const { return constancy[dim]; }
  const DimVectorT &getConstancy() const { return constancy; }

  void print(llvm::raw_ostream &os) const {
    assert(false && "print");
  }

  int getRank() const { return contiguity.size(); }
  std::optional<int64_t> getConstantValue() const { return constantValue; }

  // The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs) {
    // If one argument is not initialized, return the other.
    if (lhs.getRank() == 0) {
      return rhs;
    }
    if (rhs.getRank() == 0) {
      return lhs;
    }
    DimVectorT contiguity;
    DimVectorT divisibility;
    DimVectorT constancy;
    for (auto d = 0; d < lhs.getRank(); ++d) {
      contiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
      divisibility.push_back(gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
      constancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
    }
    std::optional<int64_t> constantValue;
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value() &&
        lhs.getConstantValue() == rhs.getConstantValue()) {
      constantValue = lhs.getConstantValue();
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }

  bool operator==(const AxisInfo &other) const {
    return contiguity == other.contiguity &&
      divisibility == other.divisibility && constancy == other.constancy &&
      constantValue == other.constantValue;
  }

  static AxisInfo getPessimisticValueState(mlir::Value value) {
    auto rank = 1;
    if (mlir::TensorType ty = value.getType().dyn_cast<mlir::TensorType>()) {
      rank = ty.getRank();
    }
    if (mlir::triton::PointerType ty = value.getType().dyn_cast<mlir::triton::PointerType>()) {
      if (mlir::TensorType elemTy = ty.getPointeeType().dyn_cast<mlir::TensorType>()) {
        rank = elemTy.getRank();
      }
    }

    DimVectorT knownContiguity(rank, 1);
    DimVectorT knownDivisibility(rank, 1);
    DimVectorT knownConstancy(rank, 1);

    mlir::BlockArgument blockArg = value.dyn_cast<mlir::BlockArgument>();

    if (blockArg && blockArg.getOwner()->isEntryBlock()) {
      mlir::Operation *op = blockArg.getOwner()->getParentOp();
      if (auto fun = llvm::dyn_cast<mlir::FunctionOpInterface>(op)) {
        initPessimisticStateFromFunc(blockArg.getArgNumber(), fun,
            &knownContiguity, &knownDivisibility,
            &knownConstancy);
      // llvm codegen check alignment to generate vector load/store
      // would be nice if this want's the case
      } else if (auto fun = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
        initPessimisticStateFromFunc(blockArg.getArgNumber(), fun,
            &knownContiguity, &knownDivisibility,
            &knownConstancy);
      }
    } else if (mlir::Operation *op = value.getDefiningOp()) {
      if (mlir::Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
        auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
        knownDivisibility = DimVectorT(vals.begin(), vals.end());
      }
      if (mlir::Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
        auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
        knownContiguity = DimVectorT(vals.begin(), vals.end());
      }
      if (mlir::Attribute attr = op->getDiscardableAttr("tt.constancy")) {
        auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
        knownConstancy = DimVectorT(vals.begin(), vals.end());
      }
    }

    return AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
  }

  template <typename T>
  static void initPessimisticStateFromFunc(int argNumber, T funcOp,
      DimVectorT *contiguity,
      DimVectorT *divisibility,
      DimVectorT *constancy) {
    // list of attributes that we care about
    llvm::SmallVector<std::pair<DimVectorT *, std::string>> retVecs;
    retVecs.push_back({contiguity, "tt.contiguity"});
    retVecs.push_back({divisibility, "tt.divisibility"});
    retVecs.push_back({constancy, "tt.constancy"});
    for (auto [vec, attrName] : retVecs) {
      mlir::Attribute attr = funcOp.getArgAttr(argNumber, attrName);
      if (auto int_attr = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
        *vec = DimVectorT(contiguity->size(), int_attr.getValue().getZExtValue());
      }
      if (auto dense_attr = attr.dyn_cast_or_null<mlir::DenseElementsAttr>()) {
        auto vals = dense_attr.getValues<int>();
        *vec = DimVectorT(vals.begin(), vals.end());
      }
    }
  }
 private:
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;

  // The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;
};

class AxisInfoVisitor {
 public:
  AxisInfoVisitor() = default;
  virtual ~AxisInfoVisitor() = default;
  virtual bool match(mlir::Operation *op) = 0;

  virtual AxisInfo getAxisInfo(mlir::Operation *op, llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) = 0;

  static bool isConstantDim(const AxisInfo &info, llvm::ArrayRef<int64_t> shape,
      int dim) {
    return info.getConstancy(dim) == shape[dim];
  }
};

// Base class for all operations
template <typename OpTy>
class AxisInfoVisitorImpl : public AxisInfoVisitor {
 public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(mlir::Operation *op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) final {
    return getAxisInfo(llvm::cast<OpTy>(op), operands);
  }

  bool match(mlir::Operation *op) final { return llvm::isa<OpTy>(op); }

  virtual AxisInfo
  getAxisInfo(OpTy op, llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) {
    llvm_unreachable("Unimplemented getAxisInfo");
  }
};

class BroadcastOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<mlir::triton::BroadcastOp> {
 public:
  using AxisInfoVisitorImpl<mlir::triton::BroadcastOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::triton::BroadcastOp op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    mlir::Type _retTy = *op->result_type_begin();
    mlir::Type _opTy = *op->operand_type_begin();
    mlir::TensorType retTy = _retTy.cast<mlir::TensorType>();
    mlir::TensorType opTy = _opTy.cast<mlir::TensorType>();
    llvm::ArrayRef<int64_t> retShape = retTy.getShape();
    llvm::ArrayRef<int64_t> opShape = opTy.getShape();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;

    for (int d = 0; d < retTy.getRank(); ++d) {
      contiguity.push_back(opShape[d] == 1 ? 1 : opInfo.getContiguity(d));
      divisibility.push_back(opInfo.getDivisibility(d));
      constancy.push_back(opShape[d] == 1 ? retShape[d]
                                          : opInfo.getConstancy(d));
    }
    return AxisInfo(contiguity, divisibility, constancy,
        operands[0]->getValue().getConstantValue());
  }
};

class ExpandDimsOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<mlir::triton::ExpandDimsOp> {
 public:
  using AxisInfoVisitorImpl<mlir::triton::ExpandDimsOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::triton::ExpandDimsOp op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity = opInfo.getContiguity();
    AxisInfo::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfo::DimVectorT constancy = opInfo.getConstancy();
    int64_t newDivisibility = 1;
    if (opInfo.getConstantValue().has_value()) {
      newDivisibility = highestPowOf2Divisor(opInfo.getConstantValue().value());
    } else if (opInfo.getRank()) {
      newDivisibility =
        opInfo.getContiguity(0) > 1 ? 1 : opInfo.getDivisibility(0);
      for (int d = 1; d < opInfo.getRank(); ++d) {
        newDivisibility = gcd(newDivisibility,
              opInfo.getContiguity(d) > 1 ? 1 : opInfo.getDivisibility(d));
      }
    }
    contiguity.insert(contiguity.begin() + op.getAxis(), 1);
    divisibility.insert(divisibility.begin() + op.getAxis(), newDivisibility);
    constancy.insert(constancy.begin() + op.getAxis(), 1);
    return AxisInfo(contiguity, divisibility, constancy,
        operands[0]->getValue().getConstantValue());
  }
};

class SplatOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<mlir::triton::SplatOp> {
 public:
  using AxisInfoVisitorImpl<mlir::triton::SplatOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::triton::SplatOp op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    mlir::Type _retTy = *op->result_type_begin();
    mlir::TensorType retTy = _retTy.cast<mlir::TensorType>();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (int d = 0; d < retTy.getRank(); ++d) {
      contiguity.push_back(1);
      divisibility.push_back(opInfo.getDivisibility(0));
      constancy.push_back(retTy.getShape()[d]);
    }
    return AxisInfo(contiguity, divisibility, constancy,
        operands[0]->getValue().getConstantValue());
  }
};

// Binary operations
template <typename OpTy>
class BinaryOpVisitorImpl : public AxisInfoVisitorImpl<OpTy> {
 public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    assert(operands.size() == 2 && "Expected two operands");
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    auto constantValue = getConstantValue(op, lhsInfo, rhsInfo);
    for (auto d = 0; d < rank; ++d) {
      if (constantValue.has_value()) {
        contiguity.push_back(1);
        constancy.push_back(
          std::max(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
        divisibility.push_back(
          highestPowOf2Divisor<int64_t>(constantValue.value()));
      } else {
        contiguity.push_back(getContiguity(op, lhsInfo, rhsInfo, d));
        constancy.push_back(getConstancy(op, lhsInfo, rhsInfo, d));
        divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, d));
      }
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }
 protected:
  virtual int64_t getContiguity(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getDivisibility(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getConstancy(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs) {
    return {};
  }
};

template <typename OpTy>
class AddSubOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
 public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;
 private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
      int dim) override {
    // ?
    return std::max(gcd(lhs.getConstancy(dim), rhs.getContiguity(dim)),
        gcd(lhs.getContiguity(dim), rhs.getConstancy(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
      int dim) override {
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if constexpr (std::is_same_v<OpTy, mlir::triton::AddPtrOp>) {
      auto elemSize = std::max<int64_t>(
        1, mlir::triton::getPointeeBitWidth(op.getPtr().getType()) / 8);
      rhsDivisibility = multiplyDivisor(rhs.getDivisibility(dim), elemSize);
    }
    return gcd(lhs.getDivisibility(dim), rhsDivisibility);
  }

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
      int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      assert(false && "getConstantValue");
    }
    return {};
  }
};

class MulIOpAxisInfoVisitor final : public BinaryOpVisitorImpl<mlir::arith::MulIOp> {
 public:
  using BinaryOpVisitorImpl<mlir::arith::MulIOp>::BinaryOpVisitorImpl;

 private:
  int64_t getContiguity(mlir::arith::MulIOp op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) override {
    auto lhsContiguity =
        rhs.getConstantValue().has_value() && rhs.getConstantValue() == 1
        ? lhs.getContiguity(dim)
        : 1;
    auto rhsContiguity =
        lhs.getConstantValue().has_value() && lhs.getConstantValue() == 1
        ? rhs.getContiguity(dim)
        : 1;
    return std::max(lhsContiguity, rhsContiguity);
  }

  int64_t getDivisibility(mlir::arith::MulIOp op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) override {
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 &&
        !(rhs.getConstantValue().has_value() && rhs.getConstantValue() == 1)) {
      lhsDivisibility = 1;
    }
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if (rhs.getContiguity(dim) > 1 &&
        !(lhs.getConstantValue().has_value() && lhs.getConstantValue() == 1)) {
      rhsDivisibility = 1;
    }
    return multiplyDivisor(lhsDivisibility, rhsDivisibility);
  }

  int64_t getConstancy(mlir::arith::MulIOp op, const AxisInfo &lhs,
      const AxisInfo &rhs, int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  std::optional<int64_t> getConstantValue(mlir::arith::MulIOp op, const AxisInfo &lhs,
      const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      return {lhs.getConstantValue().value() * rhs.getConstantValue().value()};
    }
    return {};
  }
};

template <typename OpTy>
class ConstantOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
 public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op, llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    auto intAttr = op.getValue().template dyn_cast<mlir::IntegerAttr>();
    auto boolAttr = op.getValue().template dyn_cast<mlir::BoolAttr>();
    if (intAttr || boolAttr) {
      int64_t value{};
      if (intAttr) {
        value = intAttr.getValue().getZExtValue();
      } else {
        value = boolAttr.getValue() ? 1 : 0;
      }
      return AxisInfo(/*contiguity=*/{1},
                      /*divisibility=*/{highestPowOf2Divisor(value)},
                      /*constancy=*/{1},
                      /*knownConstantValue=*/{value});
    }
    assert(false && "getAxisInfo");
  }
};

template <typename OpTy>
class CmpOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
 public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    auto resTy = op.getType().template dyn_cast<mlir::RankedTensorType>();
    if (!resTy) {
      return AxisInfo();
    }
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      int64_t constHint = 1;
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        assert(false && "constant value");
      } else {
        constHint = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
        if ((gtPredicate(getPredicate(op)) || lePredicate(getPredicate(op))) &&
            AxisInfoVisitor::isConstantDim(lhsInfo, shape, d)) {
          constHint = std::max(constHint, gcd(rhsInfo.getContiguity(d),
              gcd(lhsInfo.getDivisibility(d),
                  rhsInfo.getDivisibility(d))));
        } else if ((ltPredicate(getPredicate(op)) ||
                    gePredicate(getPredicate(op))) &&
                   AxisInfoVisitor::isConstantDim(rhsInfo, shape, d)) {
          constHint = std::max(constHint, gcd(lhsInfo.getContiguity(d),
              gcd(lhsInfo.getDivisibility(d),
                  rhsInfo.getDivisibility(d))));
        }
      }
      constancy.push_back(constHint);
      divisibility.push_back(1);
      contiguity.push_back(1);
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }
 private:
  static mlir::arith::CmpIPredicate getPredicate(mlir::arith::CmpIOp op) {
    return op.getPredicate();
  }

  static bool gtPredicate(mlir::arith::CmpIPredicate predicate) {
    return predicate == mlir::arith::CmpIPredicate::sgt ||
           predicate == mlir::arith::CmpIPredicate::ugt;
  }

  static bool gePredicate(mlir::arith::CmpIPredicate predicate) {
    return predicate == mlir::arith::CmpIPredicate::sge ||
           predicate == mlir::arith::CmpIPredicate::uge;
  }

  static bool ltPredicate(mlir::arith::CmpIPredicate predicate) {
    return predicate == mlir::arith::CmpIPredicate::slt ||
           predicate == mlir::arith::CmpIPredicate::ult;
  }

  static bool lePredicate(mlir::arith::CmpIPredicate predicate) {
    return predicate == mlir::arith::CmpIPredicate::sle ||
           predicate == mlir::arith::CmpIPredicate::ule;
  }
};

template <typename OpTy>
class LogicalOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
 public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

 private:
  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
      int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
      const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      assert(false && "constant value");
    }
    return {};
  }
};

template <typename OpTy>
class CastOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
 public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op, llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    return operands[0]->getValue();
  }
};

class MakeRangeOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<mlir::triton::MakeRangeOp> {
 public:
  using AxisInfoVisitorImpl<mlir::triton::MakeRangeOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::triton::MakeRangeOp op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override {
    auto start = op.getStart();
    auto end = op.getEnd();
    return AxisInfo(/*contiguity=*/{end - start},
      /*divisibility=*/{tritoncc::highestPowOf2Divisor(start)},
      /*constancy=*/{1});
  }
};

class LoadOpAxisInfoVisitor final : public AxisInfoVisitorImpl<mlir::triton::LoadOp> {
 public:
  using AxisInfoVisitorImpl<mlir::triton::LoadOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::triton::LoadOp op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) override  {
    // If pointers and mask both have constancy properties, those properties
    // will also extend to output.
    AxisInfo ptrInfo = operands[0]->getValue();
    std::optional<AxisInfo> maskInfo;
    if (operands.size() > 1) {
      maskInfo = operands[1]->getValue();
    }
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;

    for (int d = 0; d < ptrInfo.getRank(); ++d) {
      contiguity.push_back(1);
      divisibility.push_back(1);
      constancy.push_back(
        gcd(ptrInfo.getConstancy(d),
          maskInfo.has_value() ? maskInfo->getConstancy(d) : 0));
    }
    return AxisInfo(contiguity, divisibility, constancy);
  }
};

class AxisInfoVisitorList {
 public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AxisInfo apply(mlir::Operation *op, llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands) {
    for (auto &visitor : visitors) {
      if (visitor->match(op)) {
        return visitor->getAxisInfo(op, operands);
      }
    }
    return AxisInfo();
  }
 private:
  std::vector<std::unique_ptr<AxisInfoVisitor>> visitors;
};

class AxisInfoAnalysis : public mlir::dataflow::SparseForwardDataFlowAnalysis<mlir::dataflow::Lattice<AxisInfo>> {
 private:
  AxisInfoVisitorList visitors;

  void setToEntryState(mlir::dataflow::Lattice<AxisInfo> *lattice) override {
    propagateIfChanged(
      lattice,
      lattice->join(AxisInfo::getPessimisticValueState(lattice->getPoint())));
  }
 public:
  AxisInfoAnalysis(mlir::DataFlowSolver &solver)
      : mlir::dataflow::SparseForwardDataFlowAnalysis<mlir::dataflow::Lattice<AxisInfo>>(solver) {
    // populator visitors
    visitors.append<MakeRangeOpAxisInfoVisitor>();
    visitors.append<LoadOpAxisInfoVisitor>();
    visitors.append<BroadcastOpAxisInfoVisitor>();
    visitors.append<ExpandDimsOpAxisInfoVisitor>();
    visitors.append<SplatOpAxisInfoVisitor>();
    visitors.append<AddSubOpAxisInfoVisitor<mlir::triton::AddPtrOp>,
        AddSubOpAxisInfoVisitor<mlir::arith::AddIOp>,
        AddSubOpAxisInfoVisitor<mlir::arith::SubIOp>,
        AddSubOpAxisInfoVisitor<mlir::LLVM::AddOp>>();

    visitors.append<MulIOpAxisInfoVisitor>();
    visitors.append<ConstantOpAxisInfoVisitor<mlir::arith::ConstantOp>,
                    ConstantOpAxisInfoVisitor<mlir::LLVM::ConstantOp>>();

    visitors.append<CmpOpAxisInfoVisitor<mlir::arith::CmpIOp>>();
    visitors.append<LogicalOpAxisInfoVisitor<mlir::arith::AndIOp>,
                    LogicalOpAxisInfoVisitor<mlir::arith::OrIOp>,
                    LogicalOpAxisInfoVisitor<mlir::arith::XOrIOp>>();

    visitors.append<CastOpAxisInfoVisitor<mlir::arith::ExtSIOp>,
                    CastOpAxisInfoVisitor<mlir::arith::ExtUIOp>,
                    CastOpAxisInfoVisitor<mlir::arith::TruncIOp>,
                    CastOpAxisInfoVisitor<mlir::arith::IndexCastOp>,
                    CastOpAxisInfoVisitor<mlir::triton::gpu::ConvertLayoutOp>,
                    CastOpAxisInfoVisitor<mlir::UnrealizedConversionCastOp>,
                    CastOpAxisInfoVisitor<mlir::triton::BitcastOp>>();
  }

  void visitOperation(mlir::Operation *op,
      llvm::ArrayRef<const mlir::dataflow::Lattice<AxisInfo>*> operands,
      llvm::ArrayRef<mlir::dataflow::Lattice<AxisInfo>*> results) override {
    for (auto op : operands) {
      if (op->getValue().getRank() == 0) {
        setToEntryState((mlir::dataflow::Lattice<AxisInfo>*) op);
      }
    }
    AxisInfo curr = visitors.apply(op, operands);
    if (curr.getRank() == 0) {
      return setAllToEntryStates(results);
    }
    // override with hint
    auto newContiguity = curr.getContiguity();
    auto newDivisibility = curr.getDivisibility();
    auto newConstancy = curr.getConstancy();
    if (mlir::Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
      auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
      newContiguity = AxisInfo::DimVectorT(vals.begin(), vals.end());
    }
    if (mlir::Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
      auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
      newDivisibility = AxisInfo::DimVectorT(vals.begin(), vals.end());
    }
    if (mlir::Attribute attr = op->getDiscardableAttr("tt.constancy")) {
      auto vals = attr.cast<mlir::DenseElementsAttr>().getValues<int>();
      newConstancy = AxisInfo::DimVectorT(vals.begin(), vals.end());
    }
    curr = AxisInfo(newContiguity, newDivisibility, newConstancy,
        curr.getConstantValue());
    // join all latttice elements
    for (auto *result : results) {
      propagateIfChanged(result, result->join(curr));
    }
  }

  using mlir::dataflow::SparseForwardDataFlowAnalysis<mlir::dataflow::Lattice<AxisInfo>>::getLatticeElement;
};

using AxisInfoMapT = llvm::DenseMap<mlir::Value, AxisInfo>;
class ModuleAxisInfoAnalysis : public mlir::CallGraph<AxisInfoMapT> {
 public:
  explicit ModuleAxisInfoAnalysis(mlir::ModuleOp moduleOp) : CallGraph<AxisInfoMapT>(moduleOp) {
    llvm::SmallVector<mlir::FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<mlir::WalkOrder::PreOrder, mlir::WalkOrder::PostOrder>(
        // pre-order edge walk callback
        [](mlir::CallOpInterface callOp, mlir::FunctionOpInterface funcOp) {},
        // post-order node walk callback
        [&](mlir::FunctionOpInterface funcOp) {
          funcs.push_back(funcOp);
          funcMap.try_emplace(funcOp, AxisInfoMapT{});
        });
    }
    llvm::SetVector<mlir::FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    mlir::SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp);
      funcOp.walk([&](mlir::CallOpInterface callOp) {
        auto callee =
            llvm::dyn_cast<mlir::FunctionOpInterface>(callOp.resolveCallable(&symbolTable));
        update(callOp, callee);
      });
    }
  }

  unsigned getPtrContiguity(mlir::Value ptr) {
    auto tensorTy = ptr.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorTy) {
      return 1;
    }
    auto layout = tensorTy.getEncoding();

    // Here order should be ordered by contiguous first, so the first element
    // should have the largest contiguous.
    auto order = mlir::triton::gpu::getOrder(layout);
    unsigned align = getPtrAlignment(ptr);

    auto uniqueContigPerThread =
        mlir::triton::gpu::getUniqueContigPerThread(layout, tensorTy.getShape());
    assert(order[0] < uniqueContigPerThread.size() &&
        "Unexpected uniqueContigPerThread size");
    unsigned contiguity = uniqueContigPerThread[order[0]];
    contiguity = std::min(align, contiguity);

    return contiguity;
  }

  unsigned getPtrAlignment(mlir::Value ptr) {
    auto tensorTy = ptr.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorTy) {
      return 1;
    }
    auto *axisInfo = getAxisInfo(ptr);
    if (!axisInfo) {
      return 1;
    }
    auto layout = tensorTy.getEncoding();
    auto order = mlir::triton::gpu::getOrder(layout);
    auto maxMultipleBytes = axisInfo->getDivisibility(order[0]);
    auto maxContig = axisInfo->getContiguity(order[0]);
    auto elemNumBits = mlir::triton::getPointeeBitWidth(tensorTy);
    auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
    auto maxMultiple = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);
    unsigned alignment = std::min(maxMultiple, maxContig);
    return alignment;
  }

  unsigned getMaskAlignment(mlir::Value mask) {
    auto tensorTy = mask.getType().dyn_cast<mlir::RankedTensorType>();
    if (!tensorTy) {
      return 1;
    }
    auto *axisInfo = getAxisInfo(mask);
    if (!axisInfo) {
      return 1;
    }
    auto maskOrder = mlir::triton::gpu::getOrder(tensorTy.getEncoding());
    auto alignment = std::max<unsigned>(axisInfo->getConstancy(maskOrder[0]), 1);
    return alignment;
  }

  AxisInfo *getAxisInfo(mlir::Value value) {
    auto funcOp = value.getParentRegion()->getParentOfType<mlir::FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }
 private:
  void initialize(mlir::FunctionOpInterface funcOp) {
    std::unique_ptr<mlir::DataFlowSolver> solver = mlir::createDataFlowSolver();
    AxisInfoAnalysis *analysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(funcOp))) {
      return;
    }
    auto *axisInfoMap = getFuncData(funcOp);
    auto updateAxisInfoMap = [&](mlir::Value value) {
      auto axisInfo = analysis->getLatticeElement(value)->getValue();
      AxisInfo curAxisInfo;
      if (axisInfoMap->count(value)) {
        curAxisInfo = AxisInfo::join(axisInfo, axisInfoMap->lookup(value));
      } else {
        curAxisInfo = axisInfo;
      }
      (*axisInfoMap)[value] = curAxisInfo;
    };
    funcOp.walk([&](mlir::Operation *op) {
      for (auto value : op->getResults()) {
        updateAxisInfoMap(value);
      }
    });
    funcOp.walk([&](mlir::Block *block) {
      for (auto value : block->getArguments()) {
        updateAxisInfoMap(value);
      }
    });
  }

  void update(mlir::CallOpInterface callOp, mlir::FunctionOpInterface funcOp) {
    assert(false && "update");
  }
};

}

#endif
