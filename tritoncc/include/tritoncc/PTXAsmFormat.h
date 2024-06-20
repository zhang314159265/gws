#pragma once

#ifdef USE_TRITON
#undef USE_TRITON
#endif
#define USE_TRITON 0
#if USE_TRITON

#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#else

namespace tritoncc {

struct PTXInstr;
struct PTXInstrCommon;
struct PTXInstrExecution;

struct PTXBuilder {
  struct Operand {
    std::string constraint;
    mlir::Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    Operand() = default;
    Operand(const mlir::Operation &) = delete;
    Operand(mlir::Value value, llvm::StringRef constraint)
        : constraint(constraint), value(value) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  template <typename INSTR = PTXInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  Operand *newAddrOperand(mlir::Value addr, llvm::StringRef constraint, int off = 0) {
    auto *opr = newOperand(addr, constraint);
    opr->repr = [off](int idx) -> std::string {
      std::stringstream ss;
      ss << "[ $" << idx << " + " << off << " ]";
      return ss.str();
    };
    return opr;
  }

  Operand *newListOperand(llvm::ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand() {
    return newOperand();
  }

  Operand *newOperand(mlir::Value value, llvm::StringRef constraint,
      std::function<std::string(int idx)> formatter = nullptr) {
    argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
    auto *opr = argArchive.back().get();
    opr->repr = formatter;
    opr->idx = oprCounter++;
    return opr;
  }

  Operand *newOperand(llvm::StringRef constraint, bool init = false) {
    // Constraint should be something like "=r"
    assert(constraint.size() == 2 && constraint[0] == '=');
    auto *opr = newOperand();
    opr->idx = oprCounter++;
    opr->constraint = constraint;
    if (init) {
      initOperand(opr);
    }
    return opr;
  }

  Operand *newOperand(unsigned operandIndex) {
    assert(false && "newOperand");
  }

  Operand *newConstantOperand(int64_t v) {
    std::stringstream ss;
    ss << "0x" << std::hex << v;
    return newConstantOperand(ss.str());
  }

  Operand *newConstantOperand(const std::string &v) {
    argArchive.emplace_back(std::make_unique<Operand>());
    argArchive.back()->repr = [v](int idx) { return v; };
    return argArchive.back().get();
  }

  llvm::SmallVector<Operand *, 4> getAllArgs() const {
    llvm::SmallVector<Operand *, 4> res;
    for (auto &x : argArchive) {
      if(!x->isList()) {
        res.push_back(x.get());
      }
    }
    return res;
  }

  llvm::SmallVector<mlir::Value, 4> getAllMLIRArgs() const {
    llvm::SmallVector<mlir::Value, 4> res;
    for (auto &arg : argArchive) {
      if (!arg->isList() && arg->value) {
        res.push_back(arg->value);
      }
    }
    return res;
  }

  std::string getConstraints() const {
    auto args = getAllArgs();
    llvm::SmallVector<std::string, 4> argReprs;
    for (auto arg : args) {
      argReprs.push_back(arg->constraint);
    }
    return tritoncc::strJoin(argReprs, ",");
  }

  std::string dump() const;

  mlir::Value launch(mlir::OpBuilder &rewriter, mlir::Location loc, mlir::Type resTy,
      bool hasSideEffect = true, bool isAlignStack = false,
      llvm::ArrayRef<mlir::Attribute> attrs = {}) const {
    auto *ctx = rewriter.getContext();
    auto inlineAsm = rewriter.create<mlir::LLVM::InlineAsmOp>(
      loc, resTy, getAllMLIRArgs(), // operands
      dump(), // asm_string
      getConstraints(), // constraints
      hasSideEffect, // has_side_effects
      isAlignStack, // is_align_stack
      mlir::LLVM::AsmDialectAttr::get(ctx,
          mlir::LLVM::AsmDialect::AD_ATT), // asm_dialect
      mlir::ArrayAttr::get(ctx, attrs) // operand_attrs
    );
    return inlineAsm.getRes();
    assert(false && "launch");
  }
 private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  void initOperand(Operand *opr);

  void reorderArgArchive(llvm::ArrayRef<Operand *> order) {
    assert(order.size() == argArchive.size());
    sort(argArchive.begin(), argArchive.end(),
      [&](std::unique_ptr<Operand> &a, std::unique_ptr<Operand> &b) {
        auto ida = std::find(order.begin(), order.end(), a.get());
        auto idb = std::find(order.begin(), order.end(), b.get());
        assert(ida != order.end());
        assert(idb != order.end());
        return ida < idb;
      });
  }

  friend struct PTXInstr;
  friend struct PTXInstrCommon;
 protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<PTXInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<PTXInstrExecution>, 4> executions;
  int oprCounter{};
};

struct PTXInstrCommon {
  explicit PTXInstrCommon(PTXBuilder *builder) : builder(builder) {}

  using Operand = PTXBuilder::Operand;

  PTXInstrExecution& operator()(Operand *a, Operand *b) {
    return call({a, b});
  }

  PTXInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs,
      bool onlyAttachMLIRArgs = false) {
    return call(oprs, onlyAttachMLIRArgs);
  }
 protected:
  PTXInstrExecution &call(llvm::ArrayRef<Operand *> oprs,
      bool onlyAttachMLIRArgs = false) {
    if (onlyAttachMLIRArgs) {
      assert(builder->executions.empty() &&
        "builder can only hold a single execution when onlyAttachMLIRArgs "
        "is true.");
      builder->reorderArgArchive(oprs);
    }
    builder->executions.emplace_back(
        std::make_unique<PTXInstrExecution>(this, oprs, onlyAttachMLIRArgs));
    return *builder->executions.back();
  }

  PTXBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend struct PTXInstrExecution;
};

template <class ConcreteT>
struct PTXInstrBase : public PTXInstrCommon {
  explicit PTXInstrBase(PTXBuilder *builder, const std::string &name)
      : PTXInstrCommon(builder) {
    o(name);
  }

  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate) {
      instrParts.push_back(suffix);
    }
    return *static_cast<ConcreteT *>(this);
  }
};

struct PTXInstr : public PTXInstrBase<PTXInstr> {
  using PTXInstrBase<PTXInstr>::PTXInstrBase;

  PTXInstr &global() {
    o("global");
    return *this;
  }

  PTXInstr &shared() {
    o("shared");
    return *this;
  }

  // Append a ".v[0-9]+" to the instruction
  PTXInstr &v(int vecWidth, bool predicate = true) {
    if (vecWidth > 1) {
      o("v" + std::to_string(vecWidth), predicate);
    }
    return *this;
  }

  // Append a ".b[0-9]+" to the instruction
  PTXInstr &b(int width) {
    o("b" + std::to_string(width));
    return *this;
  }
};

// Record the operands and context for "launching" a PTXInstr
struct PTXInstrExecution {
  using Operand = PTXBuilder::Operand;

  llvm::SmallVector<Operand *> argsInOrder;

  PTXInstrExecution() = default;
  explicit PTXInstrExecution(PTXInstrCommon *instr,
      llvm::ArrayRef<Operand *> oprs,
      bool onlyAttachMLIRArgs)
    : argsInOrder(oprs.begin(), oprs.end()), instr(instr),
      onlyAttachMLIRArgs(onlyAttachMLIRArgs) {}

  // Prefix a predicate to the instruction.
  PTXInstrExecution &predicate(mlir::Value value, llvm::StringRef constraint = "b") {
    pred = instr->builder->newOperand(value, constraint);
    return *this;
  }

  PTXInstrExecution &predicateNot(mlir::Value value, llvm::StringRef constraint) {
    assert(false && "predicateNot");
  }

  std::string dump() const;
  PTXInstrCommon *instr{};
  Operand *pred{};
  bool onlyAttachMLIRArgs{};
};

std::string PTXBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return tritoncc::strJoin(lines, "\n\t");
}

void PTXBuilder::initOperand(Operand *opr) {
  auto numBits = 0;
  // Derive numBits from the constraint.
  if (opr->constraint[1] == 'c' || opr->constraint[1] == 'h') {
    numBits = 16;
  } else if (opr->constraint[1] == 'r') {
    numBits = 32;
  } else if (opr->constraint[1] == 'l') {
    numBits = 64;
  } else {
    llvm_unreachable(("Unknown constraint: " + opr->constraint).c_str());
  }
  // If numBits is less than 16, we use 16 as default because PTX does not
  // support 8-bit mov.
  numBits = numBits < 16 ? 16 : numBits;
  auto *zero = newConstantOperand(0);
  auto &init = create<>("mov")->o("u" + std::to_string(numBits));
  init(opr, zero);
}

std::string PTXInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  std::string instrRepr = tritoncc::strJoin(instr->instrParts, ".");
  if (onlyAttachMLIRArgs) {
    return instrRepr;
  }

  if (pred) {
    if (!pred->repr) {
      os << "@" << pred->dump() << " ";
    } else {
      os << pred->repr(pred->idx) << " ";
    }
  }

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = tritoncc::strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

std::string PTXInstr::Operand::dump() const {
  if (repr) {
    return repr(idx);
  }
  if (!isList()) {
    return "$" + std::to_string(idx);
  }

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list) {
    oprs.push_back(opr->dump());
  }
  return "{ " + tritoncc::strJoin(oprs, ", ") + " }";
}

}
#endif
