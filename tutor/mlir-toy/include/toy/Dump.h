#pragma once

#include "llvm/ADT/TypeSwitch.h"

#include "toy/AST.h"

namespace toy {

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT() \
  Indent level_(curIndent); \
  indent();

template <typename T>
static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
      llvm::Twine(loc.col)).str();
}

class ASTDumper {
 public:
  void dump(ModuleAST *node) {
    INDENT();
    llvm::errs() << "Module:\n";

    for (auto &record : *node) {
      if (FunctionAST *function = llvm::dyn_cast<FunctionAST>(record.get())) {
        dump(function);
      } else if(StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        dump(str);
      } else {
        llvm::errs() << "<unknown Record, kind " << record->getKind() << ">\n";
      }
    }
  }
 private:
  void dump(StructAST *node) {
    INDENT();
    llvm::errs() << "Struct: " << node->getName() << " " << loc(node) << "\n";

    {
      INDENT();
      llvm::errs() << "Variables: [\n";
      for (auto &variable : node->getVariables()) {
        dump(variable.get());
      }
      indent();
      llvm::errs() << "]\n";
    }
  }

  void dump(FunctionAST *node) {
    INDENT();
    llvm::errs() << "Function \n";
    dump(node->getProto());
    dump(node->getBody());
  }

  void dump(PrototypeAST *node) {
    INDENT();
    llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
    indent();
    llvm::errs() << "Params: [";
    llvm::interleaveComma(node->getArgs(), llvm::errs(),
        [](auto &arg) { llvm::errs() << arg->getName(); });
    llvm::errs() << "]\n";
  }

  void dump(ExprASTList *exprList) {
    INDENT();
    llvm::errs() << "Block {\n";
    for (auto &expr : *exprList) {
      dump(expr.get());
    }
    indent();
    llvm::errs() << "} // Block\n";
  }

  // Dispatch a generic expression to the appropriate subclass using RTTI
  void dump(ExprAST *expr) {
    llvm::TypeSwitch<ExprAST *>(expr)
        .Case<BinaryExprAST, ReturnExprAST, VarDeclExprAST, CallExprAST,
            VariableExprAST, LiteralExprAST, PrintExprAST, StructLiteralExprAST>(
          [&](auto *node) { this->dump(node); })
        .Default([&](ExprAST *) {
          // No match, fallback to a generic message
          INDENT();
          llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
        });
  }

  void dump(StructLiteralExprAST *node) {
    INDENT();
    llvm::errs() << "Struct Literal: " << loc(node) << "\n";
    for (auto &value : node->getValues()) {
      dump(value.get());
    }
  }

  void dump(PrintExprAST *node) {
    INDENT();
    llvm::errs() << "Print [ " << loc(node) << "\n";
    dump(node->getArg());
    indent();
    llvm::errs() << "]\n";
  }

  void printLitHelper(ExprAST *litOrNum) {
    if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
      llvm::errs() << num->getValue();
      return;
    }
    auto *literal = llvm::cast<LiteralExprAST>(litOrNum);

    // dim
    llvm::errs() << "<";
    llvm::interleaveComma(literal->getDims(), llvm::errs());
    llvm::errs() << ">";

    // content
    llvm::errs() << "[ ";
    llvm::interleaveComma(literal->getValues(), llvm::errs(),
        [&](auto &elt) { printLitHelper(elt.get()); });
    llvm::errs() << "]";
  }

  void dump(LiteralExprAST *node) {
    INDENT();
    llvm::errs() << "Literal: ";
    printLitHelper(node);
    llvm::errs() << " " << loc(node) << "\n";
  }

  void dump(VariableExprAST *node) {
    INDENT();
    llvm::errs() << "var: " << node->getName() << " " << loc(node) << "\n";
  }

  void dump(CallExprAST *node) {
    INDENT();
    llvm::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
    for (auto &arg : node->getArgs()) {
      dump(arg.get());
    }
    indent();
    llvm::errs() << "]\n";
  }

  void dump(const VarType &type) {
    llvm::errs() << "<";
    llvm::interleaveComma(type.shape, llvm::errs());
    llvm::errs() << ">";
  }

  void dump(VarDeclExprAST *varDecl) {
    INDENT();
    llvm::errs() << "VarDecl " << varDecl->getName();
    dump(varDecl->getType());
    llvm::errs() << " " << loc(varDecl) << "\n";
    if (auto *initVal = varDecl->getInitVal()) {
      dump(initVal);
    }
  }

  void dump(ReturnExprAST *node) {
    INDENT();
    llvm::errs() << "Return\n";
    if (node->getExpr().has_value()) {
      return dump(*node->getExpr());
    }
    {
      INDENT();
      llvm::errs() << "(void)\n";
    }
  }

  void dump(BinaryExprAST *node) {
    INDENT();
    llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
    dump(node->getLHS());
    dump(node->getRHS());
  }

  void indent() {
    for (int i = 0; i < curIndent; ++i) {
      llvm::errs() << "  ";
    }
  }
  int curIndent = 0;
};

void dump(ModuleAST &module) {
  ASTDumper().dump(&module);
}

}
