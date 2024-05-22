#pragma once

#include <numeric>

#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

namespace toy {

class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) { }

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST) {
      mlirGen(f);
    }

    if (mlir::failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }
    return theModule;
  }
 private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
        loc.line, loc.col);
  }

  mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
        getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<mlir::toy::FuncOp>(location, proto.getName(), funcType);
  }

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) {
      return mlir::failure();
    }
    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function) {
      return nullptr;
    }
    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table
    for (const auto nameValue :
        llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (mlir::failed(declare(std::get<0>(nameValue)->getName(),
          std::get<1>(nameValue)))) {
        return nullptr;
      }
    }

    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty()) {
      returnOp = llvm::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    }

    if (!returnOp) {
      builder.create<mlir::toy::ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      function.setType(builder.getFunctionType(
        function.getFunctionType().getInputs(),
        getType(VarType{})));
    }

    return function;
  }

  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
    for (auto & expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. There can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = llvm::dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl)) {
          return mlir::failure();
        }
        continue;
      }
      if (auto *ret = llvm::dyn_cast<ReturnExprAST>(expr.get())) {
        return mlirGen(*ret);
      }
      if (auto *print = llvm::dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print))) {
          return mlir::success();
        }
        continue;
      }

      // Generic expression dispatch codegen
      if (!mlirGen(*expr)) {
        return mlir::failure();
      }
      return mlir::success();
    }
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg) {
      return mlir::failure();
    }

    builder.create<mlir::toy::PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr()))) {
        return mlir::failure();
      }
    }

    builder.create<mlir::toy::ReturnOp>(location,
        expr ? llvm::ArrayRef(expr) : llvm::ArrayRef<mlir::Value>());
    return mlir::success();
  }

  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case ExprAST::Expr_BinOp:
      return mlirGen(llvm::cast<BinaryExprAST>(expr));
    case ExprAST::Expr_Call:
      return mlirGen(llvm::cast<CallExprAST>(expr));
    case ExprAST::Expr_Var:
      return mlirGen(llvm::cast<VariableExprAST>(expr));
    case ExprAST::Expr_Literal:
      return mlirGen(llvm::cast<LiteralExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << llvm::Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
        std::multiplies<int>()));
    collectData(lit, data);

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    return builder.create<mlir::toy::ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = llvm::dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues()) {
        collectData(*value, data);
      }
      return;
    }
    assert(llvm::isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(llvm::cast<NumberExprAST>(expr).getValue());
  }

  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName())) {
      return variable;
    }

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    llvm::SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg) {
        return nullptr;
      }
      operands.push_back(arg);
    }

    // Call a buildin function
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
            "does not accept multiple arguments");
        return nullptr;
      }
      return builder.create<mlir::toy::TransposeOp>(location, operands[0]);
    }

    // Call a user defined function
    return builder.create<mlir::toy::GenericCallOp>(location, callee, operands);
  }

  mlir::Value mlirGen(BinaryExprAST &binop) {
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs) {
      return nullptr;
    }
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs) {
      return nullptr;
    }
    auto location = loc(binop.loc());

    switch (binop.getOp()) {
    case '+':
      return builder.create<mlir::toy::AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<mlir::toy::MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
          "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value) {
      return nullptr;
    }

    // We have the initialize value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = builder.create<mlir::toy::ReshapeOp>(loc(vardecl.loc()),
          getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (mlir::failed(declare(vardecl.getName(), value))) {
      return nullptr;
    }
    return value;
  }

  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    // if the shape is empty, then this type is unranked
    if (shape.empty()) {
      return mlir::UnrankedTensorType::get(builder.getF64Type()); 
    }

    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
    ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

}
