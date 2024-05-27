#pragma once

#include <numeric>

#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "toy/Dialect.h"

namespace toy {

class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) { }

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &record : moduleAST) {
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        mlir::toy::FuncOp func = mlirGen(*funcAST);
        if (!func) {
          return nullptr;
        }
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str))) {
          return nullptr;
        }
      } else {
        llvm_unreachable("unknown record type");
      }
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
  llvm::ScopedHashTable<llvm::StringRef, std::pair<mlir::Value, VarDeclExprAST*>> symbolTable;
  // A mapping for the functions that have been code genereated to MLIR.
  llvm::StringMap<mlir::toy::FuncOp> functionMap;

  // A mapping for named struct types to the underlying MLIR type and the
  // original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;

  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
        loc.line, loc.col);
  }

  // Create an MLIR type for the given struct.
  mlir::LogicalResult mlirGen(StructAST &str) {
    if (structMap.count(str.getName())) {
      return emitError(loc(str.loc())) << "error: struct type with name `"
          << str.getName() << "' already exists";
    }

    auto variables = str.getVariables();
    std::vector<mlir::Type> elementTypes;
    elementTypes.reserve(variables.size());
    for (auto &variable : variables) {
      if (variable->getInitVal()) {
        return emitError(loc(variable->loc()))
            << "error: variables within  a struct definition must not have "
               "initializers";
      }
      if (!variable->getType().shape.empty()) {
        return emitError(loc(variable->loc()))
            << "error: variables within a struct definition must not have "
               "shape specified";
      }

      mlir::Type type = getType(variable->getType(), variable->loc());
      if (!type) {
        return mlir::failure();
      }
      elementTypes.push_back(type);
    }
    structMap.try_emplace(str.getName(), mlir::toy::StructType::get(elementTypes), &str);
    return mlir::success();
  }

  mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto &arg : proto.getArgs()) {
      mlir::Type type = getType(arg->getType(), arg->loc());
      if (!type) {
        return nullptr;
      } 
      argTypes.push_back(type);
    }
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<mlir::toy::FuncOp>(location, proto.getName(), funcType);
  }

  mlir::LogicalResult declare(VarDeclExprAST &var, mlir::Value value) {
    if (symbolTable.count(var.getName())) {
      return mlir::failure();
    }
    symbolTable.insert(var.getName(), {value, &var});
    return mlir::success();
  }

  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    SymbolTableScopeT varScope(symbolTable);

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
      if (mlir::failed(declare(*std::get<0>(nameValue),
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
        *returnOp.operand_type_begin()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main") {
      function.setPrivate();
    }

    functionMap.insert({function.getName(), function});
    return function;
  }

  using SymbolTableScopeT = llvm::ScopedHashTableScope<
    llvm::StringRef, std::pair<mlir::Value, VarDeclExprAST *>>;

  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    SymbolTableScopeT varScope(symbolTable);
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
    case ExprAST::Expr_StructLiteral:
      return mlirGen(llvm::cast<StructLiteralExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << llvm::Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  std::pair<mlir::ArrayAttr, mlir::Type>
  getConstantAttr(StructLiteralExprAST &lit) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;

    for (auto &var : lit.getValues()) {
      if (auto *number = llvm::dyn_cast<NumberExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*number));
        typeElements.push_back(getType(std::nullopt));
      } else if (auto *lit = llvm::dyn_cast<LiteralExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*lit));
        typeElements.push_back(getType(std::nullopt));
      } else {
        auto *structLit = llvm::cast<StructLiteralExprAST>(var.get());
        auto attrTypePair = getConstantAttr(*structLit);
        attrElements.push_back(attrTypePair.first);
        typeElements.push_back(attrTypePair.second);
      }
    }

    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    mlir::Type dataType = mlir::toy::StructType::get(typeElements);
    return std::make_pair(dataAttr, dataType);
  }

  mlir::DenseElementsAttr getConstantAttr(LiteralExprAST &lit) {
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
        std::multiplies<int>()));
    collectData(lit, data);

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    return mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));
  }

  mlir::DenseElementsAttr getConstantAttr(NumberExprAST &lit) {
    assert(false && "getConstantAttr NumberExprAST");
  }

  mlir::Value mlirGen(StructLiteralExprAST &lit) {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(lit);

    return builder.create<mlir::toy::StructConstantOp>(loc(lit.loc()), dataType, dataAttr);
  }

  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());
    mlir::DenseElementsAttr dataAttribute = getConstantAttr(lit);
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
    if (auto variable = symbolTable.lookup(expr.getName()).first) {
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

    auto calledFuncIt = functionMap.find(callee);
    if (calledFuncIt == functionMap.end()) {
      emitError(location) << "no defined function found for '" << callee << "'";
      return nullptr;
    }
    mlir::toy::FuncOp calledFunc = calledFuncIt->second;

    // Call a user defined function
    return builder.create<mlir::toy::GenericCallOp>(
        location,
        calledFunc.getFunctionType().getResult(0),
        mlir::SymbolRefAttr::get(builder.getContext(), callee),
        operands);
  }

  // Return the struct type that is the result of the given exprssion, or null
  // if it cannot be inferred.
  StructAST *getStructFor(ExprAST *expr) {
    llvm::StringRef structName;
    if (auto *decl = llvm::dyn_cast<VariableExprAST>(expr)) {
      auto varIt = symbolTable.lookup(decl->getName());
      if (!varIt.first) {
        return nullptr;
      }
      structName = varIt.second->getType().name;
    } else if (auto *access = llvm::dyn_cast<BinaryExprAST>(expr)) {
      if (access->getOp() != '.') {
        return nullptr;
      }
      auto *name = llvm::dyn_cast<VariableExprAST>(access->getRHS());
      if (!name) {
        return nullptr;
      }
      StructAST *parentStruct = getStructFor(access->getLHS());
      if (!parentStruct) {
        return nullptr;
      }

      // Get the element within the struct corresponding to the name.
      VarDeclExprAST *decl = nullptr;
      for (auto &var : parentStruct->getVariables()) {
        if (var->getName() == name->getName()) {
          decl = var.get();
          break;
        }
      }
      if (!decl) {
        return nullptr;
      }
      structName = decl->getType().name;
    }

    if (structName.empty()) {
      return nullptr;
    }

    auto structIt = structMap.find(structName);
    if (structIt == structMap.end()) {
      return nullptr;
    }
    return structIt->second.second;
  }

  // Return the numeric member index of the given struct access expression.
  std::optional<size_t> getMemberIndex(BinaryExprAST &accessOp) {
    assert(accessOp.getOp() == '.' && "expected access operation");

    // Lookup the struct node for the LHS.
    StructAST *structAST = getStructFor(accessOp.getLHS());
    if (!structAST) {
      return std::nullopt;
    }

    // Get the name from the RHS.
    VariableExprAST *name = llvm::dyn_cast<VariableExprAST>(accessOp.getRHS());
    if (!name) {
      return std::nullopt;
    }

    auto structVars = structAST->getVariables();
    const auto *it = llvm::find_if(structVars, [&](auto &var) {
      return var->getName() == name->getName();
    });
    if (it == structVars.end()) {
      return std::nullopt;
    }
    return it - structVars.begin();
  }

  mlir::Value mlirGen(BinaryExprAST &binop) {
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs) {
      return nullptr;
    }
    auto location = loc(binop.loc());

    // If this is an access operation, handle it immediately.
    if (binop.getOp() == '.') {
      std::optional<size_t> accessIndex = getMemberIndex(binop);
      if (!accessIndex) {
        emitError(location, "invalid access into struct expression");
        return nullptr;
      }
      return builder.create<mlir::toy::StructAccessOp>(location, lhs, *accessIndex);
    }

    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs) {
      return nullptr;
    }

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

    // Handle the case where we are initializing a struct value.
    VarType varType = vardecl.getType();
    if (!varType.name.empty()) {
      // Check that the initializer type is the same as the varaible
      // declaration.
      mlir::Type type = getType(varType, vardecl.loc());
      if (!type) {
        return nullptr;
      }
      if (type != value.getType()) {
        emitError(loc(vardecl.loc()))
            << "struct type of initializer is different to the varaible "
               "declaration. Got "
            << value.getType() << ", but expected " << type;
        return nullptr;
      }

      // Otherwise, we have the initialize value, but in case the variable was
      // declared with specific shape, we emit a "reshape" operation. It will
      // get optimized out later as needed.
    } else if (!varType.shape.empty()) {
      value = builder.create<mlir::toy::ReshapeOp>(loc(vardecl.loc()),
          getType(varType.shape), value);
    }

    // Register the value in the symbol table.
    if (mlir::failed(declare(vardecl, value))) {
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

  mlir::Type getType(const VarType &type, const Location &location) {
    if (!type.name.empty()) {
      auto it = structMap.find(type.name);
      if (it == structMap.end()) {
        emitError(loc(location))
            << "error: unknown struct type '" << type.name << "'";
        return nullptr;
      }
      return it->second.first;
    }
    return getType(type.shape); 
  }
};

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
    ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

}
