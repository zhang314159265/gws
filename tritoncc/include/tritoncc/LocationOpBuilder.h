#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "tritoncc/dialect/Triton/Dialect.h"

namespace tritoncc {

mlir::arith::CmpIPredicate cmpIStrToPredicate(const std::string &opstr) {
  if (opstr == "slt") {
    return mlir::arith::CmpIPredicate::slt;
  } else {
    assert(false && "cmpIStrToPredicate");
  }
}

class LocationOpBuilder {
 public:
  LocationOpBuilder(mlir::MLIRContext *ctx) {
    builder = std::make_unique<mlir::OpBuilder>(ctx);
  }

  mlir::Location getLastLoc() {
    return builder->getUnknownLoc();
  }

  void setInsertionPointToStart(mlir::Block *block) {
    builder->setInsertionPointToStart(block);
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(), mlir::Value>
  createOrFold(Args &&...args) {
    return builder->createOrFold<OpTy>(getLastLoc(), std::forward<Args>(args)...); 
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    return builder->createOrFold<OpTy>(getLastLoc(), std::forward<Args>(args)...);
  }

  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return builder->create<OpTy>(getLastLoc(), std::forward<Args>(args)...);
  }

  mlir::_tritoncc::FuncOp createFuncOp(const std::string &funcName, std::vector<mlir::Type> inTypes, std::vector<mlir::Type> outTypes, bool isPublic) {
    mlir::FunctionType functy = builder->getFunctionType(inTypes, outTypes).dyn_cast<mlir::FunctionType>();
    llvm::SmallVector<mlir::NamedAttribute> attrs = {
      mlir::NamedAttribute(
        builder->getStringAttr("sym_visibility"),
        builder->getStringAttr(isPublic ? "public" : "private")
      ),
      mlir::NamedAttribute(
        builder->getStringAttr("noinline"),
        builder->getBoolAttr(false)
      )
    };

    return this->create<mlir::_tritoncc::FuncOp>(funcName, functy, attrs);
  }

  mlir::Value createGetProgramId(int axis) {
    return create<mlir::_tritoncc::GetProgramIdOp>(
      builder->getI32Type(),
      mlir::_tritoncc::ProgramIDDimAttr::get(
        builder->getContext(),
        mlir::_tritoncc::ProgramIDDim(axis)
      ) 
    );
  }

  mlir::Value getConstant(int32_t v) {
    return mlir::Value(create<mlir::arith::ConstantIntOp>(
      v, builder->getI32Type()
    ));
  }

  mlir::Value createMulI(mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::arith::MulIOp>(lhs, rhs);
  }

  mlir::Value createAddI(mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::arith::AddIOp>(lhs, rhs);
  }

  mlir::Value createAddF(mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::arith::AddFOp>(lhs, rhs);
  }

  mlir::Value createMakeRange(int start, int end) {
    return create<mlir::_tritoncc::MakeRangeOp>(
      mlir::RankedTensorType::get(
        {end - start}, builder->getI32Type()
      ),
      start,
      end
    );
  }

  mlir::Value createSplat(mlir::Value arg, std::vector<int64_t> shape) {
    return createOrFold<mlir::_tritoncc::SplatOp>(
      mlir::RankedTensorType::get(shape, arg.getType()), arg
    );
  }

  mlir::Value createExpandDims(mlir::Value arg, int axis) {
    auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
    mlir::Type argEltType = argType.getElementType();
    std::vector<int64_t> retShape = argType.getShape();
    retShape.insert(retShape.begin() + axis, 1);
    return create<mlir::_tritoncc::ExpandDimsOp>(
      mlir::RankedTensorType::get(retShape, argEltType),
      arg,
      axis
    );
  }

  mlir::Value createBroadcast(mlir::Value arg, std::vector<int64_t> shape) {
    mlir::RankedTensorType argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
    return createOrFold<mlir::_tritoncc::BroadcastOp>(
      mlir::RankedTensorType::get(shape, argType.getElementType()),
      arg
    );
  }

  mlir::Value createCmpI(const std::string &opstr, mlir::Value lhs, mlir::Value rhs) {
    mlir::arith::CmpIPredicate op = cmpIStrToPredicate(opstr);
    return create<mlir::arith::CmpIOp>(op, lhs, rhs);
  }

  mlir::Value createAndI(mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::arith::AndIOp>(lhs, rhs);
  }

  mlir::Value createAddPtr(mlir::Value ptr, mlir::Value offset) {
    return create<mlir::_tritoncc::AddPtrOp>(ptr.getType(), ptr, offset);
  }

  mlir::Value createLoad(mlir::Value ptr, mlir::Value mask) {
    return create<mlir::_tritoncc::LoadOp>(
      ptr,
      mask,
      mlir::Value(),
      mlir::_tritoncc::CacheModifier::NONE,
      mlir::_tritoncc::EvictionPolicy::NORMAL,
      false
    );
  }

  void createStore(mlir::Value ptr, mlir::Value val, mlir::Value mask) {
    create<mlir::_tritoncc::StoreOp>(
      ptr, val, mask,
      mlir::_tritoncc::CacheModifier::NONE,
      mlir::_tritoncc::EvictionPolicy::NORMAL
    );
  }

  mlir::OpState createReturn(std::vector<mlir::Value> vals) {
    return create<mlir::_tritoncc::ReturnOp>(vals);
  }

  mlir::OpState createReduce(std::vector<mlir::Value> operands, int axis) {
    return create<mlir::_tritoncc::ReduceOp>(operands, axis);
  }

  mlir::OpState createReduceReturn(std::vector<mlir::Value> args) {
    return create<mlir::_tritoncc::ReduceReturnOp>(args);
  }

  // note that return type is mlir::OpState rather than mlir::Value
  mlir::OpState createCall(mlir::_tritoncc::FuncOp func,
      std::vector<mlir::Value> args) {
    return create<mlir::_tritoncc::CallOp>(func, args);
  }

  mlir::Type i32ty() {
    return builder->getI32Type();
  }

  mlir::Type f32ty() {
    return builder->getF32Type();
  }

  mlir::Type f32pty() {
    return mlir::_tritoncc::PointerType::get(f32ty(), 1);
  }

  mlir::RankedTensorType blockty(std::vector<int64_t> shape, mlir::Type elemTy) {
    return mlir::RankedTensorType::get(shape, elemTy);
  }

  mlir::OpBuilder &getBuilder() { return *builder; }
 private:
  std::unique_ptr<mlir::OpBuilder> builder;
};

}
