#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "tritoncc/legacy/Util.h"

namespace tritoncc {

/*
 * Wrap mlir::OpBuilder so caller does not need to pass in a location
 * for each create call.
 * 
 * Already include other handy methods like createFuncOp
 */
class LocationOpBuilder {
 public:
  LocationOpBuilder(mlir::MLIRContext* ctx) {
    builder = std::make_unique<mlir::OpBuilder>(ctx);
  }

  mlir::Location getLastLoc() {
    return builder->getUnknownLoc();
  }

  void setInsertionPointToStart(mlir::Block* block) {
    builder->setInsertionPointToStart(block);
  }

  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return builder->create<OpTy>(getLastLoc(), std::forward<Args>(args)...);
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

  mlir::triton::FuncOp createFuncOp(const std::string& funcName, std::vector<mlir::Type> inTypes, std::vector<mlir::Type> outTypes, bool ispublic) {
    mlir::FunctionType functy = builder->getFunctionType(inTypes, outTypes).dyn_cast<mlir::FunctionType>();
    llvm::SmallVector<mlir::NamedAttribute> attrs = {
      mlir::NamedAttribute(
        builder->getStringAttr("sym_visibility"),
        builder->getStringAttr(ispublic ? "public" : "private")
      ),
      mlir::NamedAttribute(
        builder->getStringAttr("noinline"),
        builder->getBoolAttr(false)
      )
    };

    return this->create<mlir::triton::FuncOp>(funcName, functy, attrs);
  }

  mlir::Value createGetProgramId(int axis) {
    return create<mlir::triton::GetProgramIdOp>(
      builder->getI32Type(),
      mlir::triton::ProgramIDDimAttr::get(
        builder->getContext(),
        mlir::triton::ProgramIDDim(axis)
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
    return create<mlir::triton::MakeRangeOp>(
      mlir::RankedTensorType::get(
        {end - start}, builder->getI32Type()
      ),
      start,
      end
    );
  }

  mlir::Value createSplat(mlir::Value arg, std::vector<int64_t> shape) {
    return createOrFold<mlir::triton::SplatOp>(
      mlir::RankedTensorType::get(shape, arg.getType()), arg
    );
  }

  mlir::Value createExpandDims(mlir::Value arg, int axis) {
    mlir::RankedTensorType argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
    mlir::Type argEltType = argType.getElementType();
    std::vector<int64_t> retShape = argType.getShape();
    retShape.insert(retShape.begin() + axis, 1);
    return create<mlir::triton::ExpandDimsOp>(
      mlir::RankedTensorType::get(retShape, argEltType),
      arg,
      axis
    );
  }

  mlir::Value createBroadcast(mlir::Value arg, std::vector<int64_t> shape) {
    mlir::RankedTensorType argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
    return createOrFold<mlir::triton::BroadcastOp>(
      mlir::RankedTensorType::get(shape, argType.getElementType()),
      arg);
  }

  mlir::Value createCmpI(const std::string& opstr, mlir::Value lhs, mlir::Value rhs) {
    mlir::arith::CmpIPredicate op = cmpIStrToPredicate(opstr);
    return create<mlir::arith::CmpIOp>(
      op, lhs, rhs);
  }

  mlir::Value createAndI(mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::arith::AndIOp>(lhs, rhs);
  }

  mlir::Value createAddPtr(mlir::Value ptr, mlir::Value offset) {
    return create<mlir::triton::AddPtrOp>(ptr.getType(), ptr, offset);
  }

  mlir::Value createLoad(mlir::Value ptr, mlir::Value mask) {
    return create<mlir::triton::LoadOp>(
      ptr, mask, mlir::Value(), mlir::triton::CacheModifier::NONE,
      mlir::triton::EvictionPolicy::NORMAL, false);
  }

  void createStore(mlir::Value ptr, mlir::Value val, mlir::Value mask) {
    create<mlir::triton::StoreOp>(ptr, val, mask, 
      mlir::triton::CacheModifier::NONE,
      mlir::triton::EvictionPolicy::NORMAL
    );
  }

  mlir::OpState createReturn(std::vector<mlir::Value> vals) {
    return create<mlir::triton::ReturnOp>(vals);
  }

  mlir::OpState createReduce(std::vector<mlir::Value> operands, int axis) {
    return create<mlir::triton::ReduceOp>(operands, axis);
  }

  mlir::OpState createReduceReturn(std::vector<mlir::Value> args) {
    return create<mlir::triton::ReduceReturnOp>(args);
  }

  // note that return type is mlir::OpState rather than mlir::Value.
  mlir::OpState createCall(mlir::triton::FuncOp func, std::vector<mlir::Value> args) {
    return create<mlir::triton::CallOp>(func, args);
  }

  mlir::Type i32ty() {
    return builder->getI32Type();
  }
  mlir::Type f32ty() {
    return builder->getF32Type();
  }
  mlir::Type f32pty() {
    return mlir::triton::PointerType::get(f32ty(), 1);
  }
  mlir::RankedTensorType blockty(std::vector<int64_t> shape, mlir::Type elemTy) {
    return mlir::RankedTensorType::get(shape, elemTy);
  }

  mlir::OpBuilder& getBuilder() { return *builder; }
 private:
  std::unique_ptr<mlir::OpBuilder> builder;
};

}
