#include <iostream>
#include <cassert>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "tritoncc/legacy/Util.h"
#include "tritoncc/legacy/ProcessPipeline.h"
#include "tritoncc/legacy/MLIRUtil.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define BLOCK_SIZE 32

using namespace tritoncc;

int main(void) {
  mlir::MLIRContext ctx;
  tritoncc::loadDialects(ctx);
  auto builder = std::make_unique<mlir::OpBuilder>(&ctx);
  auto unkloc = builder->getUnknownLoc();
  auto module = builder->create<mlir::ModuleOp>(unkloc);
  auto function_name = "pointwise_fn";
  int address_space = 1;
  assert(!module.lookupSymbol(function_name));

  ctx.loadDialect<mlir::triton::TritonDialect>();

  auto i32ty = builder->getI32Type();
  auto f32ty = builder->getF32Type();
  auto f32pty = mlir::triton::PointerType::get(f32ty, address_space);

  auto functy = builder->getFunctionType({f32pty, f32pty, f32pty, i32ty}, {}).dyn_cast<mlir::FunctionType>();

  llvm::SmallVector<mlir::NamedAttribute> attrs = {
    mlir::NamedAttribute(
      builder->getStringAttr("sym_visibility"),
      builder->getStringAttr("public")
    ),
    mlir::NamedAttribute(
      builder->getStringAttr("noinline"),
      builder->getBoolAttr(false)
    )
  };
  auto funcop = builder->create<mlir::triton::FuncOp>(unkloc, function_name, functy, attrs);
  module.push_back(funcop);

  builder->setInsertionPointToStart(funcop.addEntryBlock());

  auto pid = builder->create<mlir::triton::GetProgramIdOp>(unkloc, i32ty, mlir::triton::ProgramIDDimAttr::get(&ctx, mlir::triton::ProgramIDDim(0)));
  auto range = builder->create<mlir::triton::MakeRangeOp>(unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, i32ty),
    0, BLOCK_SIZE);

  auto broadcasted = builder->createOrFold<mlir::triton::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, i32ty),
    builder->create<mlir::arith::MulIOp>(unkloc,
      pid,
      builder->create<mlir::arith::ConstantIntOp>(unkloc, BLOCK_SIZE, i32ty))
  );

  auto idx = builder->create<mlir::arith::AddIOp>(
    unkloc,
    range,
    broadcasted
  );

  auto broadcasted_num = builder->createOrFold<mlir::triton::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, i32ty),
    funcop.getArgument(3)
  );

  auto mask = builder->create<mlir::arith::CmpIOp>(
    unkloc,
    mlir::arith::CmpIPredicate::slt,
    idx,
    broadcasted_num
  );

  // lhs_ptr
  auto lhs_ptr = builder->createOrFold<mlir::triton::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(0)
  );
    
  // load lhs
  auto lhs = builder->create<mlir::triton::LoadOp>(
    unkloc,
    builder->create<mlir::triton::AddPtrOp>(
      unkloc,
      lhs_ptr.getType(),
      lhs_ptr,
      idx
    ),
    mask,
    mlir::Value(),
    mlir::triton::CacheModifier::NONE,
    mlir::triton::EvictionPolicy::NORMAL,
    false
  );

  // rhs_ptr
  auto rhs_ptr = builder->createOrFold<mlir::triton::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(1)
  );
    
  // load rhs
  auto rhs = builder->create<mlir::triton::LoadOp>(
    unkloc,
    builder->create<mlir::triton::AddPtrOp>(
      unkloc,
      rhs_ptr.getType(),
      rhs_ptr,
      idx
    ),
    mask,
    mlir::Value(),
    mlir::triton::CacheModifier::NONE,
    mlir::triton::EvictionPolicy::NORMAL,
    false
  );

  // compute ans
  auto ans = builder->create<mlir::arith::AddFOp>(
    unkloc,
    lhs,
    rhs
  );

  // ans_ptr
  auto ans_ptr = builder->createOrFold<mlir::triton::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(2)
  );

  // store ans
  builder->create<mlir::triton::StoreOp>(
    unkloc,
    builder->create<mlir::triton::AddPtrOp>(
      unkloc,
      ans_ptr.getType(),
      ans_ptr,
      idx
    ),
    ans,
    mask,
    mlir::triton::CacheModifier::NONE,
    mlir::triton::EvictionPolicy::NORMAL
  );

  builder->create<mlir::triton::ReturnOp>(unkloc);

  module.dump();
  Option opt{
     .num_warps=4, // how is this decided?
     .num_ctas=1,
     .capability=90, // H100
  };
  processPipeline(module, opt);

  std::cout << "bye" << std::endl;
  return 0;
}
