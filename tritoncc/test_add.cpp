#include <iostream>
#include <cassert>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "tritoncc/stage/pipeline.h"
#include "tritoncc/dialect_util.h"

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

  ctx.loadDialect<mlir::_tritoncc::TritonDialect>();

  auto i32ty = builder->getI32Type();
  auto f32ty = builder->getF32Type();
  auto f32pty = mlir::_tritoncc::PointerType::get(f32ty, address_space);

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
  auto funcop = builder->create<mlir::_tritoncc::FuncOp>(unkloc, function_name, functy, attrs);
  module.push_back(funcop);

  builder->setInsertionPointToStart(funcop.addEntryBlock());

  auto pid = builder->create<mlir::_tritoncc::GetProgramIdOp>(unkloc, i32ty, mlir::_tritoncc::ProgramIDDimAttr::get(&ctx, mlir::_tritoncc::ProgramIDDim(0)));
  auto range = builder->create<mlir::_tritoncc::MakeRangeOp>(unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, i32ty),
    0, BLOCK_SIZE);

  auto broadcasted = builder->createOrFold<mlir::_tritoncc::SplatOp>(
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

  auto broadcasted_num = builder->createOrFold<mlir::_tritoncc::SplatOp>(
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
  auto lhs_ptr = builder->createOrFold<mlir::_tritoncc::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(0)
  );
    
  // load lhs
  auto lhs = builder->create<mlir::_tritoncc::LoadOp>(
    unkloc,
    builder->create<mlir::_tritoncc::AddPtrOp>(
      unkloc,
      lhs_ptr.getType(),
      lhs_ptr,
      idx
    ),
    mask,
    mlir::Value(),
    mlir::_tritoncc::CacheModifier::NONE,
    mlir::_tritoncc::EvictionPolicy::NORMAL,
    false
  );

  // rhs_ptr
  auto rhs_ptr = builder->createOrFold<mlir::_tritoncc::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(1)
  );
    
  // load rhs
  auto rhs = builder->create<mlir::_tritoncc::LoadOp>(
    unkloc,
    builder->create<mlir::_tritoncc::AddPtrOp>(
      unkloc,
      rhs_ptr.getType(),
      rhs_ptr,
      idx
    ),
    mask,
    mlir::Value(),
    mlir::_tritoncc::CacheModifier::NONE,
    mlir::_tritoncc::EvictionPolicy::NORMAL,
    false
  );

  // compute ans
  auto ans = builder->create<mlir::arith::AddFOp>(
    unkloc,
    lhs,
    rhs
  );

  // ans_ptr
  auto ans_ptr = builder->createOrFold<mlir::_tritoncc::SplatOp>(
    unkloc,
    mlir::RankedTensorType::get({BLOCK_SIZE}, f32pty),
    funcop.getArgument(2)
  );

  // store ans
  builder->create<mlir::_tritoncc::StoreOp>(
    unkloc,
    builder->create<mlir::_tritoncc::AddPtrOp>(
      unkloc,
      ans_ptr.getType(),
      ans_ptr,
      idx
    ),
    ans,
    mask,
    mlir::_tritoncc::CacheModifier::NONE,
    mlir::_tritoncc::EvictionPolicy::NORMAL
  );

  builder->create<mlir::_tritoncc::ReturnOp>(unkloc);

  module.dump();
  Option opt{
     .num_warps=4, // how is this decided?
     .num_ctas=1,
     .capability=90, // H100
  };
  std::string cubinBytes = compile(module, opt);

  std::cout << "bye" << std::endl;
  return 0;
}
