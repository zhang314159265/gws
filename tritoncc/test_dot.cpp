#include <iostream>

#include "mlir/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "tritoncc/ProcessTTIR.h"

using namespace tritoncc;

mlir::Type i32ty, f32ty, f32pty;
std::unique_ptr<mlir::OpBuilder> builder;
mlir::Location* punkloc;
mlir::Value icst32, icst5, fp32cst0;
mlir::MLIRContext *pctx;
mlir::triton::FuncOp funcop;
mlir::ModuleOp module;

void optimize() {
  processTTIR(module);
}

mlir::Value build_offset() {
  // range for row
  mlir::Value range1 = builder->create<mlir::triton::MakeRangeOp>(*punkloc,
    mlir::RankedTensorType::get({32}, i32ty), 0, 32);

  mlir::Value exp_range1 = builder->create<mlir::triton::ExpandDimsOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 1}, i32ty),
    range1,
    1
  );

  // stride
  mlir::Value bcast_stride = builder->createOrFold<mlir::triton::SplatOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 1}, i32ty),
    icst32
  );

  // range for column
  mlir::Value range2 = builder->create<mlir::triton::MakeRangeOp>(*punkloc,
    mlir::RankedTensorType::get({32}, i32ty), 0, 32);

  mlir::Value exp_range2 = builder->create<mlir::triton::ExpandDimsOp>(
    *punkloc,
    mlir::RankedTensorType::get({1, 32}, i32ty),
    range2,
    0
  );

  // build offset
  mlir::Value row_scale = builder->create<mlir::arith::MulIOp>(
    *punkloc,
    exp_range1,
    bcast_stride
  );
  return builder->create<mlir::arith::AddIOp>(
    *punkloc,
    builder->createOrFold<mlir::triton::BroadcastOp>(
      *punkloc,
      mlir::RankedTensorType::get({32, 32}, i32ty),
      row_scale
    ),
    builder->createOrFold<mlir::triton::BroadcastOp>(
      *punkloc,
      mlir::RankedTensorType::get({32, 32}, i32ty),
      exp_range2
    )
  );
}

mlir::Value build_load(int argIdx, mlir::Value offset) {
  mlir::Value arg = funcop.getArgument(argIdx);
  mlir::Value splat_arg = builder->createOrFold<mlir::triton::SplatOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 32}, f32pty),
    arg);
  mlir::Value ptr = builder->create<mlir::triton::AddPtrOp>(
    *punkloc,
    splat_arg.getType(),
    splat_arg,
    offset);

  return builder->create<mlir::triton::LoadOp>(
    *punkloc,
    ptr,
    mlir::triton::CacheModifier::NONE,
    mlir::triton::EvictionPolicy::NORMAL,
    false);
}

mlir::Value build_dot(mlir::Value lhs, mlir::Value rhs) {
  mlir::Value splat_0 = builder->createOrFold<mlir::triton::SplatOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 32}, f32ty),
    fp32cst0
  );
  mlir::Value dot = builder->create<mlir::triton::DotOp>(
    *punkloc,
    splat_0.getType(),
    lhs,
    rhs,
    splat_0,
    true,
    0
  );
  return dot;
}

mlir::Value build_plus_scalar(mlir::Value dot) {
  mlir::Value splat_5 = builder->createOrFold<mlir::triton::SplatOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 32}, i32ty),
    icst5
  );
  mlir::Value splat_fp5 = builder->create<mlir::arith::SIToFPOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 32}, f32ty),
    splat_5
  );
  return builder->create<mlir::arith::AddFOp>(
    *punkloc,
    dot,
    splat_fp5
  );
}

void build_store(mlir::Value ans, mlir::Value offset) {
  mlir::Value arg = funcop.getArgument(2);
  mlir::Value splat_arg = builder->createOrFold<mlir::triton::SplatOp>(
    *punkloc,
    mlir::RankedTensorType::get({32, 32}, f32pty),
    arg
  );
  mlir::Value ptr = builder->create<mlir::triton::AddPtrOp>(
    *punkloc,
    splat_arg.getType(),
    splat_arg,
    offset);
  builder->create<mlir::triton::StoreOp>(
    *punkloc,
    ptr,
    ans,
    mlir::triton::CacheModifier::NONE,
    mlir::triton::EvictionPolicy::NORMAL
  );
}

int main(void) {
  mlir::MLIRContext ctx;
  pctx = &ctx;
  builder = std::move(std::make_unique<mlir::OpBuilder>(&ctx));
  auto unkloc = builder->getUnknownLoc();
  punkloc = &unkloc;
  module = builder->create<mlir::ModuleOp>(unkloc);
  auto function_name = "dot_fn";
  int address_space = 1;

  ctx.loadDialect<mlir::triton::TritonDialect>();
  i32ty = builder->getI32Type();
  f32ty = builder->getF32Type();
  f32pty = mlir::triton::PointerType::get(f32ty, address_space);

  auto functy = builder->getFunctionType({f32pty, f32pty, f32pty}, {}).dyn_cast<mlir::FunctionType>();
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
  funcop = builder->create<mlir::triton::FuncOp>(unkloc, function_name, functy, attrs);
  module.push_back(funcop);
  builder->setInsertionPointToStart(funcop.addEntryBlock());

  icst32 = builder->create<mlir::arith::ConstantIntOp>(unkloc, 32, i32ty);
  icst5 = builder->create<mlir::arith::ConstantIntOp>(unkloc, 5, i32ty);
  fp32cst0 = builder->create<mlir::arith::ConstantOp>(
    unkloc,
    builder->getF32FloatAttr(0.0f)
  );
  auto off = build_offset();
  auto lhs = build_load(0, off);
  auto rhs = build_load(1, off);
  build_store(build_plus_scalar(build_dot(lhs, rhs)), off);

  builder->create<mlir::triton::ReturnOp>(unkloc);
  module.dump();
  optimize();
  std::cout << "After optimize:" << std::endl;
  module.dump();
  std::cout << "bye dot" << std::endl;
  return 0;
}
