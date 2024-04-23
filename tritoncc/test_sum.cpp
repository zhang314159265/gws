#include <iostream>

#include "tritoncc/LocationOpBuilder.h"
#include "tritoncc/stage/pipeline.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "tritoncc/legacy/MLIRUtil.h"

using namespace tritoncc;

mlir::triton::FuncOp createSumCombineFunc(mlir::MLIRContext& ctx, mlir::ModuleOp& M) {
  LocationOpBuilder B(&ctx);
  mlir::triton::FuncOp F = B.createFuncOp("sum_combine",
    {B.f32ty(), B.f32ty()},
    {B.f32ty()},
    false);
  M.push_back(F);
  B.setInsertionPointToStart(F.addEntryBlock());

  B.createReturn({B.createAddF(F.getArgument(0), F.getArgument(1))});
  return F;
}

// NOTE: each call to CodeGenerator create a separate builder object
mlir::triton::FuncOp createSumReduceFunc(mlir::MLIRContext& ctx, mlir::ModuleOp& M) {
  LocationOpBuilder B(&ctx);
  mlir::triton::FuncOp F = B.createFuncOp("sum_reduce",
    {B.blockty({2, 1024}, B.f32ty())},
    {B.blockty({2}, B.f32ty())},
    false);
  M.push_back(F);
  B.setInsertionPointToStart(F.addEntryBlock());

  mlir::OpState reduce_op = B.createReduce({F.getArgument(0)}, 1);
  mlir::Region& region = reduce_op->getRegion(0);
  {
    mlir::OpBuilder::InsertPoint orig_ip =  B.getBuilder().saveInsertionPoint();
    mlir::Block* block = B.getBuilder().createBlock(
      &region, {}, {B.f32ty(), B.f32ty()}, {B.getLastLoc(), B.getLastLoc()});

    mlir::Value result = B.createCall(
      createSumCombineFunc(ctx, M),
      {block->getArgument(0), block->getArgument(1)})->getResult(0);
    B.createReduceReturn({result});
    B.getBuilder().restoreInsertionPoint(orig_ip);
  }

  B.createReturn(std::vector<mlir::Value>{reduce_op->getResult(0)});
  return F;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  mlir::MLIRContext ctx;
  LocationOpBuilder builder(&ctx);
  LocationOpBuilder& B = builder;
  mlir::ModuleOp module = builder.create<mlir::ModuleOp>();

  tritoncc::loadDialects(ctx);

  mlir::Type i32ty = builder.getBuilder().getI32Type();
  mlir::Type f32ty = builder.getBuilder().getF32Type();
  mlir::Type f32pty = mlir::triton::PointerType::get(f32ty, 1);

  mlir::triton::FuncOp funcop = builder.createFuncOp("sum_fn", {f32pty, f32pty, i32ty, i32ty}, {}, true);
  {
    auto attrTy = IntegerType::get(&ctx, 32);
    funcop.setArgAttr(0, "tt.divisibility", IntegerAttr::get(attrTy, 16));
    funcop.setArgAttr(1, "tt.divisibility", IntegerAttr::get(attrTy, 16));
    funcop.setArgAttr(2, "tt.divisibility", IntegerAttr::get(attrTy, 16));
    funcop.setArgAttr(3, "tt.divisibility", IntegerAttr::get(attrTy, 16));
  }
  module.push_back(funcop);
  B.setInsertionPointToStart(funcop.addEntryBlock());

  mlir::Value pid = B.createGetProgramId(0);
  mlir::Value c2 = B.getConstant(2);

  mlir::Value xidx = B.createAddI(
    B.createSplat(B.createMulI(pid, c2), {2}),
    B.createMakeRange(0, 2)
  );

  mlir::Value ridx = B.createMakeRange(0, 1024);

  mlir::Value mask = B.createAndI(
    B.createBroadcast(
      B.createCmpI(
        "slt",
        B.createExpandDims(xidx, 1),
        B.createSplat(funcop.getArgument(2), {2, 1})
      ),
      {2, 1024}
    ),
    B.createBroadcast(
      B.createCmpI(
        "slt",
        B.createExpandDims(ridx, 0),
        B.createSplat(funcop.getArgument(3), {1, 1024})
      ),
      {2, 1024}
    )
  );

  mlir::Value load_ptr =
  B.createAddPtr(
    B.createBroadcast(
      B.createAddPtr(
        B.createSplat(funcop.getArgument(0), {2, 1}),
        B.createMulI(
          B.createExpandDims(xidx, 1),
          B.createSplat(funcop.getArgument(3), {2, 1})
        )
      ),
      {2, 1024}
    ),
    B.createBroadcast(
      B.createExpandDims(ridx, 0),
      {2, 1024}
    )
  );

  mlir::Value x = B.createLoad(load_ptr, mask);

  mlir::Value y = B.createCall(
    createSumReduceFunc(ctx, module),
    {x})->getResult(0);

  // store
  B.createStore(
    B.createAddPtr(
      B.createSplat(funcop.getArgument(1), {2}),
      xidx
    ),
    y,
    B.createCmpI("slt",
      xidx,
      B.createSplat(funcop.getArgument(2), {2})
    )
  );


  B.create<mlir::triton::ReturnOp>(std::vector<mlir::Value>());
  module.dump();
  Option opt{
     .num_warps=4, // how is this decided?
     .num_ctas=1,
     .capability=90, // H100
  };
  std::string cubinBytes = compile(module, opt);
  #if 0 // make_llir passes a string to make_ptx
  std::cout << "After optimize:" << std::endl;
  module.dump();
  #endif

  std::cout << "sum bye" << std::endl;
  return 0;
}
