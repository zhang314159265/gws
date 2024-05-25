#ifndef TOY_PATTERN
#define TOY_PATTERN

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<mlir::ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;


#endif
