# Key Classes

llvm/include/llvm/ADT/APInt.h
- APInt

mlir/include/mlir/Pass/Pass.h
- OperationPass
- Pass

mlir/include/mlir/IR/Builders.h
- OpBuilder

mlir/include/mlir/IR/BuiltinAttributes.td
- IntegerAttr

mlir/include/mlir/IR/BuiltinOps.td
- ModuleOp

mlir/include/mlir/IR/BuiltinTypes.td
- IntegerType
- RankedTensorType

mlir/include/mlir/IR/Operation.h
- Operation

mlir/include/mlir/IR/PatternMatch.h
- PatternRewriter
- RewriterBase
- RewritePattern
- RewritePatternSet

mlir/include/mlir/Pass/PassManager.h
- PassManager

mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h
- FrozenRewritePatternSet

mlir/include/mlir/Support/LogicalResult.h
- LogicalResult

mlir/include/mlir/Transforms/DialectConversion.h
- applyPartialConversion
- ConversionPattern
- ConversionPatternRewriter
- ConversionTarget
- OpConversionPattern
  - OpAdaptor = SourceOp::Adaptor
- TypeConverter

mlir/include/mlir/Transforms/Passes.h
- createCSEPass
- createInlinerPass

mlir/lib/Transforms/Utils/DialectConversion.cpp
- applyPartialConversion

# Scratch
