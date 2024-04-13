# Key Classes

llvm/include/llvm/ADT/APInt.h
- APInt

llvm/include/llvm/ADT/MapVector.h
- MapVector # like OrderedDict in python

llvm/include/llvm/ADT/SetVector.h
- SmallSetVector

llvm/include/llvm/Support/CommandLine.h
- ParseCommandLineOptions

llvm/include/llvm/Support/Debug.h
- dbgs
- `LLVM_DEBUG`

mlir/include/mlir/Conversion/LLVMCommon/Pattern.h
- ConvertOpToLLVMPattern
- ConvertToLLVMPattern

mlir/include/mlir/Conversion/LLVMCommon/TypeConverter.h
- LLVMTypeConverter

mlir/include/mlir/Dialect/LLVMIR/LLVMTypes.h
- LLVMStructType

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

mlir/include/mlir/IR/IRMapping.h
- IRMapping

mlir/include/mlir/IR/Operation.h
- Operation

mlir/include/mlir/IR/PatternMatch.h
- IRRewriter
- PatternRewriter
- RewriterBase
- RewritePattern
- RewritePatternSet

mlir/include/mlir/IR/Value.h
- class Value

mlir/include/mlir/IR/ValueRange.h
- class ValueRange

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

mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h
- applyPatternsAndFoldGreedily

mlir/include/mlir/Transforms/Passes.h
- createCSEPass
- createInlinerPass

mlir/lib/Transforms/Utils/DialectConversion.cpp
- applyPartialConversion

# Scratch
