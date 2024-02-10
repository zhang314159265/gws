# XXX non-portable definitions BEGIN
LLVM_ROOT := $(HOME)/.triton/llvm/llvm-4017f04e-ubuntu-x64
TRITON_SRC_ROOT := $(HOME)/ws/triton
TRITON_BUILD_ROOT := $(TRITON_SRC_ROOT)/python/build/cmake.linux-x86_64-cpython-3.10
LIBPYTHON_DIR := /home/shunting/ws/miniconda3/envs/pytorch/lib
LIBPYTHON := $(LIBPYTHON_DIR)/libpython3.10.so
# XXX non-portable definitions END

CFLAGS := -I$(LLVM_ROOT)/include -I$(TRITON_SRC_ROOT)/include -I$(TRITON_BUILD_ROOT)/include -I./include -I$(TRITON_SRC_ROOT) -I$(TRITON_SRC_ROOT)/lib/Conversion/TritonGPUToLLVM -fno-rtti
LDFLAGS := -L$(LLVM_ROOT)/lib

# Newer triton does not build .a any more. All symbols can be found in libtriton.so
# LDFLAGS += -L$(TRITON_BUILD_ROOT)/lib/Dialect/Triton/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/Transforms -L$(TRITON_BUILD_ROOT)/lib/Analysis -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonNvidiaGPU/IR

# mlir libs
LDFLAGS += -lMLIRIR -lMLIRDialect -lMLIRSupport -lMLIRControlFlowInterfaces -lMLIRInferTypeOpInterface -lMLIRControlFlowDialect -lMLIRArithDialect -lMLIRInferIntRangeCommon -lMLIRInferIntRangeInterface -lMLIRUBDialect -lMLIRCastInterfaces -lMLIRSCFDialect -lMLIRArithUtils -lMLIRComplexDialect -lMLIRDialectUtils -lMLIRTensorDialect -lMLIRDestinationStyleOpInterface -lMLIRAffineDialect -lMLIRMemRefDialect -lMLIRSideEffectInterfaces -lMLIRViewLikeInterface -lMLIRLoopLikeInterface -lMLIRShapedOpInterfaces -lMLIRParallelCombiningOpInterface -lMLIRMathDialect -lMLIRGPUDialect -lMLIRDLTIDialect -lMLIRDataLayoutInterfaces -lMLIRFunctionInterfaces -lMLIRAnalysis -lMLIRCallInterfaces

# llvm libs
LDFLAGS += -lLLVMCore -lLLVMSupport -lLLVMDemangle

# triton static libs
# LDFLAGS += -lTritonIR -lTritonGPUIR -lTritonGPUTransforms -lTritonAnalysis -lTritonNvidiaGPUIR

# Use libtriton.so may result in the following error:
#   CommandLine Error: Option 'use-dereferenceable-at-point-semantics' registered more than once!
# Most likely due to LLVM libraries being linked twice.
ifdef USE_LIBTRITON
# XXX need patch triton to build it with default rather hidden visibility so that symbols in libtriton.so will be availabe.
LDFLAGS += $(TRITON_SRC_ROOT)/python/triton/_C/libtriton.so
LDFLAGS += $(LIBPYTHON)
else
LDFLAGS += -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMTargetParser -lMLIRPass -lMLIRTransformUtils -lMLIRRewrite -lMLIRRewritePDL
LDFLAGS += -lMLIRPDLInterpDialect -lMLIRPDLDialect -lMLIRPDLToPDLInterp -lMLIRLLVMCommonConversion -lMLIRLLVMDialect -lMLIRMemorySlotInterfaces
LDFLAGS += -lMLIRTransforms -lMLIRNVVMDialect -lMLIRFuncToLLVM -lMLIRControlFlowToSCF -lMLIRFuncDialect -lMLIRArithToLLVM -lMLIRArithAttrToLLVMConversion
LDFLAGS += -lMLIRControlFlowToLLVM -lMLIRGPUToNVVMTransforms -lMLIRGPUToGPURuntimeTransforms -lMLIRVectorToLLVM -lMLIRVectorDialect -lMLIRValueBoundsOpInterface
LDFLAGS += -lMLIRPresburger -lMLIRMaskingOpInterface -lMLIRMaskableOpInterface -lMLIRVectorInterfaces -lMLIRVectorTransforms -lMLIRTargetLLVMIRExport
LDFLAGS += -lMLIRMemRefToLLVM -lMLIRMemRefUtils -lMLIRGPUTransforms -lMLIRIndexDialect -lMLIRMathToLLVM
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Analysis/CMakeFiles/TritonAnalysis.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ConvertLayoutOpToLLVM/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/DotOpToLLVM/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Conversion/TritonToTritonGPU/CMakeFiles/TritonToTritonGPU.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/Triton/Transforms/CMakeFiles/TritonTransforms.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/lib/Dialect/NVGPU/IR/CMakeFiles/NVGPUIR.dir/*.o
# LDFLAGS += $(TRITON_BUILD_ROOT)
endif

OVERRIDE_LIB_PATH := $(TRITON_SRC_ROOT)/python/triton/_C:$(LIBPYTHON_DIR)

ALLFLAGS := $(CFLAGS) -Wl,--start-group $(LDFLAGS) -Wl,--end-group

first: test_sum

test_sum:
	g++ test_sum.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out

test_dot:
	g++ test_dot.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out

test_add:
	g++ test_add.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
