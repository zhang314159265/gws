# XXX non-portable definitions BEGIN
LLVM_ROOT := $(HOME)/.triton/llvm/llvm-4017f04e-ubuntu-x64
TRITON_SRC_ROOT := $(HOME)/ws/triton
TRITON_BUILD_ROOT := $(TRITON_SRC_ROOT)/python/build/cmake.linux-x86_64-cpython-3.10
LIBPYTHON_DIR := /home/shunting/ws/miniconda3/envs/pytorch/lib
LIBPYTHON := $(LIBPYTHON_DIR)/libpython3.10.so
# XXX non-portable definitions END

CFLAGS := -I$(LLVM_ROOT)/include -I$(TRITON_SRC_ROOT)/include -I$(TRITON_BUILD_ROOT)/include
LDFLAGS := -L$(LLVM_ROOT)/lib

# Newer triton does not build .a any more. All symbols can be found in libtriton.so
# LDFLAGS += -L$(TRITON_BUILD_ROOT)/lib/Dialect/Triton/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/Transforms -L$(TRITON_BUILD_ROOT)/lib/Analysis -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonNvidiaGPU/IR

# mlir libs
LDFLAGS += -lMLIRIR -lMLIRDialect -lMLIRSupport -lMLIRControlFlowInterfaces -lMLIRInferTypeOpInterface -lMLIRControlFlowDialect -lMLIRArithDialect -lMLIRInferIntRangeCommon -lMLIRInferIntRangeInterface -lMLIRUBDialect -lMLIRCastInterfaces -lMLIRSCFDialect -lMLIRArithUtils -lMLIRComplexDialect -lMLIRDialectUtils -lMLIRTensorDialect -lMLIRDestinationStyleOpInterface -lMLIRAffineDialect -lMLIRMemRefDialect -lMLIRSideEffectInterfaces -lMLIRViewLikeInterface -lMLIRLoopLikeInterface -lMLIRShapedOpInterfaces -lMLIRParallelCombiningOpInterface -lMLIRMathDialect -lMLIRGPUDialect -lMLIRDLTIDialect -lMLIRDataLayoutInterfaces -lMLIRFunctionInterfaces -lMLIRAnalysis -lMLIRCallInterfaces

# llvm libs
LDFLAGS += -lLLVMCore -lLLVMSupport -lLLVMDemangle

# triton static libs
# LDFLAGS += -lTritonIR -lTritonGPUIR -lTritonGPUTransforms -lTritonAnalysis -lTritonNvidiaGPUIR

# XXX need patch triton to build it with default rather hidden visibility so that symbols in libtriton.so will be availabe.
LDFLAGS += $(TRITON_SRC_ROOT)/python/triton/_C/libtriton.so $(LIBPYTHON)

OVERRIDE_LIB_PATH := $(TRITON_SRC_ROOT)/python/triton/_C:$(LIBPYTHON_DIR)

ALLFLAGS := $(CFLAGS) -Wl,--start-group $(LDFLAGS) -Wl,--end-group

test_dot:
	g++ test_dot.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out

test_add:
	g++ test_add.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
