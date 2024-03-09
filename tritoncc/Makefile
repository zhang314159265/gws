TORCH_ROOT = $(HOME)/ws/pytorch
OVERRIDE_TORCH_LIB_PATH := $(TORCH_ROOT)/torch/lib

# XXX non-portable definitions BEGIN
# USE_NEW_LLVM := 1
ifdef USE_NEW_LLVM
LLVM_ROOT := $(HOME)/.triton/llvm/llvm-5e5a22ca-ubuntu-x64
else
LLVM_ROOT := $(HOME)/old.triton/llvm/llvm-4017f04e-ubuntu-x64
endif
TRITON_SRC_ROOT := $(HOME)/ws/triton
TRITON_BUILD_ROOT := $(TRITON_SRC_ROOT)/python/build/cmake.linux-x86_64-cpython-3.10
LIBPYTHON_DIR := /home/shunting/ws/miniconda3/envs/pytorch/lib
LIBPYTHON := $(LIBPYTHON_DIR)/libpython3.10.so
# XXX non-portable definitions END

PTXAS := $(TRITON_SRC_ROOT)/third_party/nvidia/backend/bin/ptxas

CFLAGS := -I$(LLVM_ROOT)/include -I$(TRITON_SRC_ROOT)/third_party -I$(TRITON_SRC_ROOT)/include -I$(TRITON_BUILD_ROOT)/include -I./include -I$(TRITON_SRC_ROOT) -I$(TRITON_SRC_ROOT)/lib/Conversion/TritonGPUToLLVM -I$(TRITON_SRC_ROOT)/include/triton/Conversion/TritonGPUToLLVM -fno-rtti
LDFLAGS := -L$(LLVM_ROOT)/lib

# Newer triton does not build .a any more. All symbols can be found in libtriton.so
# LDFLAGS += -L$(TRITON_BUILD_ROOT)/lib/Dialect/Triton/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/IR -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonGPU/Transforms -L$(TRITON_BUILD_ROOT)/lib/Analysis -L$(TRITON_BUILD_ROOT)/lib/Dialect/TritonNvidiaGPU/IR

# misc libs
LDFLAGS += -lz

# mlir libs
LDFLAGS += -lMLIRIR -lMLIRDialect -lMLIRSupport -lMLIRControlFlowInterfaces -lMLIRInferTypeOpInterface -lMLIRControlFlowDialect -lMLIRArithDialect -lMLIRInferIntRangeCommon -lMLIRInferIntRangeInterface -lMLIRUBDialect -lMLIRCastInterfaces -lMLIRSCFDialect -lMLIRArithUtils -lMLIRComplexDialect -lMLIRDialectUtils -lMLIRTensorDialect -lMLIRDestinationStyleOpInterface -lMLIRAffineDialect -lMLIRMemRefDialect -lMLIRSideEffectInterfaces -lMLIRViewLikeInterface -lMLIRLoopLikeInterface -lMLIRShapedOpInterfaces -lMLIRParallelCombiningOpInterface -lMLIRMathDialect -lMLIRGPUDialect -lMLIRDLTIDialect -lMLIRDataLayoutInterfaces -lMLIRFunctionInterfaces -lMLIRAnalysis -lMLIRCallInterfaces -lMLIRNVVMToLLVMIRTranslation

# llvm libs
LDFLAGS += -lLLVMCore -lLLVMSupport -lLLVMDemangle -lLLVMX86AsmParser -lLLVMAMDGPUAsmParser -lLLVMMCParser -lLLVMMC -lLLVMAMDGPUUtils -lLLVMAMDGPUInfo -lLLVMX86CodeGen -lLLVMSelectionDAG -lLLVMCodeGen -lLLVMTarget -lLLVMAnalysis -lLLVMTransformUtils -lLLVMObjCARCOpts -lLLVMCodeGenTypes -lLLVMGlobalISel -lLLVMX86Desc -lLLVMMCDisassembler -lLLVMX86Info -lLLVMScalarOpts -lLLVMIRPrinter -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMInstrumentation -lLLVMAMDGPUCodeGen -lLLVMAMDGPUDesc -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMVectorize -lLLVMipo -lLLVMFrontendOpenMP -lLLVMBitReader -lLLVMFrontendOffloading -lLLVMPasses -lLLVMHipStdPar -lLLVMMIRParser -lLLVMAsmParser -lLLVMDebugInfoDWARF -lLLVMObject -lLLVMTextAPI -lLLVMIRReader

LDFLAGS += -lLLVMProfileData -lLLVMCFGuard -lMLIRLLVMIRTransforms -lMLIRBuiltinToLLVMIRTranslation -lMLIRLLVMToLLVMIRTranslation

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

ifdef USE_NEW_LLVM
LDFLAGS += -lMLIRBufferizationDialect -lMLIRSparseTensorDialect -lMLIRLLVMDialect
else
LDFLAGS += -lMLIRRewritePDL
endif
LDFLAGS += -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMTargetParser -lMLIRPass -lMLIRTransformUtils -lMLIRRewrite
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
LDFLAGS += $(TRITON_BUILD_ROOT)/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/ConvertLayoutOpToLLVM/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/DotOpToLLVM/*.o
LDFLAGS += $(TRITON_BUILD_ROOT)/third_party/nvidia/lib/NVGPUToLLVM/CMakeFiles/NVGPUToLLVM.dir/*.o
# LDFLAGS += $(TRITON_BUILD_ROOT)
endif

OVERRIDE_LIB_PATH := $(TRITON_SRC_ROOT)/python/triton/_C:$(LIBPYTHON_DIR)

ALLFLAGS := $(CFLAGS) -Wl,--start-group $(LDFLAGS) -Wl,--end-group

# first: run_add_ptx
# first: run_sum_ptx
# first: run_dot_ptx
# first: test_load_add_cubin_and_run
# first: test_load_dot_cubin_and_run
# first: test_dot
first: test_sum

all: test_add test_sum

test_sum:
	g++ test_sum.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run /tmp/tritoncc.cubin

# XXX my generted dot ptx does not contains ldmatrix/mma instr. It contains
# a lot of fma instr. Looks like the matmul is implemented naively rather than
# leveraging specialized HW instrs.
test_dot:
	g++ test_dot.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run /tmp/tritoncc.cubin

test_add:
	time g++ test_add.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_add_cubin_and_run /tmp/tritoncc.cubin

TORCH_CFLAGS = -I$(TORCH_ROOT)/torch/include -I$(TORCH_ROOT)/torch/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_ROOT)/torch/lib
TORCH_LDFLAGS += -lc10 -ltorch_cpu -ltorch_cuda

CUDA_CFLAGS = -I$(TRITON_SRC_ROOT)/third_party/nvidia/backend/include
CUDA_LDFLAGS = -lcuda

# NOTE: this assumes /tmp/sum.cubin is already properly setup
#
# NOTE: /tmp/sum_ref.cubin passes the accuracy check! That means the launcher and runner and working correctly.
# NOTE: XXX however /tmp/sum.cubin does not pass accuracy check yet!
test_load_sum_cubin_and_run:
	g++ test_load_sum_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -I./include -o out/test_load_sum_cubin_and_run
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run skip-checkin/sum.cubin 
	# LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) ./a.out skip-checkin/sum_ref.cubin fn_0d1d2de3de

test_load_add_cubin_and_run:
	g++ test_load_add_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -I./include -o out/test_load_add_cubin_and_run
	# LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) ./a.out skip-checkin/add_ref.cubin fn_0d1d2d3de
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_add_cubin_and_run skip-checkin/add.cubin

test_load_dot_cubin_and_run:
	g++ test_load_dot_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -I./include -o out/test_load_dot_cubin_and_run
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run skip-checkin/dot_ref.cubin fn_0d1d2d

run_add_ptx:
	# $(PTXAS) --gpu-name=sm_90a add.ptx -o /tmp/tritoncc.cubin
	ptxas -arch=sm_90 add.ptx -o /tmp/tritoncc.cubin
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_add_cubin_and_run /tmp/tritoncc.cubin add_fn

run_sum_ptx:
	ptxas -arch=sm_90 sum.ptx -o /tmp/tritoncc.cubin
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run /tmp/tritoncc.cubin sum_fn

run_dot_ptx:
	ptxas -arch=sm_90 dot.ptx -o /tmp/tritoncc.cubin
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run /tmp/tritoncc.cubin dot_fn
