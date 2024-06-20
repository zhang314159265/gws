TORCH_ROOT = $(HOME)/ws/pytorch
OVERRIDE_TORCH_LIB_PATH := $(TORCH_ROOT)/torch/lib

# XXX non-portable definitions BEGIN
# USE_NEW_LLVM := 1
ifdef USE_NEW_LLVM
LLVM_ROOT := $(HOME)/.triton/llvm/llvm-5e5a22ca-ubuntu-x64
else
LLVM_ROOT := $(HOME)/old.triton/llvm/llvm-4017f04e-ubuntu-x64
endif
TBLGEN := ${LLVM_ROOT}/bin/mlir-tblgen

# TODO: don't add path in triton/
TBLGEN_INC_FLAGS := -I${LLVM_ROOT}/include -I../include
TRITON_SRC_ROOT := $(HOME)/ws/triton
TRITON_BUILD_ROOT := $(TRITON_SRC_ROOT)/python/build/cmake.linux-x86_64-cpython-3.10
LIBPYTHON_DIR := /home/shunting/ws/miniconda3/envs/pytorch/lib

PTXAS := $(TRITON_SRC_ROOT)/third_party/nvidia/backend/bin/ptxas

CFLAGS := -I$(LLVM_ROOT)/include -I../include -I../out/include -fno-rtti
LDFLAGS := -L$(LLVM_ROOT)/lib

# misc libs
LDFLAGS += -lz

OVERRIDE_LIB_PATH := $(TRITON_SRC_ROOT)/python/triton/_C:$(LIBPYTHON_DIR)

include ../../llvm_or_mlir_libs.make
LDFLAGS += ${LLVM_OR_MLIR_LIBS}

ALLFLAGS := $(CFLAGS) -Wl,--start-group $(LDFLAGS) -Wl,--end-group

# first: run_add_ptx
# first: run_sum_ptx
# first: run_dot_ptx
# first: test_load_add_cubin_and_run
# first: test_load_dot_cubin_and_run
# first: test_dot
first: test_sum
# first: test_add  # does not generate any anchor nodes for remove-layout-conversion due to small block size
# first: tritoncc

all: test_add test_sum

tritoncc:
	time g++ tritoncc.cpp $(ALLFLAGS) -o out/tritoncc
	# out/tritoncc skip-checkin/add_ref.ttir

TRITONCC_INC_DIR := ../include
TRITONCC_OUT_DIR := ../out

run_tblgen:
	mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU
	mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU
	mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU
	mkdir -p ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/Dialect.cpp.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonNvidiaGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonNvidiaGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Dialect.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/Ops.cpp.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/NVGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/NVGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Dialect.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/Ops.cpp.inc
	${TBLGEN} -gen-attrdef-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrDefs.h.inc
	${TBLGEN} -gen-attrdef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrDefs.cpp.inc
	${TBLGEN} -gen-attr-interface-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrInterfaces.h.inc
	${TBLGEN} -gen-attr-interface-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/TritonGPU/AttrDefs.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/TritonGPU/AttrInterfaces.cpp.inc
	${TBLGEN} -gen-op-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Ops.h.inc
	${TBLGEN} -gen-op-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Ops.cpp.inc
	${TBLGEN} -gen-dialect-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Dialect.h.inc
	${TBLGEN} -gen-dialect-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Dialect.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Dialect.cpp.inc
	${TBLGEN} -gen-enum-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/OpsEnums.h.inc
	${TBLGEN} -gen-enum-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Ops.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/OpsEnums.cpp.inc
	${TBLGEN} -gen-typedef-decls ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Types.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Types.h.inc
	${TBLGEN} -gen-typedef-defs ${TBLGEN_INC_FLAGS} ${TRITONCC_INC_DIR}/tritoncc/dialect/Triton/Types.td -o ${TRITONCC_OUT_DIR}/include/tritoncc/dialect/Triton/Types.cpp.inc

test_sum: run_tblgen
	g++ test_sum.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out -debug-only=convert-layout-op-to-llvm
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run /tmp/tritoncc.cubin

# XXX my generted dot ptx does not contains ldmatrix/mma instr. It contains
# a lot of fma instr. Looks like the matmul is implemented naively rather than
# leveraging specialized HW instrs.
test_dot:
	g++ test_dot.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run /tmp/tritoncc.cubin

test_add: run_tblgen
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
	g++ test_load_sum_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -I../include -o out/test_load_sum_cubin_and_run
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
