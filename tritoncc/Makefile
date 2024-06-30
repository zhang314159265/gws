TRITON_SRC_ROOT := $(HOME)/ws/triton

TORCH_ROOT = $(HOME)/ws/pytorch
TORCH_CFLAGS = -I$(TORCH_ROOT)/torch/include -I$(TORCH_ROOT)/torch/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_ROOT)/torch/lib -lc10 -ltorch_cpu -ltorch_cuda

CUDA_CFLAGS = -I$(TRITON_SRC_ROOT)/third_party/nvidia/backend/include
CUDA_LDFLAGS = -lcuda

LLVM_ROOT := $(HOME)/old.triton/llvm/llvm-4017f04e-ubuntu-x64

first: ext

TRITONCC_ROOT := .
include ../llvm_or_mlir_libs.make
include tblgen.make
TRITONCC_CFLAGS := -Iinclude -I$(LLVM_ROOT)/include -Iout/include -fno-rtti
TRITONCC_CFLAGS += -fPIC  # need this for pybind
# PyBind requires rtti?
PYBIND_CFLAGS := -Iinclude -I$(LLVM_ROOT)/include -Iout/include
TRITONCC_LDFLAGS := -L$(LLVM_ROOT)/lib -lz -Wl,--start-group $(LLVM_OR_MLIR_LIBS) -Wl,--end-group

RUNTIME_TORCH_LIB_PATH := $(TORCH_ROOT)/torch/lib

build_test_load_sum_cubin_and_run:
	g++ test_load_sum_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -Iinclude -o out/test_load_sum_cubin_and_run

build_triton_dialect_o:
	@mkdir -p out/lib/dialect/Triton
	time g++ lib/dialect/Triton/Dialect.cpp $(TRITONCC_CFLAGS) -c -o out/lib/dialect/Triton/Dialect.o

build_triton_gpu_dialect_o:
	@mkdir -p out/lib/dialect/TritonGPU
	time g++ lib/dialect/TritonGPU/Dialect.cpp $(TRITONCC_CFLAGS) -c -o out/lib/dialect/TritonGPU/Dialect.o

build_triton_nvidia_gpu_dialect_o:
	@mkdir -p out/lib/dialect/TritonNvidiaGPU
	time g++ lib/dialect/TritonNvidiaGPU/Dialect.cpp $(TRITONCC_CFLAGS) -c -o out/lib/dialect/TritonNvidiaGPU/Dialect.o

build_nvgpu_dialect_o:
	@mkdir -p out/lib/dialect/NVGPU
	time g++ lib/dialect/NVGPU/Dialect.cpp $(TRITONCC_CFLAGS) -c -o out/lib/dialect/NVGPU/Dialect.o

build_dialects: build_triton_dialect_o build_triton_gpu_dialect_o build_triton_nvidia_gpu_dialect_o build_nvgpu_dialect_o
DIALECT_OBJ_LIST := out/lib/dialect/Triton/Dialect.o out/lib/dialect/TritonGPU/Dialect.o out/lib/dialect/TritonNvidiaGPU/Dialect.o out/lib/dialect/NVGPU/Dialect.o

build_test_sum_o:
	time g++ test_sum.cpp $(TRITONCC_CFLAGS) -c -o out/test_sum.o

test_sum: run_tblgen build_test_load_sum_cubin_and_run build_test_sum_o build_dialects
	time g++ out/test_sum.o $(DIALECT_OBJ_LIST) $(TRITONCC_LDFLAGS) -o out/test_sum
	out/test_sum
	LD_LIBRARY_PATH=$(RUNTIME_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run /tmp/tritoncc.cubin

build_test_add_o:
	time g++ test_add.cpp $(TRITONCC_CFLAGS) -c -o out/test_add.o

test_add: run_tblgen out/test_load_add_cubin_and_run build_test_add_o build_dialects
	time g++ out/test_add.o $(DIALECT_OBJ_LIST) $(TRITONCC_LDFLAGS) -o out/test_add
	out/test_add
	LD_LIBRARY_PATH=$(RUNTIME_TORCH_LIB_PATH) out/test_load_add_cubin_and_run /tmp/tritoncc.cubin

out/test_load_add_cubin_and_run: test_load_add_cubin_and_run.cpp include/tritoncc/runtime.h
	g++ test_load_add_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -Iinclude -o out/test_load_add_cubin_and_run

build_tritoncc_o:
	time g++ tritoncc.cpp $(TRITONCC_CFLAGS) -c -o out/tritoncc.o
	
tritoncc: build_tritoncc_o build_dialects
	time g++ out/tritoncc.o $(DIALECT_OBJ_LIST) $(TRITONCC_LDFLAGS) -o out/tritoncc

# NOTE: not depend on build_dialects and assume dialect .o's are up to date
# It's for faster build time
B := 1
ext:
ifeq ($(B), 1)
	time g++ pybind.cpp -shared -fPIC $(PYBIND_CFLAGS) $(DIALECT_OBJ_LIST) $(TRITONCC_LDFLAGS) $(shell python -m pybind11 --includes) -o _C.so
endif
	PYTHONPATH=`realpath ..` python test_fe_add.py
