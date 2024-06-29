TRITON_SRC_ROOT := $(HOME)/ws/triton

TORCH_ROOT = $(HOME)/ws/pytorch
TORCH_CFLAGS = -I$(TORCH_ROOT)/torch/include -I$(TORCH_ROOT)/torch/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_ROOT)/torch/lib -lc10 -ltorch_cpu -ltorch_cuda

CUDA_CFLAGS = -I$(TRITON_SRC_ROOT)/third_party/nvidia/backend/include
CUDA_LDFLAGS = -lcuda

LLVM_ROOT := $(HOME)/old.triton/llvm/llvm-4017f04e-ubuntu-x64

first: test_sum

TRITONCC_ROOT := .
include ../llvm_or_mlir_libs.make
include tblgen.make
TRITONCC_CFLAGS := -Iinclude -I$(LLVM_ROOT)/include -Iout/include -fno-rtti
TRITONCC_LDFLAGS := -L$(LLVM_ROOT)/lib -lz -Wl,--start-group $(LLVM_OR_MLIR_LIBS) -Wl,--end-group

RUNTIME_TORCH_LIB_PATH := $(TORCH_ROOT)/torch/lib

tritoncc:
	time g++ tritoncc.cpp $(TRITONCC_CFLAGS) $(TRITONCC_LDFLAGS) -o out/tritoncc

test_sum: run_tblgen build_test_load_sum_cubin_and_run
	time g++ test_sum.cpp $(TRITONCC_CFLAGS) $(TRITONCC_LDFLAGS) -o out/test_sum
	out/test_sum
	LD_LIBRARY_PATH=$(RUNTIME_TORCH_LIB_PATH) out/test_load_sum_cubin_and_run /tmp/tritoncc.cubin

build_test_load_sum_cubin_and_run:
	g++ test_load_sum_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -Iinclude -o out/test_load_sum_cubin_and_run

test_add: run_tblgen build_test_load_add_cubin_and_run
	time g++ test_add.cpp $(TRITONCC_CFLAGS) $(TRITONCC_LDFLAGS) -o out/test_add
	out/test_add
	LD_LIBRARY_PATH=$(RUNTIME_TORCH_LIB_PATH) out/test_load_add_cubin_and_run /tmp/tritoncc.cubin

build_test_load_add_cubin_and_run:
	g++ test_load_add_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -Iinclude -o out/test_load_add_cubin_and_run
