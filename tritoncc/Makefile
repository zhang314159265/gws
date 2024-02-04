# XXX non-portable definitions BEGIN
LLVM_ROOT := $(HOME)/.triton/llvm/llvm-4017f04e-ubuntu-x64
# XXX non-portable definitions END

CFLAGS := -I$(LLVM_ROOT)/include
LDFLAGS := -L$(LLVM_ROOT)/lib
# mlir libs
LDFLAGS += -lMLIRIR -lMLIRDialect -lMLIRSupport
# llvm libs
LDFLAGS += -lLLVMCore -lLLVMSupport -lLLVMDemangle

all:
	g++ main.cpp $(CFLAGS) $(LDFLAGS) -o a.out
	./a.out
