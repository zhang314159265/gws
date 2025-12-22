NVCC_FLAGS := \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_100,code=sm_100

NVCC_PATH := $(shell which nvcc)
CUDA_HOME := $(abspath $(NVCC_PATH)/../..)

all:
	nvcc --compiler-options '-fPIC' $(NVCC_FLAGS) -c getting_started.cu -o getting_started.o
	g++ bind.cpp getting_started.o -shared -fPIC $(shell python -m pybind11 --includes) -o bind.so -L $(CUDA_HOME)/lib64 -lcudart
	python main.py
