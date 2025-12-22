NVCC_FLAGS := \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_100,code=sm_100

all:
	nvcc --compiler-options '-fPIC' $(NVCC_FLAGS) -c plus1.cu -o plus1.o
	g++ mylib.cpp plus1.o -shared -fPIC $(shell python -m pybind11 --includes) -o mylib.so
	python main.py
	echo bye
