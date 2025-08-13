nvcc_path := $(shell which nvcc)
cuda_bin_dir := $(shell dirname $(nvcc_path))
cuda_home := $(shell dirname $(cuda_bin_dir))

include_dirs := -I$(cuda_home)/include

arch := sm_90a  # h100

first: curun

runner: cubin
	g++ runner.cpp -o runner $(include_dirs) -lcuda
	./runner

cubin:
	nvcc -cubin add.cu -o add.cubin -arch=$(arch)

curun:
	g++ curun.cpp -shared -fPIC $(include_dirs) $(shell python -m pybind11 --includes) -o curun.so -lcuda
	python tutor.py

.PHONY: runner
