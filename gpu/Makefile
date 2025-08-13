nvcc_path := $(shell which nvcc)
cuda_bin_dir := $(shell dirname $(nvcc_path))
cuda_home := $(shell dirname $(cuda_bin_dir))

include_dirs := -I$(cuda_home)/include

arch := sm_90a  # h100

runner: cubin
	g++ runner.cpp -o runner $(include_dirs) -lcuda
	./runner

cubin:
	nvcc -cubin add.cu -o add.cubin -arch=$(arch)

.PHONY: runner
