nvcc_path := $(shell which nvcc)
cuda_bin_dir := $(shell dirname $(nvcc_path))
cuda_home := $(shell dirname $(cuda_bin_dir))

include_dirs := -I$(cuda_home)/include

arch := sm_90a  # h100

first: run_matmul

run_%:
	PYTHONPATH=. python kernel/$*.py

runner: cubin
	g++ runner.cpp -o runner $(include_dirs) -lcuda
	./runner

cubin_%:
	mkdir -p out/
	nvcc -cubin -I. kernel/$*.cu -o out/$*.cubin -arch=$(arch)

ptx_%:
	mkdir -p out/
	nvcc -ptx -I. kernel/$*.cu -o out/$*.ptx -arch=$(arch)

.PHONY: runner
