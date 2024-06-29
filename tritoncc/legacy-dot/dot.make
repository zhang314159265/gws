# XXX my generted dot ptx does not contains ldmatrix/mma instr. It contains
# a lot of fma instr. Looks like the matmul is implemented naively rather than
# leveraging specialized HW instrs.
test_dot:
	g++ test_dot.cpp $(ALLFLAGS) -o a.out
	LD_LIBRARY_PATH=$(OVERRIDE_LIB_PATH) ./a.out
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run /tmp/tritoncc.cubin

test_load_dot_cubin_and_run:
	g++ test_load_dot_cubin_and_run.cpp $(TORCH_CFLAGS) $(TORCH_LDFLAGS) $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -I./include -o out/test_load_dot_cubin_and_run
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run skip-checkin/dot_ref.cubin fn_0d1d2d

run_dot_ptx:
	ptxas -arch=sm_90 dot.ptx -o /tmp/tritoncc.cubin
	LD_LIBRARY_PATH=$(OVERRIDE_TORCH_LIB_PATH) out/test_load_dot_cubin_and_run /tmp/tritoncc.cubin dot_fn
