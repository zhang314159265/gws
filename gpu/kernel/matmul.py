import torch
from util import bench, check_and_bench, ceildiv
import torch._inductor.config as inductor_config
import functools
from matmul_triton import matmul_triton
import curun

inductor_config.benchmark_kernel = False
inductor_config.max_autotune = True
# inductor_config.max_autotune_gemm_backends = "triton"

M, N, K = 4096, 4096, 4096

A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

# A.fill_(1.0); B.fill_(2.0);

def baseline(A, B):
    return A @ B

compiled_baseline = torch.compile(baseline)

def run_kernel(kernel, A, B, accum_in_regs=False):
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    BSIZ = 64
    num_blocks = (ceildiv(M, BSIZ), ceildiv(N, BSIZ))

    num_threads = 1024
    shared = BSIZ * BSIZ * 2 * torch.bfloat16.itemsize
    if not accum_in_regs:
        shared += BSIZ * BSIZ * torch.float32.itemsize
    kernel[num_blocks, num_threads, shared](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BSIZ,
    )
    return C

h_matmul_cuda_kernel_cuda_core = curun.open("kernel/matmul_cuda_core.cu").sym("matmul_cuda_kernel_cuda_core")
def matmul_cuda_kernel_cuda_core(A, B):
    return run_kernel(h_matmul_cuda_kernel_cuda_core, A, B)

h_matmul_cuda_kernel_cuda_core_accum_in_regs = curun.open("kernel/matmul_cuda_core_accum_in_regs.cu").sym("matmul_cuda_kernel_cuda_core_accum_in_regs")
def matmul_cuda_kernel_cuda_core_accum_in_regs(A, B):
    return run_kernel(h_matmul_cuda_kernel_cuda_core_accum_in_regs, A, B, accum_in_regs=True) 

ref = baseline(A, B)
total_bytes = (M * K + K * N + M * N) * torch.bfloat16.itemsize
total_flops = M * N * K * 2
check_and_bench = functools.partial(
    check_and_bench,
    args=(A, B),
    ref=ref,
    total_bytes=total_bytes,
    total_flops=total_flops,
    tol=1e-2,
)
y = matmul_triton(A, B)

bench(lambda: baseline(A, B), "baseline", total_bytes=total_bytes, total_flops=total_flops)
check_and_bench(compiled_baseline, label="compiled_baseline")
check_and_bench(matmul_triton)
check_and_bench(matmul_cuda_kernel_cuda_core)
# make sure the next call does not reuse the result of the previous call
v1, v2 = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
check_and_bench(matmul_cuda_kernel_cuda_core_accum_in_regs)

print("bye")
