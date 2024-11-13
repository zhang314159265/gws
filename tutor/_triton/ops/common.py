import torch
from triton.testing import do_bench

def bench(torch_fn, triton_fn, args, tol=1e-4):
    expected = torch_fn(*args)
    actual = triton_fn(*args)
    torch.cuda.synchronize()

    if not torch.allclose(expected, actual, atol=tol, rtol=tol):
        print("Numerical check fail!")
        print(f"expected:\n{expected}\nactual:\n{actual}\n")
        return
    else:
        print("Pass the numerical check")

    # reclaim the memory to avoid confuse the measuring of peak memory
    # below.
    del expected, actual

    torch.cuda.reset_peak_memory_stats()
    torch_fn(*args)
    torch.cuda.synchronize()
    torch_peak_mem = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    triton_fn(*args)
    torch.cuda.synchronize()
    triton_peak_mem = torch.cuda.max_memory_allocated()
    print(f"Torch peak memory {torch_peak_mem}, triton peak memory {triton_peak_mem}")

    ms_torch = do_bench(lambda: torch_fn(*args)) 
    ms_triton = do_bench(lambda: triton_fn(*args))
    print(f"Torch latency {ms_torch:.3f}ms, triton latency {ms_triton:.3f}ms")


