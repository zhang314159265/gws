import numpy as np
import torch
import torch.nn.functional as F
from triton.testing import do_bench

def check_and_bench_inductor(x, w, dy, ref, options={}, tol=1e-2):
    torch._dynamo.reset()
    opt_fwd = torch.compile(ref_fwd, options=options)
    y = opt_fwd(x, w)
    act = ref_bwd(x, w, dy, y)
    torch.testing.assert_close(ref, act, atol=tol, rtol=tol)
    ms = do_bench(lambda: ref_bwd(x, w, dy, y))
    return ms

def ref_fwd(x, w, eps=1e-5):
    return F.rms_norm(x, x.shape[-1:], w, eps)

def ref_bwd(x, w, dy, y):
    return torch.autograd.grad(y, [x, w], dy, retain_graph=True)

def create_inputs(shape):
    M, N = shape
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(N, dtype=torch.float, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    return x, w, dy

def plot_bench_result(bench_result):
    print(bench_result)
    import matplotlib.pyplot as plt

    first_result = next(iter(bench_result.values()))
    keys = list(first_result.keys())
    plt.figure(figsize=(12, 6))
    
    for shape, result in bench_result.items():
        y = np.array(list(result.values()))
        plt.plot(keys, y, "-o", label=str(shape))
    
   
    device_name = torch.cuda.get_device_name(0) 
    plt.title(f"RMSNorm Bwd ({device_name})", fontsize=16)
    plt.ylabel("Memory Throughput (TB/s)")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()

    # save
    path = "/tmp/perf.png"
    plt.savefig(path)
    print(f"Plot saved to {path}")

def handle_one_shape(shape):
    assert len(shape) == 2
    x, w, dy = create_inputs(shape)
    y = ref_fwd(x, w)
    ref = ref_bwd(x, w, dy, y)
    
    bench_result = {}
    bench_result["eager"] = do_bench(lambda: ref_bwd(x, w, dy, y))
    bench_result["inductor"] = check_and_bench_inductor(x, w, dy, ref, {
        "triton.mix_order_reduction": False,
    })
    bench_result["fused_split_reduction"] = check_and_bench_inductor(x, w, dy, ref, {
        "triton.mix_order_reduction": True,
        "split_reductions": True,
        # for shape [1152 * 500, 384], split size is chosen as 2043
    })

    for split_size_pow in range(1, 13):
        split_size = 2 ** split_size_pow
        bench_result[f"fused_ss_{split_size}"] = check_and_bench_inductor(x, w, dy, ref, {
            "triton.mix_order_reduction": True,
            "split_reductions": False,
            "triton.mix_order_reduction_split_size": split_size,
        })
  
    rsqrt_nbytes = shape[0] * torch.float.itemsize
    total_bytes = (
        x.nbytes * 2 + w.nbytes * 2 + dy.nbytes + rsqrt_nbytes
    )
    for key in list(bench_result):
        bench_result[key] = (total_bytes * 1e-12) / (bench_result[key] * 1e-3)

    return bench_result

torch.manual_seed(1337)

bench_result = {}
for shape in [
    (1152 * 500, 384),
    (1152 * 500, 512),
    (1152 * 1000, 384),
    (1152 * 1000, 512),

    # more shapes
] + [(32768, 2 ** n) for n in range(8, 12)]:
    bench_result[shape] = handle_one_shape(shape)
plot_bench_result(bench_result)

print("bye")
