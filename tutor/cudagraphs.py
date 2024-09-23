import torch
from torch._inductor import config
from triton.testing import do_bench
import os

config.triton.unique_kernel_names = True

torch.set_default_device("cuda")

@torch.compile
def f(x):
    return x.sin()

def run(x, steps=10):
    for i in range(steps):
        x = f(x)
    return x

def run_graph(graph, capture_input, capture_output, actual_input):
    capture_input.copy_(actual_input)
    graph.replay()
    out = torch.empty_like(capture_output)
    out.copy_(capture_output)
    return out

NITEM = int(os.environ.get("NITEM", "1048576"))
print(f"nitem {NITEM}")
warmup_input = torch.randn(NITEM)
# do some warm up runs
run(warmup_input, steps=5)

# capture
graph = torch.cuda.CUDAGraph()
capture_input = warmup_input  # reuse warmup_input
with torch.cuda.graph(graph):
    capture_output = run(capture_input)

test_input = torch.randn(NITEM)

# test for correctness
ref = run(test_input)
act = run_graph(graph, capture_input, capture_output, test_input)
assert torch.allclose(ref, act), f"ref\n{ref}\nact\n{act}"

# perf test
nocg_ms = do_bench(lambda: run(test_input))
cg_ms = do_bench(lambda: run_graph(graph, capture_input, capture_output, test_input))
print(f"latency without cuda graphs v.s. latency with cuda graphs: {nocg_ms:.3f}ms v.s. {cg_ms:.3f}ms. Cudagraph speedup ratio {nocg_ms / cg_ms:.3f}x")

# profiling
torch.cuda.synchronize()
with torch.profiler.profile() as p:
    run(test_input)
    run_graph(graph, capture_input, capture_output, test_input)
    torch.cuda.synchronize()

# trace for 2**20 items: https://gist.github.com/shunting314/5a0d21eb0226a384d1d3de81e56d4273 . CudaGraphs speedup 12x!
# trace for 10**7 items: https://gist.github.com/shunting314/4e4338597b4dfffd03b3d7f59268c028 . CudaGraphs speedup 1.322x!
# trace for 10**9 items: https://gist.github.com/shunting314/d5633992753a371443afdcec85299c9b . CudaGraphs speedup 0.822x due to extra copies for input and output.
profile_path = "/tmp/chrome.json"
p.export_chrome_trace(profile_path)
print(f"Chrome trace is written to {profile_path}")

print("bye")
