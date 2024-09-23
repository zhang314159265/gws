"""
In this test, the triton kernel has 3 inputs a, b, c and one output o.
Adding divisibility hint for a and o will result in bad layout;
without hint, triton will generate a good layout.
"""

import os

with open(os.path.join(os.path.dirname(__file__), "bad_layout.ptx")) as f:
    ptx_with_hint = f.read()

with open(os.path.join(os.path.dirname(__file__), "good_layout.ptx")) as f:
    ptx_without_hint = f.read()

import triton
import copy
import ptxrunner
import os
import torch
from torch._inductor.runtime.triton_heuristics import reduction, grid
from torch._inductor.runtime.hints import DeviceProperties
import triton.language as tl
from torch._inductor.ir import ReductionHint
from triton.compiler.compiler import AttrsDescriptor
from torch._dynamo.testing import rand_strided
from triton.testing import do_bench
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._dynamo.testing import reset_rng_state

reset_rng_state()

fp32_output = os.environ.get("FP32_OUTPUT") == "1"

def get_args():
    a = rand_strided((8192, 50272), (50272, 1), device='cuda:0', dtype=torch.float32)
    b = rand_strided((8192, 50272), (50272, 1), device='cuda:0', dtype=torch.float32)
    c = rand_strided((8192, 50272), (50272, 1), device='cuda:0', dtype=torch.float32)
    o = rand_strided((8192, 50272), (50272, 1), device='cuda:0', dtype=torch.float32 if fp32_output else torch.float16)
    return a, b, c, o

@triton.jit
def triton_kernel(a, b, c, o, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    """
    In either case, the kernel uses config:
        XBLOCK: 1, RBLOCK: 2048, num_warps: 8
    """
    xnumel = 8192
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(a + (r1 + (50272*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = _tmp13 + tmp0
        # remove this tl.where cause the fast version slow down from 4.587 to 4.829 ms, but does not affect the slow version.
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(a + (r1 + (50272*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(b + (r1 + (50272*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(c + (r1 + (50272*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp16 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(o + (r1 + (50272*x0)), tmp31, rmask)

def apply_inductor_hint(with_divisible_hints):
    divisible_by_16 = (
        (0, 1, 2, 3,)
        if with_divisible_hints
        else
        (1, 2,)
    )
    return reduction(
        size_hints=[8192, 65536],
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            'signature': {
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32' if fp32_output else '*fp16',
            },
            'device': DeviceProperties(tyep="cuda", index=0),
            'constants': {},
            'configs': [AttrsDescriptor(divisible_by_16=divisible_by_16, equal_to_1=())]
        },
    )(triton_kernel)

kernel_with_divisible_hints = apply_inductor_hint(True)
kernel_without_disivible_hints = apply_inductor_hint(False)

def call_kernel(kernel, args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        kernel.run(*args, grid=grid(8192), stream=stream0)

args = get_args()

# use kernel_with_divisible_hints as reference
call_kernel(kernel_with_divisible_hints, args)
ref = copy.deepcopy(args[-1])

use_ptx = os.environ.get("USE_PTX", "1") == "1"

def assert_close(ref, act):
    tol = {"atol": 4 * 1e-3, "rtol": 4 * 1e-3}
    correct = torch.allclose(ref, act, **tol)

    if not correct:
        mask = ~torch.isclose(ref, act, **tol)
        print(f"ref[mask]:\n{ref[mask]}")
        print(f"act[mask]:\n{act[mask]}")

    assert correct, f"Ref:\n{ref}\nAct:\n{act}"

if use_ptx:
    print("Use ptx for the kernel with hints")
    cu_func_with_hint = ptxrunner.load_cu_func_from_ptx_code(ptx_with_hint)
    ms_with_hints = do_bench(lambda: ptxrunner.launch(cu_func_with_hint, args=args, gridDim=8192, blockDim=32*8, shared=32))

    assert_close(ref, args[-1])
else:
    ms_with_hints = do_bench(lambda: call_kernel(kernel_with_divisible_hints, args))

    assert_close(ref, args[-1])

if use_ptx:
    print("Use ptx for the kernel without hints")
    cu_func_without_hint = ptxrunner.load_cu_func_from_ptx_code(ptx_without_hint)
    # nshared = 2056 # the number reported by triton is an over estimate
    nshared = 2048
    ms_without_hints = do_bench(lambda: ptxrunner.launch(cu_func_without_hint, args=args, gridDim=8192, blockDim=32*8, shared=nshared))
    assert_close(ref, args[-1])
else:
    ms_without_hints = do_bench(lambda: call_kernel(kernel_without_disivible_hints, args))
    assert_close(ref, args[-1])

# FP32 output:
# ms_with_hints 5.101 v.s. ms_without_hints 5.168

# FP16 output:
# ms_with_hints 6.049 v.s. ms_without_hints 4.669
# ms_without_hints:
# - in this ersion %p1 is uninitialized: https://gist.github.com/shunting314/c04661a1463c9cf984c9d7a3c7ae9a77
#   But it passes accuracy check and got 3.855ms perf...
# - fixing the above by removing %p1 since it's always true, the perf go back to 4.59ms..
#   https://gist.github.com/shunting314/ec39de678b2c127be2a84a630082dca4
# - removing an noop, perf goes to 4.793ms 
#   https://gist.github.com/shunting314/7816f43aa1e4fe699b0368e057782ab8
# - use vectorized load for aptr. Perf goes to 4.665 ms
#   https://gist.github.com/shunting314/f3b66576b145f797c061fe3bb5d66040
print(f"ms_with_hints {ms_with_hints:.3f} v.s. ms_without_hints {ms_without_hints:.3f}")

print("bye")
