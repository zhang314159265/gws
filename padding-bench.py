import torch
import os
import copy

from model_factory import create_model
from torch._inductor import config as inductor_config
from torch._inductor.utils import do_bench
from torch._dynamo.testing import reduce_to_scalar_loss

MODEL_NAME = os.environ.get("MODEL_NAME", "DistillGPT2")

DO_ACC_TEST = os.environ.get("DO_ACC_TEST", "1") == "1"
DO_PERF_TEST = os.environ.get("DO_PERF_TEST", "1") == "1"
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "1") == "1"
WITH_STACK = os.environ.get("WITH_STACK") == "1"
USE_COMPILE = os.environ.get("USE_COMPILE", "1") == "1"
DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "cuda")

OPTIM_TYPE = os.environ.get("OPTIM_TYPE", "adam")

inductor_config.benchmark_kernel = True
inductor_config.triton.unique_kernel_names = True
inductor_config.triton.cudagraphs = USE_CUDA_GRAPHS

def get_optim(m):
    if OPTIM_TYPE == "adam":
        return torch.optim.Adam(
            m.parameters(), lr=0.01, capturable=True, foreach=True
        )
    elif OPTIM_TYPE == "sgd":
        optim = torch.optim.SGD(m.parameters(), lr=0.01, foreach=True)
        # follow pt2 benchmark scripts.
        optim.step = torch._dynamo.disable(optim.step)
        return optim
    else:
        assert False, f"Unrecognized optimizer type {OPTIM_TYPE}"

def check_close(ref, act, tol):
    if isinstance(ref, (tuple, list)) and isinstance(ref[0], torch.Tensor):
        ref = ref[0]
        act = act[0]
    if isinstance(ref, dict) and "loss" in ref:
        ref = ref["loss"]
        act = act["loss"]
    if type(ref).__name__ == "SequenceClassifierOutput":
        ref = ref.logits
        act = act.logits
    assert torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"

def common_numeric_check(f, tol, args, kwargs):
    ref = f(*args, **kwargs)
    opt_f = torch.compile(f)
    act = opt_f(*args, **kwargs)
    check_close(ref, act, tol)

def do_profiling(f_lhs, f_rhs, tag_lhs="With padding", tag_rhs="Without padding", args=(), kwargs={}):
   torch.cuda.synchronize()
   with torch.profiler.profile(with_stack=WITH_STACK) as p:
       niter = 3
       for _ in range(niter):
           with torch.profiler.record_function(tag_lhs):
               f_lhs(*args, **kwargs)

           with torch.profiler.record_function(tag_rhs):
               f_rhs(*args, **kwargs)
       torch.cuda.synchronize()

   profile_path = "/tmp/chrome.json"
   p.export_chrome_trace(profile_path)
   print(f"Chrome trace is written to {profile_path}")

def run_acc_and_perf_test(model, inputs, perf_inputs=None, tol=1e-3):
    if perf_inputs is None:
        perf_inputs = inputs

    def _process_inputs(x):
        """
        return args and kwargs
        """
        if isinstance(x, dict):
            return [], x

        if not isinstance(inputs, (tuple, list)):
            x = [x]

        return x, {}

    args, kwargs = _process_inputs(inputs)
    perf_args, perf_kwargs = _process_inputs(perf_inputs)

    if DO_ACC_TEST:
        model.eval()
        common_numeric_check(model, tol, args, kwargs)
    else:
        print("Accuracy test skipped")

    model.train()

    if DO_PERF_TEST:
        print("Do performance test")

        def get_f(m, optim):
            def f(*args, **kwargs):
                optim.zero_grad(True)
                with torch.cuda.amp.autocast():
                    pred = m(*args, **kwargs)
                    loss = reduce_to_scalar_loss(pred)
                loss.backward()
                optim.step()

            return f

        latency_with_padding = None
        print("Benchmark with padding")
        with inductor_config.patch(
            comprehensive_padding=True
        ):
            m_copy_with_padding = copy.deepcopy(model)
            optim_with_padding = get_optim(m_copy_with_padding)
            opt_f_with_padding = get_f(m_copy_with_padding, optim_with_padding)
            if USE_COMPILE:
                opt_f_with_padding = torch.compile(opt_f_with_padding)
            latency_with_padding = do_bench(lambda: opt_f_with_padding(*perf_args, **perf_kwargs))
        latency_without_padding = None
        print("bencmark without padding")
        with inductor_config.patch(comprehensive_padding=False):
            m_copy_without_padding = copy.deepcopy(model)
            optim_without_padding = get_optim(m_copy_without_padding)
            opt_f_without_padding = torch.compile(
                get_f(m_copy_without_padding, optim_without_padding)
            )
            latency_without_padding = do_bench(
                lambda: opt_f_without_padding(*perf_args, **perf_kwargs)
            )
        print(
            f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
        )

        # profiling
        do_profiling(opt_f_with_padding, opt_f_without_padding, args=perf_args, kwargs=perf_kwargs)


if __name__ == "__main__":
    model_name = MODEL_NAME
    print(f"model_name {model_name}")
    torch.set_default_device(DEFAULT_DEVICE)
    torch.set_float32_matmul_precision("high")
    model, inputs, perf_inputs = create_model(model_name)
    run_acc_and_perf_test(model, inputs, perf_inputs=perf_inputs, tol=1e-2)
    print("bye")
