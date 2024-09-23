# - match numerical with pt2 huggingface.py for inference accuracy test.
# TODO support training
# TODO do perf test (with larger batch size)

import sys
sys.path.append(".")

class args:
    training = True # training or inference
    perf = False # perf or accuracy

    # args no need to change
    iterations = 2  # number of training steps to run. Only used for accuracy test
    amp = True

class bench_state:
    optimizer = None
    model_iter_fn = None

import torch
from transformers import BertForMaskedLM as model_cls
from model.bert_model import MyBertModel
from torch._dynamo.testing import reset_rng_state, collect_results
from torch._dynamo.utils import same
from triton.testing import do_bench
from torch._inductor import config as inductor_config
import contextlib

# The compiled inference perf will be >10x worse for the original hf model
# without the following line
if not args.training: torch.set_grad_enabled(False)

torch.backends.cuda.matmul.allow_tf32 = True

def download_model(model_cls, config):
    return model_cls(config)

autocast = torch.cuda.amp.autocast if args.amp else contextlib.nullcontext
model_name = "BertForMaskedLM"
config = model_cls.config_class()
reset_rng_state()
model = download_model(model_cls, config)
device = "cuda"
dtype = torch.float32
model = model.to(device, dtype=dtype)
batch_size = 16 if args.perf else 1
seq_length = 512
vocab_size = model.config.vocab_size
reset_rng_state()
input = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64, requires_grad=False)
input_dict = {
    "input_ids": input,
    "labels": labels,
}

def model_iter_fn_fwd(model, args, collect_outputs=True):
    with autocast():
        return model(**args)

def model_iter_fn_fwd_bwd(model, args, collect_outputs=True):
    with autocast():
        bench_state.optimizer.zero_grad(True)
        pred = model(**args)
    
        # pred[0] works for MaskedLMOutput but does not work for a reguard dict retuend
        # in my model
        loss = pred[next(iter(pred))]
        loss.backward()
        bench_state.optimizer.step()
        if collect_outputs:
            results = collect_results(model, pred, loss, args)
    
            # key may be different due to simplification of module tree
            results[2] = tuple(results[2].values())
            results[3] = tuple(results[3].values())
            return results

def init_optimizer(params):
    if args.training:
        bench_state.optimizer = torch.optim.Adam(params, lr=0.01, capturable=True, foreach=True)
    else:
        bench_state.optimizer = None

def run_n_iterations(mod, inputs):
    n = args.iterations
    for _ in range(n - 1):
        bench_state.model_iter_fn(mod, inputs, collect_outputs=False)
    return bench_state.model_iter_fn(mod, inputs, collect_outputs=True)

bench_state.model_iter_fn = model_iter_fn_fwd_bwd if args.training else model_iter_fn_fwd
if args.training:
    model.train()
else:
    model.eval()

reset_rng_state()
my_model = MyBertModel().to(device="cuda")
if args.training:
    my_model.train()
else:
    my_model.eval()

if args.perf:
    # No amp
    #   eager infernce, mymodel 38.86ms, hfmodel 42.60ms
    #   eager training, mymodel 129.97ms hfmodel 133.60ms
    #   compile inference, mymodel 30.41ms, hfmodel 30.48ms
    #   compile training, mymodel 114.74ms, hfmodel 115.51ms
    # With amp
    #   eager inference, mymodel 30.65ms, hfmodel 35.38ms
    #   compile inference, mymodel 13.11ms, hfmodel 13.22ms
    #     - XXX why my model is faster than pt2 bench script result (which takes 14ms)
    #   eager training, mymodel 99.12ms,
    #   compile training, mymodel 50.75ms
    #     - XXX why my model is faster than pt2 bench script result (which takes 52.21ms)

    picked_model = my_model
    # picked_model = model
    init_optimizer(picked_model.parameters())

    print(do_bench(lambda: bench_state.model_iter_fn(picked_model, input_dict)))
    inductor_config.triton.cudagraphs = True
    if not args.training:
        inductor_config.freezing = True
    opt_fn = torch.compile(bench_state.model_iter_fn)
    print(do_bench(lambda: opt_fn(picked_model, input_dict)))
else:
    init_optimizer(model.parameters())
    # This is important to make sure dropout behavior is deterministic for training.
    reset_rng_state()
    ref_output = run_n_iterations(model, input_dict)
    
    init_optimizer(my_model.parameters())
    reset_rng_state()
    act_output = run_n_iterations(my_model, input_dict)
    
    if not same(ref_output, act_output):
        breakpoint()
    
    assert same(ref_output, act_output)
    print("Pass")

print("bye")
