import torch
from torch import nn
from torch import distributed
import contextlib
import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = os.getenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")

class script_args:
    model_name = "Qwen/Qwen3-0.6B"
    profile = False
    compile = False

if __name__ == "__main__":
    if script_args.profile:
        profile = torch.profiler.profile(with_stack=True)
    else:
        profile = contextlib.nullcontext()


    if script_args.compile:
        compilation_config = None
    else:
        from vllm.config import CompilationConfig, CUDAGraphMode, CompilationMode
        compilation_config = CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE, mode=CompilationMode.NONE)

    llm = LLM(model=script_args.model_name, compilation_config=compilation_config)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
    
    requests = [
        "Tell me a joke.",
        "How to estimate the value of pi in mathematics?",
        "How does quicksort works?",
    ]

    with profile:
        outputs = llm.generate(requests, sampling_params)

    assert len(outputs) == len(requests)
    for i, req_text in enumerate(requests):
        print(f"Response for request {i}: {outputs[i].outputs[0].text}") 

    if script_args.profile:
        path = "/tmp/profile.json"
        profile.export_chrome_trace(path)
        from create_perfetto_link import create_perflink
        create_perflink(path)
