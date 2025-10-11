import torch
from torch import nn
from torch import distributed
import os

if os.getenv("PATCH_NN_MODULE_INIT") == "1":
    orig_init = nn.Module.__init__
    
    def new_init(self, *args, **kwargs):
        # distributed._DistributedPdb().set_trace()
        print(f" -> init nn module {type(self)}")
        return orig_init(self, *args, **kwargs)
    
    nn.Module.__init__ = new_init

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
model_name = "openai/gpt-oss-20b"

if __name__ == "__main__":
    # XXX Fail when using the default CompilationConfig so far.
    cconfig = CompilationConfig(use_inductor=False)
    llm = LLM(model=model_name, compilation_config=cconfig)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=32)

    requests = [
        "Tell me a joke.",
        "How to estimate the value of pi in mathematics?",
        "How does quicksort works?",
    ]
    outputs = llm.generate(requests, sampling_params)

    assert len(outputs) == len(requests)
    for i, req_text in enumerate(requests):
        print(f"Response for request {i}: {outputs[i].outputs[0].text}") 
