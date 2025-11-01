import torch
from torch import nn
from torch import distributed
import os

from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen3-0.6B"

if __name__ == "__main__":
    llm = LLM(model=model_name)
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
