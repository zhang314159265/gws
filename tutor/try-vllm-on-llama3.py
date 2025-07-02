from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

if __name__ == "__main__":
    # XXX tFail when using the default CompilationConfig so far.
    cconfig = CompilationConfig(use_inductor=False)
    llm = LLM(model="/home/shunting/meta-llama/Meta-Llama-3.1-8B-Instruct", compilation_config=cconfig)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=32)

    outputs = llm.generate(["Tell me a joke."], sampling_params)
    print(outputs[0].outputs[0].text)
