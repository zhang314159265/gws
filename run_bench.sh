# HF training
python benchmarks/dynamo/huggingface.py --backend inductor --amp --performance --only BertForMaskedLM --training

python benchmarks/dynamo/huggingface.py --backend inductor --amp --performance --only BertForMaskedLM --training --cold-start-latency

# HF inference
python benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 --backend inductor --freezing --only BertForMaskedLM

# TB training
python benchmarks/dynamo/torchbench.py --backend inductor --amp --performance --only moco --training
