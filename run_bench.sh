# HF
python benchmarks/dynamo/huggingface.py --backend inductor --amp --performance --only BertForMaskedLM --training

# TB
python benchmarks/dynamo/torchbench.py --backend inductor --amp --performance --only moco --training
