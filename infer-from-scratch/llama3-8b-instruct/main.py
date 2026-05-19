from config import config
from model import get_model
from tokenizer import Tokenizer
from args import parse_args
from generate import generate, interactive
import torch

torch.set_default_dtype(torch.bfloat16)
tokenizer = Tokenizer(config.tokenizer_file)
model = get_model(config)

with torch.device("cuda"):
    if parse_args().interactive:
        interactive(tokenizer, model, config)
    else:
        generate(config.prompt, tokenizer, model, config)
