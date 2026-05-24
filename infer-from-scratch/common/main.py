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
    args = parse_args()
    if args.interactive:
        interactive(args, tokenizer, model, config)
    else:
        generate(args, config.prompt, tokenizer, model, config)
