import torch._inductor.config as inductor_config
import os

if os.getenv("NOSPLIT") == "1":
    inductor_config.split_reductions = False
