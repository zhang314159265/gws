import torch
import sys
import os
import contextlib
from torch import nn

def create_timm_model(model_name):
    from timm.models import create_model
    model = create_model(
        model_name,
        in_chans=3,
        scriptable=False,
        num_classes=None,
        drop_rate=0.0,
        drop_path_rate=None,
        drop_block_rate=None,
        # pretrained=True,
    )
    return model

def gen_transformer_inputs(vocab_size, bs, seq_length):
    def geninp():
        return torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64, requires_grad=False)

    input_dict = {
        "input_ids": geninp(),
        "labels": geninp()
    }
    return input_dict

def setup_tb_on_path():
    """
    Assumes torchbenchmark root directory is under pytorch root directory.
    """
    tb_root = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "torchbenchmark")
    assert os.path.exists(tb_root), f"TB root directory not exist {tb_root}"
    sys.path.append(tb_root)

@contextlib.contextmanager
def set_default_device_ctx(device):
    try:
        prior = torch.get_default_device()
        torch.set_default_device(device)
        yield
    finally:
        torch.set_default_device(prior)

setup_tb_on_path()

def create_model(model_name):
    out = globals()[f"create_{model_name}"]()
    if len(out) == 2:
        # the third item is perf_inputs
        return tuple([*out, None])
    else:
        assert len(out) == 3
        return out

##################### HF Models ##############

def create_AllenaiLongformerBase(bs=4):
    from transformers import AutoConfig, AutoModelForMaskedLM
    config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
    model = AutoModelForMaskedLM.from_config(config)

    vocab_size = model.config.vocab_size
    seq_length = 1024
    input_dict = gen_transformer_inputs(vocab_size, bs, seq_length)
    return model, input_dict


################### TIMM Models ################
def create_lcnet_050():
    model = create_timm_model("lcnet_050")
    inputs = torch.randn(256, 3, 224, 224)
    return model, inputs

def create_rexnet_100():
    model = create_timm_model("rexnet_100")
    inputs = torch.randn([128, 3, 224, 224])
    return model, inputs

########## TB Models #############
def create_pytorch_unet():
    from torchbenchmark.models.pytorch_unet.pytorch_unet.unet.unet_model import UNet
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    inputs = torch.randn(1, 3, 640, 959)
    return model, inputs

def create_speech_transformer():
    from torchbenchmark.models import speech_transformer
    with set_default_device_ctx("cpu"):
        benchmark = speech_transformer.Model(
            test="train",
            device="cuda",
            batch_size=None,
        )
        return benchmark.get_module()

def create_nvidia_deeprecommender():
    use_variant = True
    layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951]

    x = torch.randn(4, layer_sizes[0])

    class Model(nn.Module):
        def __init__(self, use_variant=True):
            super().__init__()
            mod_list = []
            for i in range(len(layer_sizes) - 1):
                mod_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if use_variant:
                    mod_list.append(nn.ReLU())
                else:
                    mod_list.append(nn.SELU())

                if i == 2:
                    mod_list.append(nn.Dropout(0.8))
            self.seq = nn.Sequential(*mod_list)

        def forward(self, x):
            return self.seq(x)

    m = Model(use_variant=use_variant)

    perf_inputs = torch.randn(256, layer_sizes[0])
    return m, x, perf_inputs

def create_timm_efficientdet():
    from torchbenchmark.models import timm_efficientdet
    with set_default_device_ctx("cpu"):
        benchmark = timm_efficientdet.Model(
            test="train",
            device="cuda",
            batch_size=1,
        )
    return benchmark.get_module()

def create_hf_Whisper():
    from transformers import WhisperConfig, AutoModelForAudioClassification
    config = WhisperConfig()
    model = AutoModelForAudioClassification.from_config(config)
    bs = 8
    feature_size = 80
    seq_length = 3000
    inputs = torch.randn(bs, feature_size, seq_length)
    return model, inputs

def create_hf_Longformer():
    """
    Same as create_AllenaiLongformerBase but uses a smaller batch size
    """
    return create_AllenaiLongformerBase(bs=2)
