import torch
from torch import nn
from torch._inductor import config

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

m = MyModel()

x = torch.ones(1024, device="cuda") * 3

ep = torch.export.export(m, (x,))

output_path = torch._inductor.aoti_compile_and_package(
    ep,
    (x,),
    package_path="/tmp/model.pt2")

loaded_m = torch._inductor.aoti_load_package("/tmp/model.pt2")

print(f"{ep=}")
print(f"{output_path=}")
print(f"{loaded_m(x * 2)}")
