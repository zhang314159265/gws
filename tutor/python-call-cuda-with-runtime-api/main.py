import torch
import mylib

x = torch.randn(1024, device="cuda")
y = torch.empty_like(x)
mylib.plus1(x.data_ptr(), y.data_ptr(), x.numel())
torch.testing.assert_close(x + 1, y)
print("pass")
