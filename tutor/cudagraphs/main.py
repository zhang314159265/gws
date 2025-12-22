import torch
import bind

torch.randn(1024, device="cuda");
print("After calling randn")

with torch.profiler.profile() as p:
    bind.run()

path = "/tmp/trace.json"
p.export_chrome_trace(path)
print(f"Trace is written to {path}")
print("Done")
