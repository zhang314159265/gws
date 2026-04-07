"""
Outputs:
  N 1, latency_us: 11.618781089782715
  N 1024, latency_us: 11.93685531616211
  N 4096, latency_us: 11.578917503356934
"""
import time
import torch

for N in [1, 1024, 4096]:
    for _ in range(10):
        x = torch.tensor(N, device="cuda")
    
    start = time.time()
    ITER = 10000
    for _ in range(ITER):
        x = torch.tensor(N, device="cuda")
    end = time.time()
    latency_us = ((end - start) * 1000_000) / ITER
    
    print(f"N {N}, latency_us: {latency_us}")
