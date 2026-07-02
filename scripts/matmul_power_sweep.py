"""Sweep GPU power limit and measure matmul perf at each setting.

Sets the power limit via `sudo nvidia-smi -pl` (passwordless sudo required),
runs the matmul benchmark from ~/a.py in a fresh subprocess for each limit so
each run gets a clean CUDA context, then restores the original limit.

Example output on B200:

     Power (W)   Latency (ms)    TFLOP/s    TFLOP/s/W
------------------------------------------------
       200          1.140      120.5        0.603
       250          1.140      120.6        0.482
       300          0.253      543.4        1.811
       350          0.247      557.4        1.593
       400          0.201      685.5        1.714
       450          0.173      793.4        1.763
       500          0.153      900.8        1.802
       550          0.136     1008.7        1.834
       600          0.124     1105.8        1.843
       650          0.115     1197.1        1.842
       700          0.108     1277.5        1.825
       750          0.103     1338.8        1.785

The table is plotted here: https://github.com/user-attachments/assets/f77eb137-a50c-4bf3-930d-12278c29ab50
"""

import argparse
import re
import subprocess
import sys

# Benchmark run in a subprocess. Prints one line: "<ms>". N matches ~/a.py.
BENCH_SRC = """
import torch, triton
N = {n}
x = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
y = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
for _ in range(50):
    z = x @ y
ms = triton.testing.do_bench(lambda: x @ y)
print(ms)
"""


def get_power_limit(gpu):
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu),
         "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def set_power_limit(gpu, watts):
    subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(gpu), "-pl", str(watts)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_bench(gpu, n):
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.run(
        [sys.executable, "-c", BENCH_SRC.format(n=n)],
        capture_output=True, text=True,
        env={**__import__("os").environ, **env},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark failed:\n{proc.stderr}")
    ms = float(proc.stdout.strip().splitlines()[-1])
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--start", type=int, default=200)
    p.add_argument("--stop", type=int, default=750)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--n", type=int, default=4096, help="matmul dim (NxN)")
    args = p.parse_args()

    flop = 2 * args.n ** 3  # matmul FLOPs

    original = get_power_limit(args.gpu)
    print(f"GPU {args.gpu}: saving original power limit = {original:.0f} W\n")

    results = []
    try:
        for watts in range(args.start, args.stop + 1, args.step):
            set_power_limit(args.gpu, watts)
            ms = run_bench(args.gpu, args.n)
            tflops = flop / (ms * 1e-3) / 1e12
            results.append((watts, ms, tflops))
            print(f"  {watts:4d} W  ->  {ms:8.3f} ms   {tflops:8.1f} TFLOP/s")
    finally:
        set_power_limit(args.gpu, round(original))
        print(f"\nRestored power limit to {original:.0f} W")

    print(f"\n{'Power (W)':>10} {'Latency (ms)':>14} {'TFLOP/s':>10} {'TFLOP/s/W':>12}")
    print("-" * 48)
    for watts, ms, tflops in results:
        print(f"{watts:>10} {ms:>14.3f} {tflops:>10.1f} {tflops / watts:>12.3f}")


if __name__ == "__main__":
    main()
