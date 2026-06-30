# Distributed Mode / Median

Reference implementation + tests for the "10 worker nodes, find the global
**mode** and global **median**" problem (see `/tmp/note`).

The cluster is **simulated, no GPUs and no real machines**: each worker is an OS
thread with an in-memory mailbox. This exercises the real `send` / `recv` /
`barrier` code paths while staying single-process and deterministic.

## Files

| File | What |
|------|------|
| `harness.py` | Simulated runtime. One thread per worker, `queue.Queue` mailboxes, `threading.Barrier`, a bandwidth **Meter** (read = 10 B/s, inter-node = 1 B/s, charged to the sender), and the `@distributed` decorator. |
| `ref_solution.py` | Reference: `distributed_mode` (modulo-key reshuffle, Top-1 forward), `distributed_median` (iterative `count(x<=g)` bisection), and a `mode_naive` baseline. |
| `test_solution.py` | Correctness vs. single-node ground truth across worker counts, seeds, ties, skew, empty shards; plus bandwidth-model assertions. |

## Worker contract

A solution is **just a per-worker callable** `worker(ctx)` that returns its
answer — no `sim.run` call, no results/consensus plumbing. The caller wraps it
with `distributed(...)` to get a `(sim) -> (answer, report)` entrypoint (the
test does this; see `test_solution.py`):

```python
# ref_solution.py — the solution is a plain callable
def distributed_mode(ctx):
    ...
    return answer          # every worker returns the same answer

# caller side (e.g. test_solution.py)
from harness import Sim, distributed
answer, report = distributed(distributed_mode)(Sim(shards))
```

`ctx` exposes exactly the canonical primitives:

```python
ctx.worker_id          # this node's id
ctx.WORKER_NUM         # number of nodes (10 in the canonical problem)
ctx.read()             # this node's local shard (metered at 10 B/s)
ctx.send(target, data) # 1 B/s, charged to sender
ctx.recv()             # blocking, returns the next payload from any sender
ctx.barrier()          # all workers rendezvous
```

## Strategies

- **Mode — modulo reshuffle.** Each worker ships `(value, count)` to
  `value % WORKER_NUM`. After a barrier, each worker owns the *complete* global
  count for its slice of the key space, so forwarding its single best
  `(value, count)` to the aggregator is enough — Top-1 cannot false-negative
  here (unlike top-k of raw locals; see `test_mode_global_winner_is_not_any_local_winner`).
- **Median — bisection.** Aggregator broadcasts a guess `g`; each worker replies
  `count(x <= g)`; interval is halved. Exact for integers in `~log2(range)`
  rounds; network cost is independent of dataset size.

## Run

```bash
python -m pytest test_solution.py -q   # 85 tests
python ref_solution.py                  # demo + bandwidth report
```
