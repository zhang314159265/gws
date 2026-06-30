"""Reference implementations for distributed global mode and global median.

Each solution is a plain per-worker callable ``worker(ctx)`` that ``return``s
its answer -- no ``sim.run`` call, no results/consensus plumbing. The caller
wraps it with :func:`harness.distributed` to run it across all workers and get
back ``(answer, report)`` (see ``test_solution.py``). Worker 0 is the aggregator.

Strategies (see /tmp/note for the discussion):

  * Mode -- modulo-key reshuffle. Each worker hashes every local value ``v`` to
    ``target = v % WORKER_NUM`` and ships ``(v, count)`` there. After a barrier,
    a worker owns the *complete* global count for its slice of the key space, so
    it only forwards its single best ``(value, count)`` to the aggregator. Top-1
    per worker is sufficient *because the per-key counts are now complete* -- the
    "top-k of locals" false-negative cannot happen here.

  * Median -- iterative bisection. The aggregator broadcasts a guess ``g``; each
    worker replies ``count(x <= g)``; the aggregator halves the search interval.
    Exact for integers in ``~log2(range)`` rounds.
"""

from __future__ import annotations

from collections import Counter

# ---------------------------------------------------------------------------
# Distributed MODE -- modulo-key reshuffle, Top-1 forward.
# ---------------------------------------------------------------------------


def distributed_mode(ctx):
    N = ctx.WORKER_NUM
    local = Counter(ctx.read())

    # Phase 1: partition local counts by key % N and ship each slice.
    buckets = [dict() for _ in range(N)]
    for value, count in local.items():
        buckets[value % N][value] = count
    for target in range(N):
        ctx.send(target, buckets[target])

    # Phase 2: I own one slice of the key space. Sum the complete counts.
    # recv() blocks until each slice arrives; no other traffic exists yet.
    owned = Counter()
    for _ in range(N):
        owned.update(ctx.recv())

    # Top-1 is enough: counts for my keys are globally complete.
    if owned:
        best_count = max(owned.values())
        best_value = min(v for v, c in owned.items() if c == best_count)
        my_best = (best_value, best_count)
    else:
        my_best = None

    # LOAD-BEARING barrier: every worker must finish draining its N dict slices
    # (phase 2) before *anyone* sends a (value, count) best to the aggregator.
    # Without it a fast peer's best can reach worker 0 while worker 0 is still
    # mid-phase-2, so recv() hands worker 0 a tuple where it expects a dict slice
    # and the counts get mangled. Guarded by
    # test_mode_requires_barrier_before_forwarding_best.
    ctx.barrier()

    # Phase 3: everyone forwards their single best to the aggregator.
    ctx.send(0, my_best)

    result = None
    if ctx.worker_id == 0:
        best = None  # (count, -value) maximised -> highest count, smallest value
        for _ in range(N):
            cand = ctx.recv()
            if cand is None:
                continue
            value, count = cand
            key = (count, -value)
            if best is None or key > best[0]:
                best = (key, value)
        result = None if best is None else best[1]
        for target in range(N):
            ctx.send(target, ("RESULT", result))

    payload = ctx.recv()
    assert payload[0] == "RESULT"
    return payload[1]


def mode_naive(ctx):
    """Baseline: every worker ships its full local counter to node 0."""
    N = ctx.WORKER_NUM
    local = Counter(ctx.read())
    ctx.send(0, dict(local))

    result = None
    if ctx.worker_id == 0:
        total = Counter()
        for _ in range(N):
            total.update(ctx.recv())
        if total:
            best_count = max(total.values())
            result = min(v for v, c in total.items() if c == best_count)
        for target in range(N):
            ctx.send(target, ("RESULT", result))

    payload = ctx.recv()
    assert payload[0] == "RESULT"
    return payload[1]


# ---------------------------------------------------------------------------
# Distributed MEDIAN -- iterative value bisection (exact for integers).
# ---------------------------------------------------------------------------


def distributed_median(ctx):
    """Exact integer median via count-less-than-or-equal bisection.

    Protocol (worker 0 coordinates, FIFO request/reply keeps it ordered):
      setup  : participants send (local_n, local_min, local_max) to 0.
      rounds : 0 broadcasts ('LE', g); each replies count(x <= g); 0 reduces.
      finish : 0 broadcasts ('DONE', median); everyone returns it.
    """
    N = ctx.WORKER_NUM
    data = ctx.read()
    local_n = len(data)

    # --- participant branch --------------------------------------------------
    if ctx.worker_id != 0:
        ctx.send(0, (local_n, min(data) if data else None, max(data) if data else None))
        while True:
            cmd, payload = ctx.recv()
            if cmd == "LE":
                ctx.send(0, sum(1 for x in data if x <= payload))
            elif cmd == "DONE":
                return payload

    # --- coordinator branch (worker 0) ---------------------------------------
    total_n = local_n
    lo = min(data) if data else None
    hi = max(data) if data else None
    for _ in range(N - 1):
        n_i, min_i, max_i = ctx.recv()
        total_n += n_i
        if min_i is not None:
            lo = min_i if lo is None else min(lo, min_i)
            hi = max_i if hi is None else max(hi, max_i)

    def global_count_le(g):
        # Broadcast guess; reduce count(x <= g) across all workers.
        for target in range(1, N):
            ctx.send(target, ("LE", g))
        total = sum(1 for x in data if x <= g)  # my own share
        for _ in range(N - 1):
            total += ctx.recv()
        return total

    def kth_value(k):  # 0-indexed k-th smallest value
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2  # floors toward -inf for ints in Python
            if global_count_le(mid) >= k + 1:
                b = mid
            else:
                a = mid + 1
        return a

    if total_n == 0:
        median = None
    elif total_n % 2 == 1:
        median = kth_value(total_n // 2)
    else:
        median = (kth_value(total_n // 2 - 1) + kth_value(total_n // 2)) / 2

    for target in range(1, N):
        ctx.send(target, ("DONE", median))
    return median


# ---------------------------------------------------------------------------
# Demo: run both passes and print the bandwidth-cost model.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    from harness import Sim, distributed

    rng = random.Random(0)
    data = [rng.randint(0, 1000) for _ in range(10_000)]
    shards = [data[i::10] for i in range(10)]  # 10 disjoint shards

    mode, mr = distributed(distributed_mode)(Sim([list(s) for s in shards]))
    median, dr = distributed(distributed_median)(Sim([list(s) for s in shards]))

    print(f"global mode   = {mode}")
    print(f"global median = {median}")
    print()
    print("mode  : net_bytes={:>7}  read_bytes={:>7}  ~{:.1f}s".format(
        mr.total_net_bytes, mr.total_read_bytes, mr.total_seconds))
    print("median: net_bytes={:>7}  read_bytes={:>7}  ~{:.1f}s".format(
        dr.total_net_bytes, dr.total_read_bytes, dr.total_seconds))
