"""Correctness + bandwidth tests for the distributed mode / median solutions.

Everything runs against a single-node ground truth (``collections.Counter`` for
mode, ``sorted`` for median). The cluster is the simulated, GPU-free threaded
harness in :mod:`harness`.
"""

import random

import pytest

from collections import Counter

from harness import Sim, distributed
import my_solution as solution

# The solutions in ref_solution are plain per-worker callables; wrap each with
# `distributed` here to get a `(sim) -> (answer, report)` entrypoint.
distributed_mode = distributed(solution.distributed_mode)
distributed_median = distributed(solution.distributed_median)
mode_naive = distributed(solution.mode_naive)

# ---------------------------------------------------------------------------
# Single-node ground truth (the "trivial part" you should not burn time on).
# ---------------------------------------------------------------------------


def single_node_mode(values):
    """Most common value; ties broken by smallest value for determinism."""
    counts = Counter(values)
    if not counts:
        return None
    best_count = max(counts.values())
    return min(v for v, c in counts.items() if c == best_count)


def single_node_median(values):
    """Exact median. Even length -> mean of the two middle elements."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2





def shard(values, num_workers, seed=0):
    """Split ``values`` into ``num_workers`` disjoint, shuffled shards."""
    rng = random.Random(seed)
    values = list(values)
    rng.shuffle(values)
    shards = [[] for _ in range(num_workers)]
    for i, v in enumerate(values):
        shards[i % num_workers].append(v)
    return shards


# ---------------------------------------------------------------------------
# MODE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [1, 2, 10])
@pytest.mark.parametrize("seed", range(8))
def test_mode_matches_ground_truth(num_workers, seed):
    rng = random.Random(1000 + seed)
    values = [rng.randint(0, 50) for _ in range(500)]
    expected = single_node_mode(values)

    answer, _ = distributed_mode(Sim(shard(values, num_workers, seed)))
    assert answer == expected


@pytest.mark.parametrize("num_workers", [1, 2, 10])
def test_mode_naive_matches_reshuffle(num_workers):
    rng = random.Random(7)
    values = [rng.randint(0, 30) for _ in range(400)]
    expected = single_node_mode(values)

    reshuffle, _ = distributed_mode(Sim(shard(values, num_workers)))
    naive, _ = mode_naive(Sim(shard(values, num_workers)))
    assert reshuffle == naive == expected


def test_mode_tie_break_is_smallest_value():
    # 3 and 7 both appear 4 times; the smallest value wins, deterministically.
    values = [3, 3, 3, 3, 7, 7, 7, 7, 1, 2]
    assert single_node_mode(values) == 3
    answer, _ = distributed_mode(Sim(shard(values, 10)))
    assert answer == 3


def test_mode_global_winner_is_not_any_local_winner():
    """The classic top-k false-negative case the reshuffle defends against.

    Value 5 is nobody's local mode, yet it is the global mode. A naive
    "send your local top-1" scheme would miss it; modulo reshuffle does not,
    because all of 5's counts land on the same owner.
    """
    shards = [
        [1, 1, 5],  # local mode is 1
        [2, 2, 5],  # local mode is 2
        [3, 3, 5],  # local mode is 3
        [4, 4, 5],  # local mode is 4
    ]
    flat = [v for s in shards for v in s]
    assert single_node_mode(flat) == 5  # 5 appears 4x, others 2x each

    answer, _ = distributed_mode(Sim(shards))
    assert answer == 5


def test_mode_requires_barrier_before_forwarding_best():
    """Regression test for the load-bearing barrier in distributed_mode.

    The barrier between phase 2 (drain dict slices) and phase 3 (forward best)
    guarantees no peer ships a (value, count) best to worker 0 while worker 0 is
    still receiving dict slices. We force that exact race deterministically:

      * Worker 0 owns key-slice 0 and must aggregate 10s from worker 0 and
        worker 1 (35 total) to beat worker 7's 25 sevens -> true mode is 10.
      * We delay ONLY the worker-1 -> worker-0 link. Worker 0 then blocks waiting
        for worker 1's slice while the fast peers (2..9) finish phase 2.

    With the barrier, those peers wait and the slice arrives first -> mode 10.
    Without it, a peer's best tuple laps the delayed slice and worker 0 consumes
    it as if it were a dict -> wrong answer or a crash (bounded by recv_timeout).
    """
    shards = [[10] * 5] + [[10] * 30] + [[] for _ in range(5)] + [[7] * 25] + [[], []]
    assert len(shards) == 10
    flat = [v for s in shards for v in s]
    assert single_node_mode(flat) == 10  # 10 x35 beats 7 x25

    sim = Sim(
        [list(s) for s in shards],
        link_latency={(1, 0): 0.2},  # only worker 1's slice to worker 0 is slow
        recv_timeout=2.0,  # fail fast instead of hanging if the order breaks
    )
    answer, _ = distributed_mode(sim)
    assert answer == 10


def test_mode_skewed_shards():
    # One node holds 90% of the data; correctness must not depend on balance.
    rng = random.Random(42)
    big = [rng.randint(0, 20) for _ in range(900)]
    rest = [[rng.randint(0, 20) for _ in range(11)] for _ in range(9)]
    shards = [big] + rest
    flat = [v for s in shards for v in s]

    answer, _ = distributed_mode(Sim(shards))
    assert answer == single_node_mode(flat)


# ---------------------------------------------------------------------------
# MEDIAN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [1, 2, 10])
@pytest.mark.parametrize("seed", range(8))
def test_median_odd_length(num_workers, seed):
    rng = random.Random(2000 + seed)
    values = [rng.randint(-100, 100) for _ in range(401)]  # odd
    expected = single_node_median(values)

    answer, _ = distributed_median(Sim(shard(values, num_workers, seed)))
    assert answer == expected


@pytest.mark.parametrize("num_workers", [1, 2, 10])
@pytest.mark.parametrize("seed", range(8))
def test_median_even_length(num_workers, seed):
    rng = random.Random(3000 + seed)
    values = [rng.randint(-100, 100) for _ in range(400)]  # even
    expected = single_node_median(values)

    answer, _ = distributed_median(Sim(shard(values, num_workers, seed)))
    assert answer == expected


def test_median_all_equal():
    values = [42] * 137
    answer, _ = distributed_median(Sim(shard(values, 10)))
    assert answer == 42


def test_median_skewed_shards():
    rng = random.Random(99)
    big = [rng.randint(0, 1000) for _ in range(900)]
    rest = [[rng.randint(0, 1000) for _ in range(11)] for _ in range(9)]
    shards = [big] + rest
    flat = [v for s in shards for v in s]

    answer, _ = distributed_median(Sim(shards))
    assert answer == single_node_median(flat)


def test_median_some_empty_shards():
    # Several workers hold nothing; the median is still exact.
    shards = [[5, 1, 9], [], [], [7, 3], [], [], [], [], [], [11]]
    flat = [v for s in shards for v in s]

    answer, _ = distributed_median(Sim(shards))
    assert answer == single_node_median(flat)


# ---------------------------------------------------------------------------
# BANDWIDTH MODEL
# ---------------------------------------------------------------------------


def peak_ingress(report):
    """Bytes received by the busiest single node -- the real bottleneck."""
    return max(m.recv_bytes for m in report.meters)


def test_reshuffle_balances_load_vs_naive_hotspot():
    """Naive concentrates *all* counters onto node 0 (a hotspot). Reshuffle
    spreads ingress across the key-space owners, so its busiest node receives
    far fewer bytes -- the win that matters when node 0 is the bottleneck."""
    rng = random.Random(5)
    values = [rng.randint(0, 100_000) for _ in range(5_000)]  # large alphabet
    shards = shard(values, 10)

    _, r_reshuffle = distributed_mode(Sim([list(s) for s in shards]))
    _, r_naive = mode_naive(Sim([list(s) for s in shards]))

    # Node 0 under naive eats everything; reshuffle's busiest node eats ~1/N.
    assert peak_ingress(r_reshuffle) < peak_ingress(r_naive)


def test_naive_wins_on_small_alphabet():
    """The note's claim: on a tiny alphabet the naive counter is cheap and the
    reshuffle's extra round-trip is pure overhead."""
    rng = random.Random(6)
    # values = [rng.randint(0, 10000000) for _ in range(5_000)]  # tiny alphabet
    values = [rng.randint(0, 3) for _ in range(5_000)]  # tiny alphabet
    shards = shard(values, 10)

    _, r_reshuffle = distributed_mode(Sim([list(s) for s in shards]))
    _, r_naive = mode_naive(Sim([list(s) for s in shards]))

    assert r_naive.total_net_bytes < r_reshuffle.total_net_bytes


def test_median_bisection_cost_is_logarithmic():
    """Bisection traffic scales with log2(value range), not dataset size."""
    rng = random.Random(8)
    values = [rng.randint(0, 1023) for _ in range(10_000)]  # range -> ~10 rounds
    shards = shard(values, 10)

    _, report = distributed_median(Sim(shards))

    # Read cost dwarfs network cost: that's the win.
    assert report.total_net_bytes < report.total_read_bytes
    # ~10 bisection rounds * 10 workers * (guess + reply), small constant.
    assert report.total_net_bytes < 5_000
