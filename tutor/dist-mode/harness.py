"""Simulated distributed runtime for the "Distributed Mode / Median" problem.

There are no GPUs and no real machines here: the cluster is simulated with one
OS thread per worker and an in-memory mailbox (``queue.Queue``) per worker. This
lets us exercise ``send`` / ``recv`` / ``barrier`` code paths exactly as they
would run on a real cluster, while staying single-process and deterministic.

Bandwidth model (from the problem statement):
    * local read  : 10 bytes / sec
    * inter-node  :  1 byte  / sec   (charged to the *sender* on each ``send``)

We do not actually sleep. Instead every ``read`` and ``send`` is metered in
bytes, and the simulator reports the *simulated* wall-clock seconds implied by
the bandwidth model so strategies can be compared on cost, which is the whole
point of the exercise.
"""

from __future__ import annotations

import functools
import queue
import threading
from dataclasses import dataclass, field

READ_RATE = 10  # bytes / sec from local storage
NET_RATE = 1  # bytes / sec across an inter-node link


def nbytes(obj) -> int:
    """Estimate the on-wire size of a payload in bytes.

    A deliberately simple model: scalars are 8 bytes, strings one byte per
    character, and containers are the sum of their parts. Good enough to
    compare strategies; the absolute numbers are not meant to be exact.
    """
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, (int, float)):
        return 8
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return max(1, len(obj))
    if isinstance(obj, dict):
        return sum(nbytes(k) + nbytes(v) for k, v in obj.items()) or 1
    if isinstance(obj, (list, tuple, set)):
        return sum(nbytes(x) for x in obj) or 1
    if obj is None:
        return 1
    return 8


@dataclass
class Meter:
    """Per-worker bandwidth accounting (thread-safe via the sim's lock)."""

    read_bytes: int = 0
    sent_bytes: int = 0
    recv_bytes: int = 0
    sends: int = 0
    recvs: int = 0

    @property
    def read_seconds(self) -> float:
        return self.read_bytes / READ_RATE

    @property
    def net_seconds(self) -> float:
        # Only the sender pays for the link in this model.
        return self.sent_bytes / NET_RATE


@dataclass
class Report:
    """Aggregate cost report returned alongside worker results."""

    meters: list = field(default_factory=list)

    @property
    def total_read_bytes(self) -> int:
        return sum(m.read_bytes for m in self.meters)

    @property
    def total_net_bytes(self) -> int:
        return sum(m.sent_bytes for m in self.meters)

    @property
    def total_seconds(self) -> float:
        """Wall-clock under the bandwidth model.

        Reads happen in parallel across workers, so the read phase is bounded
        by the slowest reader. Network seconds are summed because the link is a
        shared 1 B/s pipe in the problem's model (a conservative upper bound).
        """
        read = max((m.read_seconds for m in self.meters), default=0.0)
        net = sum(m.net_seconds for m in self.meters)
        return read + net


class Context:
    """The handle each worker function receives.

    Exposes exactly the canonical primitives: ``worker_id``, ``WORKER_NUM``,
    ``send(target, data)``, ``recv()`` (blocking, returns from any sender) and
    ``barrier()`` -- plus ``read()`` for the local shard.
    """

    def __init__(self, sim: "Sim", worker_id: int):
        self._sim = sim
        self.worker_id = worker_id
        self.WORKER_NUM = sim.WORKER_NUM

    # -- local storage ----------------------------------------------------
    def read(self):
        """Read this worker's local shard. Metered at READ_RATE."""
        shard = self._sim.shards[self.worker_id]
        self._sim._account_read(self.worker_id, sum(nbytes(x) for x in shard))
        return shard

    # -- network ----------------------------------------------------------
    def send(self, target: int, data) -> None:
        self._sim._account_send(self.worker_id, nbytes(data))
        self._sim._deliver(self.worker_id, target, data)

    def recv(self):
        """Blocking receive; returns the payload from whichever sender is next."""
        timeout = self._sim.recv_timeout
        data = self._sim.queues[self.worker_id].get(timeout=timeout)
        self._sim._account_recv(self.worker_id, nbytes(data))
        return data

    def barrier(self) -> None:
        self._sim.barrier.wait()


class Sim:
    """Run a worker function across ``len(shards)`` simulated nodes."""

    def __init__(self, shards, link_latency=None, recv_timeout=None):
        """``link_latency`` injects per-link delivery delay (seconds) so tests
        can force a specific message interleaving. It may be a number (applied
        to every link), a ``{(src, dst): seconds}`` dict, or a callable
        ``(src, dst) -> seconds``. Delivery is delayed without blocking the
        sender, modelling in-flight latency. ``recv_timeout`` (seconds) makes a
        blocked ``recv`` raise instead of hanging forever -- useful so a
        deadlocked/misordered run fails a test promptly rather than wedging."""
        self.shards = list(shards)
        self.WORKER_NUM = len(self.shards)
        self.queues = [queue.Queue() for _ in range(self.WORKER_NUM)]
        self.barrier = threading.Barrier(self.WORKER_NUM)
        self.meters = [Meter() for _ in range(self.WORKER_NUM)]
        self._lock = threading.Lock()
        self.link_latency = link_latency
        self.recv_timeout = recv_timeout
        self._timers = []

    def _latency(self, src, dst):
        lat = self.link_latency
        if lat is None:
            return 0
        if callable(lat):
            return lat(src, dst) or 0
        if isinstance(lat, dict):
            return lat.get((src, dst), 0)
        return lat

    def _deliver(self, src, dst, data):
        latency = self._latency(src, dst)
        if latency and latency > 0:
            timer = threading.Timer(latency, self.queues[dst].put, args=(data,))
            with self._lock:
                self._timers.append(timer)
            timer.start()
        else:
            self.queues[dst].put(data)

    def _account_read(self, wid, b):
        with self._lock:
            self.meters[wid].read_bytes += b

    def _account_send(self, wid, b):
        with self._lock:
            self.meters[wid].sent_bytes += b
            self.meters[wid].sends += 1

    def _account_recv(self, wid, b):
        with self._lock:
            self.meters[wid].recv_bytes += b
            self.meters[wid].recvs += 1

    def run(self, worker_fn):
        """Execute ``worker_fn(ctx)`` on every worker thread.

        Returns ``(results, report)`` where ``results[i]`` is worker ``i``'s
        return value and ``report`` is the bandwidth :class:`Report`.
        """
        results = [None] * self.WORKER_NUM
        errors = [None] * self.WORKER_NUM

        def target(wid):
            ctx = Context(self, wid)
            try:
                results[wid] = worker_fn(ctx)
            except Exception as exc:  # surface worker crashes to the caller
                print(f"worker {wid} got exception {exc}")
                errors[wid] = exc
                # Trip the barrier so peers don't deadlock waiting on us.
                self.barrier.abort()

        threads = [
            threading.Thread(target=target, args=(wid,), name=f"worker-{wid}")
            for wid in range(self.WORKER_NUM)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Let any in-flight delayed deliveries finish so no timer outlives the run.
        for timer in list(self._timers):
            timer.join(timeout=1.0)

        for wid, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"worker {wid} failed: {err!r}") from err

        return results, Report(self.meters)


def distributed(worker):
    """Turn a per-worker callable into a ``(sim) -> (answer, report)`` entrypoint.

    Write the solution as a plain worker ``callable(ctx)`` that ``return``s its
    answer -- no ``sim.run`` call, no results/consensus plumbing. Decorate it and
    it becomes a function you invoke as ``entry(sim)``:

        @distributed
        def distributed_mode(ctx):
            ...
            return answer

    The decorator runs the callable across every worker, checks the workers
    agree, and hands back ``(answer, report)``.
    """

    @functools.wraps(worker)
    def entry(sim):
        results, report = sim.run(worker)
        answer = results[0]
        assert all(r == answer for r in results), f"workers disagree: {results}"
        return answer, report

    return entry
