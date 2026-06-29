"""Call-stack reconstruction from execution event logs, plus follow-ups.

The canonical interview problem: given a stream of ``enter <func>`` / ``exit
<func>`` events, reconstruct the active call stack at every transition and
run-length compress consecutive identical traces. This module implements the
base problem and the follow-ups discussed in the spec:

  * base            -- parse_events, reconstruct_stacks, compress
  * follow-up 1     -- compress with prefix vs postfix (leaf-first) comparison
  * follow-up 2     -- dedup of last-m-visible (suffix-only) samples
  * bonus           -- time-bounded denoise of consecutive identical samples
  * inverse variant -- samples -> start/end events (base + debounced)

Stack representation: a tuple of frame names, outermost (root) first, innermost
(leaf) last -- e.g. ``("main", "f", "g")`` means main called f called g.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Base problem
# ---------------------------------------------------------------------------

def parse_events(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``enter <name>`` / ``exit <name>`` lines into (kind, name) tuples.

    Blank / whitespace-only lines are skipped. Any other shape raises ValueError
    -- the interviewer specifically watches whether you validate the alphabet of
    the event stream before trusting it.
    """
    events: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2 or parts[0] not in ("enter", "exit"):
            raise ValueError(f"malformed event line: {line!r}")
        events.append((parts[0], parts[1]))
    return events


def reconstruct_stacks(events: Iterable[tuple[str, str]]) -> list[tuple[str, ...]]:
    """Return the active stack (root-first tuple) after each event.

    Each ``exit`` must match the name on top of the stack (the most recent
    unmatched ``enter``); otherwise the stream is malformed.
    """
    stack: list[str] = []
    snapshots: list[tuple[str, ...]] = []
    for kind, name in events:
        if kind == "enter":
            stack.append(name)
        elif kind == "exit":
            if not stack:
                raise ValueError(f"exit {name!r} with empty stack")
            if stack[-1] != name:
                raise ValueError(
                    f"exit {name!r} does not match top of stack {stack[-1]!r}"
                )
            stack.pop()
        else:  # pragma: no cover - parse_events already guards this
            raise ValueError(f"unknown event kind: {kind!r}")
        snapshots.append(tuple(stack))
    return snapshots


def compress(stacks: Iterable[tuple[str, ...]]) -> list[tuple[tuple[str, ...], int]]:
    """Run-length compress consecutive identical stacks.

    Returns a list of ``(stack, count)`` entries where count is how many
    consecutive samples carried that exact stack.
    """
    runs: list[tuple[tuple[str, ...], int]] = []
    for stack in stacks:
        if runs and runs[-1][0] == stack:
            prev_stack, count = runs[-1]
            runs[-1] = (prev_stack, count + 1)
        else:
            runs.append((stack, 1))
    return runs


# ---------------------------------------------------------------------------
# Follow-up 1: prefix vs postfix comparison primitives
# ---------------------------------------------------------------------------

def common_prefix_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Length of the shared root-first prefix of two stacks."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def common_suffix_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Length of the shared leaf-first suffix of two stacks."""
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def suffix_compatible(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Could ``a`` and ``b`` be the same logical stack, leaf-aligned?

    True when the shorter one is exactly a leaf-first suffix of the longer one:
    the overlapping (leaf) region matches frame-for-frame, and any extra frames
    only extend further toward the root -- where the hidden prefix lives.
    """
    return common_suffix_len(a, b) == min(len(a), len(b))


# ---------------------------------------------------------------------------
# Follow-up 2: dedup of last-m-visible (suffix-only) samples
# ---------------------------------------------------------------------------

def dedup_suffix_samples(
    samples: Iterable[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], int]]:
    """Run-length dedup of suffix-only samples that could be the same stack.

    Only the last ``m`` frames of each stack are visible (``m`` unknown, may
    vary). Two consecutive samples are merged when they are leaf-aligned
    compatible (:func:`suffix_compatible`). The representative of a run is the
    longest sample seen in it -- it carries the most frames toward the root and
    therefore the least ambiguity.

    Recursion caveat: with the prefix hidden we cannot distinguish ``("f",)``
    from a deeper ``("f", "f")``; they are suffix-compatible and so get merged.
    """
    runs: list[tuple[tuple[str, ...], int]] = []
    for sample in samples:
        if runs and suffix_compatible(runs[-1][0], sample):
            rep, count = runs[-1]
            longer = sample if len(sample) > len(rep) else rep
            runs[-1] = (longer, count + 1)
        else:
            runs.append((sample, 1))
    return runs


# ---------------------------------------------------------------------------
# Bonus: time-bounded denoise
# ---------------------------------------------------------------------------

def denoise(
    samples: Iterable[tuple[float, tuple[str, ...]]],
    max_elapsed: float,
) -> list[tuple[tuple[str, ...], float, float, int]]:
    """Collapse consecutive identical stacks, bounded by elapsed time.

    ``samples`` is a sequence of ``(timestamp, stack)`` pairs sorted ascending.
    Consecutive identical stacks merge into one run, but a run is closed once it
    would span more than ``max_elapsed`` from its start -- so the dedup is
    bounded by elapsed time, not just by sample count.

    Returns ``(stack, start_ts, end_ts, count)`` runs.
    """
    runs: list[tuple[tuple[str, ...], float, float, int]] = []
    for ts, stack in samples:
        if runs:
            rep, start_ts, _end_ts, count = runs[-1]
            if rep == stack and ts - start_ts <= max_elapsed:
                runs[-1] = (rep, start_ts, ts, count + 1)
                continue
        runs.append((stack, ts, ts, 1))
    return runs


# ---------------------------------------------------------------------------
# Inverse variant: stack samples -> start/end event list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """A stack sample at a point in time. ``stack`` is outermost -> innermost."""

    ts: float
    stack: list[str]


@dataclass(frozen=True)
class Event:
    """A reconstructed lifecycle event for one frame."""

    kind: str  # "start" or "end"
    ts: float
    name: str


def convert_samples_to_events(samples: list[Sample]) -> list[Event]:
    """Diff consecutive stack samples into start/end events.

    For each adjacent pair, frames are matched by position along the longest
    common prefix (so the same name at different depths is a distinct frame --
    recursion-safe). Frames present in the older sample but missing in the newer
    are closed (innermost -> outermost); newly-appearing frames are opened
    (outermost -> innermost). Frames still live in the final sample are NOT
    closed.
    """
    events: list[Event] = []
    prev: list[str] = []
    for sample in samples:
        cur = list(sample.stack)
        lcp = common_prefix_len(tuple(prev), tuple(cur))
        # Close frames that dropped off, innermost (deepest) first.
        for name in reversed(prev[lcp:]):
            events.append(Event("end", sample.ts, name))
        # Open newly-appearing frames, outermost first.
        for name in cur[lcp:]:
            events.append(Event("start", sample.ts, name))
        prev = cur
    return events


def convert_samples_to_debounced_events(samples: list[Sample], n: int) -> list[Event]:
    """Like :func:`convert_samples_to_events`, but a frame must persist at the
    same stack position (same parents, same depth) for ``n`` consecutive
    samples before its ``start`` is emitted.

    Any break in the prefix resets the streak: separated runs do NOT add up, and
    the whole prior stack must close before a new one opens. ``start_ts`` is the
    first sample of the streak (a consistent choice across all frames). A
    confirmed frame that later disappears emits an ``end``; a frame that never
    reaches ``n`` is treated as noise and emits nothing.
    """
    events: list[Event] = []
    # tracked[i] = [name, start_ts, streak, emitted] for the frame at depth i.
    tracked: list[list] = []

    for sample in samples:
        cur = list(sample.stack)
        names = [t[0] for t in tracked]
        lcp = common_prefix_len(tuple(names), tuple(cur))

        # Frames beyond the common prefix dropped off: close the confirmed ones,
        # innermost first, then forget them.
        for entry in reversed(tracked[lcp:]):
            name, _start_ts, _streak, emitted = entry
            if emitted:
                events.append(Event("end", sample.ts, name))
        del tracked[lcp:]

        # Surviving frames continue their streak; confirm at the n-th sample.
        for entry in tracked[:lcp]:
            entry[2] += 1
            if not entry[3] and entry[2] >= n:
                entry[3] = True
                events.append(Event("start", entry[1], entry[0]))

        # New frames start a fresh streak of length 1 (outermost first). With
        # n == 1 they confirm immediately.
        for name in cur[lcp:]:
            entry = [name, sample.ts, 1, False]
            if entry[2] >= n:
                entry[3] = True
                events.append(Event("start", entry[1], entry[0]))
            tracked.append(entry)

    return events
