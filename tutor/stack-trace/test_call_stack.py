"""Tests for call_stack — stack reconstruction from event logs and its variants.

Run with:  pytest test_call_stack.py
"""

import pytest

import call_stack_impl as cs


# ---------------------------------------------------------------------------
# Base problem: parse events, reconstruct the active stack at each step.
# ---------------------------------------------------------------------------

class TestParseEvents:
    def test_parses_enter_and_exit_lines(self):
        events = cs.parse_events(["enter main", "enter f", "exit f", "exit main"])
        assert events == [
            ("enter", "main"),
            ("enter", "f"),
            ("exit", "f"),
            ("exit", "main"),
        ]

    def test_ignores_blank_lines_and_extra_whitespace(self):
        events = cs.parse_events(["  enter  main ", "", "   ", "exit main"])
        assert events == [("enter", "main"), ("exit", "main")]

    def test_rejects_unknown_event_kind(self):
        with pytest.raises(ValueError):
            cs.parse_events(["jump main"])


class TestReconstructStacks:
    def test_single_call(self):
        stacks = cs.reconstruct_stacks([("enter", "main"), ("exit", "main")])
        assert stacks == [("main",), ()]

    def test_nested_calls(self):
        events = [
            ("enter", "main"),
            ("enter", "f"),
            ("enter", "g"),
            ("exit", "g"),
            ("exit", "f"),
            ("exit", "main"),
        ]
        stacks = cs.reconstruct_stacks(events)
        assert stacks == [
            ("main",),
            ("main", "f"),
            ("main", "f", "g"),
            ("main", "f"),
            ("main",),
            (),
        ]

    def test_exit_must_match_most_recent_enter(self):
        # exiting "main" while "f" is still on top is malformed.
        with pytest.raises(ValueError):
            cs.reconstruct_stacks([("enter", "main"), ("enter", "f"), ("exit", "main")])

    def test_exit_with_empty_stack_is_error(self):
        with pytest.raises(ValueError):
            cs.reconstruct_stacks([("exit", "main")])

    def test_recursion_same_name_twice(self):
        events = [("enter", "f"), ("enter", "f"), ("exit", "f"), ("exit", "f")]
        stacks = cs.reconstruct_stacks(events)
        assert stacks == [("f",), ("f", "f"), ("f",), ()]


# ---------------------------------------------------------------------------
# Base problem: run-length compress consecutive identical stacks.
# ---------------------------------------------------------------------------

class TestCompress:
    def test_collapses_adjacent_identical_stacks(self):
        stacks = [("a",), ("a",), ("a", "b"), ("a", "b"), ("a", "b"), ("a",)]
        runs = cs.compress(stacks)
        assert runs == [
            (("a",), 2),
            (("a", "b"), 3),
            (("a",), 1),
        ]

    def test_empty_input(self):
        assert cs.compress([]) == []

    def test_no_adjacent_duplicates(self):
        stacks = [("a",), ("b",), ("a",)]
        assert cs.compress(stacks) == [(("a",), 1), (("b",), 1), (("a",), 1)]


# ---------------------------------------------------------------------------
# Follow-up 1: prefix vs postfix comparison primitives.
#
# Prefix comparison (root-first) is the natural representation. The postfix
# re-orientation compares leaf-first, which is what makes the suffix-only case
# (follow-up 2) tractable: when the prefix is hidden we can only align frames
# from the leaf inward.
# ---------------------------------------------------------------------------

class TestComparisonPrimitives:
    def test_common_prefix_len(self):
        assert cs.common_prefix_len(("a", "b", "c"), ("a", "b", "d")) == 2
        assert cs.common_prefix_len(("a", "b"), ("x", "b")) == 0
        assert cs.common_prefix_len((), ("a",)) == 0

    def test_common_suffix_len(self):
        # leaf-first: ("a","b","c") and ("x","b","c") share the "b","c" tail.
        assert cs.common_suffix_len(("a", "b", "c"), ("x", "b", "c")) == 2
        assert cs.common_suffix_len(("a", "b"), ("a", "x")) == 0
        assert cs.common_suffix_len(("a",), ()) == 0

    def test_suffix_compatible_when_shorter_is_leaf_aligned_suffix(self):
        # The shorter sample is fully explained as a suffix of the longer one,
        # so once the hidden prefix is filled in they could be the same stack.
        assert cs.suffix_compatible(("b", "c"), ("a", "b", "c")) is True
        assert cs.suffix_compatible(("a", "b", "c"), ("b", "c")) is True

    def test_suffix_incompatible_when_leaf_differs(self):
        assert cs.suffix_compatible(("a", "b", "c"), ("a", "b", "d")) is False

    def test_suffix_compatible_empty(self):
        assert cs.suffix_compatible((), ("a",)) is True


# ---------------------------------------------------------------------------
# Follow-up 2: last-m visible frames. The sampler only captures the last m
# frames of each stack (prefix hidden, m unknown). Dedup consecutive samples
# that could correspond to the same logical stack.
# ---------------------------------------------------------------------------

class TestDedupSuffixSamples:
    def test_merges_compatible_suffixes_keeping_longest_rep(self):
        # ("b","c") could be the tail of ("a","b","c"): same logical stack.
        samples = [("b", "c"), ("a", "b", "c"), ("a", "b", "c")]
        runs = cs.dedup_suffix_samples(samples)
        assert runs == [(("a", "b", "c"), 3)]

    def test_splits_when_leaf_changes(self):
        samples = [("a", "b"), ("a", "c")]
        runs = cs.dedup_suffix_samples(samples)
        assert runs == [(("a", "b"), 1), (("a", "c"), 1)]

    def test_empty(self):
        assert cs.dedup_suffix_samples([]) == []

    def test_recursion_is_inherently_ambiguous(self):
        # With only the suffix visible we cannot tell a deeper recursive frame
        # from the same shallow stack: ("f",) is a valid suffix of ("f","f").
        # We document this by merging them (could-be-same).
        samples = [("f",), ("f", "f")]
        runs = cs.dedup_suffix_samples(samples)
        assert runs == [(("f", "f"), 2)]


# ---------------------------------------------------------------------------
# Bonus: time-bounded denoise of consecutive identical samples.
# ---------------------------------------------------------------------------

class TestDenoise:
    def test_collapses_identical_within_time_bound(self):
        samples = [
            (0.0, ("a",)),
            (1.0, ("a",)),
            (2.0, ("a",)),
            (3.0, ("b",)),
        ]
        runs = cs.denoise(samples, max_elapsed=10.0)
        assert runs == [
            (("a",), 0.0, 2.0, 3),
            (("b",), 3.0, 3.0, 1),
        ]

    def test_splits_run_when_elapsed_exceeds_bound(self):
        # The dedup is bounded by elapsed time, not just sample count: the third
        # "a" is too far from the run start, so it opens a new run.
        samples = [
            (0.0, ("a",)),
            (1.0, ("a",)),
            (5.0, ("a",)),
        ]
        runs = cs.denoise(samples, max_elapsed=3.0)
        assert runs == [
            (("a",), 0.0, 1.0, 2),
            (("a",), 5.0, 5.0, 1),
        ]

    def test_empty(self):
        assert cs.denoise([], max_elapsed=1.0) == []


# ---------------------------------------------------------------------------
# Inverse variant: stack samples -> start/end event list.
# ---------------------------------------------------------------------------

def ev(kind, ts, name):
    return cs.Event(kind=kind, ts=ts, name=name)


class TestConvertSamplesToEvents:
    def test_single_sample_opens_but_does_not_close(self):
        samples = [cs.Sample(ts=1.0, stack=["main", "f"])]
        events = cs.convert_samples_to_events(samples)
        # frames live in the final sample are never closed.
        assert events == [ev("start", 1.0, "main"), ev("start", 1.0, "f")]

    def test_diff_emits_end_then_start_along_suffix(self):
        samples = [
            cs.Sample(ts=0.0, stack=["main", "f"]),
            cs.Sample(ts=1.0, stack=["main", "g"]),
        ]
        events = cs.convert_samples_to_events(samples)
        assert events == [
            ev("start", 0.0, "main"),
            ev("start", 0.0, "f"),
            ev("end", 1.0, "f"),
            ev("start", 1.0, "g"),
        ]

    def test_end_events_emitted_innermost_first(self):
        samples = [
            cs.Sample(ts=0.0, stack=["a", "b", "c"]),
            cs.Sample(ts=1.0, stack=["a"]),
        ]
        events = cs.convert_samples_to_events(samples)
        assert events == [
            ev("start", 0.0, "a"),
            ev("start", 0.0, "b"),
            ev("start", 0.0, "c"),
            ev("end", 1.0, "c"),
            ev("end", 1.0, "b"),
        ]

    def test_recursion_safe_same_name_distinct_depth(self):
        samples = [
            cs.Sample(ts=0.0, stack=["f"]),
            cs.Sample(ts=1.0, stack=["f", "f"]),
        ]
        events = cs.convert_samples_to_events(samples)
        # The shared depth-0 "f" persists; the depth-1 "f" is a NEW frame.
        assert events == [
            ev("start", 0.0, "f"),
            ev("start", 1.0, "f"),
        ]


class TestConvertSamplesToDebouncedEvents:
    def test_emits_only_after_n_consecutive_samples(self):
        samples = [
            cs.Sample(ts=1.0, stack=["a", "b"]),
            cs.Sample(ts=2.0, stack=["a", "b", "c"]),
            cs.Sample(ts=3.0, stack=["a", "b", "c"]),
        ]
        events = cs.convert_samples_to_debounced_events(samples, n=2)
        # a,b confirm at the 2nd sample; c confirms at the 3rd. start_ts is the
        # first sample of each frame's streak. Nothing closes (all live at end).
        assert events == [
            ev("start", 1.0, "a"),
            ev("start", 1.0, "b"),
            ev("start", 2.0, "c"),
        ]

    def test_prefix_change_resets_streak_and_emits_nothing(self):
        # The reset corner: a,b never reach N=2 because the prefix flips.
        samples = [
            cs.Sample(ts=1.0, stack=["a", "b"]),
            cs.Sample(ts=2.0, stack=["c", "b", "a"]),
        ]
        events = cs.convert_samples_to_debounced_events(samples, n=2)
        assert events == []

    def test_confirmed_frame_that_disappears_is_closed(self):
        samples = [
            cs.Sample(ts=1.0, stack=["a"]),
            cs.Sample(ts=2.0, stack=["a"]),
            cs.Sample(ts=3.0, stack=[]),
        ]
        events = cs.convert_samples_to_debounced_events(samples, n=2)
        assert events == [ev("start", 1.0, "a"), ev("end", 3.0, "a")]

    def test_n_equals_one_emits_immediately(self):
        samples = [cs.Sample(ts=1.0, stack=["a"])]
        events = cs.convert_samples_to_debounced_events(samples, n=1)
        assert events == [ev("start", 1.0, "a")]
