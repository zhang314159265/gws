"""Tests for a best-fit MemoryAllocator.

API under test
--------------
MemoryAllocator(n):
    Manage `n` consecutive memory units, indexed 0 .. n-1, all free initially.

allocate(size) -> int:
    Reserve `size` consecutive free units using the *best-fit* strategy:
    among every maximal run of free units whose length is >= size, choose the
    one with the *smallest* length; break ties by the *lowest* start index.
    Reserve the first `size` units of that run and return its start index.
    Return -1 if no run is large enough (or size <= 0).

free(addr) -> int:
    Free the block that *starts* at index `addr` and return the number of
    units freed. Return 0 if `addr` is not the start of a currently
    allocated block. Freeing makes adjacent free units coalesce.
"""

import pytest
import os

if os.getenv("USE_REF") == "1":
    from ref import MemoryAllocator
    print(" == USE REFERENCE IMPLEMENTATION")
else:
    from myimpl import MemoryAllocator
    print(" == USE MY OWN IMPLEMENTATION")


# --------------------------------------------------------------------------
# Basic allocation
# --------------------------------------------------------------------------

def test_first_allocation_starts_at_zero():
    a = MemoryAllocator(10)
    assert a.allocate(3) == 0


def test_sequential_allocations_are_contiguous():
    a = MemoryAllocator(10)
    assert a.allocate(2) == 0
    assert a.allocate(3) == 2
    assert a.allocate(1) == 5


def test_allocate_exactly_full_capacity():
    a = MemoryAllocator(4)
    assert a.allocate(4) == 0


def test_allocate_more_than_capacity_returns_minus_one():
    a = MemoryAllocator(4)
    assert a.allocate(5) == -1


def test_allocate_when_full_returns_minus_one():
    a = MemoryAllocator(4)
    assert a.allocate(4) == 0
    assert a.allocate(1) == -1


def test_allocate_nonpositive_size_returns_minus_one():
    a = MemoryAllocator(4)
    assert a.allocate(0) == -1
    assert a.allocate(-2) == -1


# --------------------------------------------------------------------------
# Freeing
# --------------------------------------------------------------------------

def test_free_returns_number_of_units_freed():
    a = MemoryAllocator(10)
    a.allocate(3)              # occupies 0..2
    assert a.free(0) == 3


def test_free_makes_space_available_again():
    a = MemoryAllocator(4)
    assert a.allocate(4) == 0
    assert a.allocate(1) == -1  # full
    assert a.free(0) == 4
    assert a.allocate(1) == 0   # reusable after free


def test_free_invalid_address_returns_zero():
    a = MemoryAllocator(10)
    a.allocate(3)               # occupies 0..2
    # index 1 is allocated but is NOT the start of the block
    assert a.free(1) == 0
    # index 5 was never allocated
    assert a.free(5) == 0


def test_double_free_returns_zero_second_time():
    a = MemoryAllocator(10)
    a.allocate(3)
    assert a.free(0) == 3
    assert a.free(0) == 0


# --------------------------------------------------------------------------
# Best-fit selection (the heart of this allocator)
# --------------------------------------------------------------------------

def test_best_fit_picks_smallest_sufficient_hole():
    """A size-1 request must go to the size-1 hole, not the larger size-2 hole.

    First-fit would pick the earlier (size-2) hole at index 3; best-fit must
    pick the tighter (size-1) hole at index 9.
    """
    a = MemoryAllocator(12)
    a.allocate(3)               # 0..2
    b = a.allocate(2)           # 3..4
    a.allocate(4)               # 5..8
    d = a.allocate(1)           # 9
    a.allocate(2)               # 10..11  -> full
    assert a.free(b) == 2       # hole size 2 at index 3
    assert a.free(d) == 1       # hole size 1 at index 9
    assert a.allocate(1) == 9   # best fit chooses the tight size-1 hole


def test_best_fit_prefers_tight_hole_after_a_larger_earlier_hole():
    """The tight hole appears *after* a larger one, proving this is not first-fit."""
    a = MemoryAllocator(10)
    first = a.allocate(4)       # 0..3
    a.allocate(1)               # 4
    third = a.allocate(3)       # 5..7
    a.allocate(1)               # 8
    a.allocate(1)               # 9  -> full
    assert a.free(first) == 4   # hole size 4 at index 0
    assert a.free(third) == 3   # hole size 3 at index 5
    assert a.allocate(3) == 5   # best fit: size-3 hole, not the size-4 one at 0


def test_best_fit_ties_break_by_lowest_address():
    a = MemoryAllocator(9)
    a.allocate(2)               # 0..1
    a.allocate(2)               # 2..3
    a.allocate(2)               # 4..5
    a.allocate(2)               # 6..7
    a.allocate(1)               # 8  -> full
    assert a.free(0) == 2       # hole size 2 at index 0
    assert a.free(4) == 2       # hole size 2 at index 4
    # two equal-size holes -> pick the lowest address
    assert a.allocate(2) == 0


def test_best_fit_exact_fit_consumes_whole_hole():
    a = MemoryAllocator(10)
    a.allocate(3)               # 0..2
    b = a.allocate(2)           # 3..4
    a.allocate(5)               # 5..9  -> full
    assert a.free(b) == 2       # hole size 2 at index 3
    assert a.allocate(2) == 3   # exact fit
    assert a.allocate(1) == -1  # hole fully consumed, nothing left


# --------------------------------------------------------------------------
# Coalescing of adjacent free blocks
# --------------------------------------------------------------------------

def test_adjacent_frees_coalesce():
    a = MemoryAllocator(6)
    a.allocate(2)               # 0..1
    a.allocate(2)               # 2..3
    a.allocate(2)               # 4..5  -> full
    assert a.free(0) == 2
    assert a.free(2) == 2
    # 0..3 must coalesce into a single size-4 hole
    assert a.allocate(4) == 0


def test_coalesce_with_following_free_region():
    a = MemoryAllocator(10)
    a.allocate(3)               # 0..2
    b = a.allocate(3)           # 3..5   (6..9 stay free)
    assert a.free(b) == 3
    # freed 3..5 must join the trailing free 6..9 -> size-7 hole at index 3
    assert a.allocate(7) == 3


# --------------------------------------------------------------------------
# A short end-to-end scenario
# --------------------------------------------------------------------------

def test_scenario_mixed_operations():
    a = MemoryAllocator(8)
    assert a.allocate(2) == 0   # 0..1
    assert a.allocate(2) == 2   # 2..3
    assert a.allocate(2) == 4   # 4..5
    assert a.allocate(2) == 6   # 6..7  -> full
    assert a.allocate(1) == -1
    assert a.free(2) == 2       # hole size 2 at index 2
    assert a.free(6) == 2       # hole size 2 at index 6
    assert a.allocate(1) == 2   # best fit, lowest-address tie -> 2
    assert a.allocate(1) == 3   # remainder of that hole
    assert a.allocate(2) == 6   # the size-2 hole at index 6
    assert a.allocate(1) == -1  # full again
