"""Bootloader instruction simulator — detect the loop, then repair it.

The classic "find where the boot sequence loops" puzzle, with a twist on the
repair step. A program is a list of instructions, each an op plus a signed
integer operand:

  * ``plus <n>`` -- add <n> to a global accumulator, then advance one line.
  * ``next <n>`` -- advance one line; the operand is ignored for control flow
                    (a no-op move).
  * ``jump <n>`` -- jump to ``current_index + n``.

Execution halts when the program counter leaves the listing (runs off either
end) or when it is about to revisit a line it has already executed (an infinite
loop).

Exactly one line has had its ``jump`` / ``next`` op swapped, which is what
forces the loop. The task has two parts:

  1. Detect the loop -- report the first line that would be visited twice
     (:func:`detect_loop`) and the accumulator reached just before that repeat
     (:func:`run`).
  2. Repair the run (:func:`run_repaired`): when a move would land on an
     already-visited line, flip that one instruction for that step -- a ``jump``
     onto a visited line becomes a ``next`` (advance one), and a ``next`` onto a
     visited line becomes a ``jump`` (use its operand). Return the accumulator
     reached once the repaired program halts.

The return contract (a reported point of confusion in the interview) is the
*terminal* accumulator -- the value when the repaired program halts -- NOT the
sum of every ``plus`` operand in the file.
"""

from __future__ import annotations

import sys
from typing import Iterable, NamedTuple

OPS = ("plus", "next", "jump")


class RunResult(NamedTuple):
    """Outcome of a plain (un-repaired) run."""

    acc: int                 # accumulator at halt
    looped: bool             # True if halted on a revisit, False if it ran out of bounds
    loop_index: int | None   # first line that would be visited twice (None if terminated)


class RepairResult(NamedTuple):
    """Outcome of a repaired run."""

    acc: int                  # terminal accumulator when the repaired program halts
    fixed_index: int | None   # the line whose jump/next was flipped (None if no repair needed)


def parse_program(lines: Iterable[str]) -> list[tuple[str, int]]:
    """Parse ``<op> <signed-int>`` lines into ``(op, operand)`` tuples.

    Blank / whitespace-only lines are skipped. Anything else -- an unknown op,
    a non-integer operand, or the wrong number of tokens -- raises ValueError.
    Validating the alphabet of the program up front is part of the exercise.
    """
    program: list[tuple[str, int]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2 or parts[0] not in OPS:
            raise ValueError(f"malformed instruction: {line!r}")
        op, operand = parts
        try:
            program.append((op, int(operand)))
        except ValueError:
            raise ValueError(f"operand is not an integer: {line!r}") from None
    return program


def run(program: list[tuple[str, int]]) -> RunResult:
    """Execute the program as written, with no repair.

    Halts on the first revisit (``looped=True``, ``loop_index`` is the line that
    would be entered a second time) or when the program counter leaves the
    listing (``looped=False``, ``loop_index=None``).
    """
    acc = 0
    pc = 0
    visited: set[int] = set()
    n = len(program)
    while 0 <= pc < n:
        if pc in visited:
            return RunResult(acc, True, pc)
        visited.add(pc)
        op, operand = program[pc]
        if op == "plus":
            acc += operand
            pc += 1
        elif op == "next":
            pc += 1
        else:  # jump
            pc += operand
    return RunResult(acc, False, None)


def detect_loop(program: list[tuple[str, int]]) -> int | None:
    """The first line that would be visited twice, or None if the program
    terminates on its own."""
    return run(program).loop_index


def run_repaired(program: list[tuple[str, int]]) -> RepairResult:
    """Execute with the inline repair rule and return the terminal accumulator.

    Whenever the natural next move would land on an already-visited line, the
    single offending instruction is flipped for that step:

      * a ``jump`` onto a visited line is treated as a ``next`` (advance one);
      * a ``next`` onto a visited line is treated as a ``jump`` (use its operand).

    ``fixed_index`` records the first line so flipped (None if the program
    already terminates and needs no repair).

    Termination is guaranteed: every iteration either leaves the listing or
    marks a brand-new line visited, and the run stops the instant it would step
    onto a line already in the visited set -- so it cannot loop forever even on
    a malformed program (it just returns the partial accumulator reached).
    """
    acc = 0
    pc = 0
    visited: set[int] = set()
    fixed_index: int | None = None
    n = len(program)
    while 0 <= pc < n:
        if pc in visited:
            break
        visited.add(pc)
        op, operand = program[pc]
        if op == "plus":
            acc += operand
            pc += 1
        elif op == "next":
            if (pc + 1) in visited:          # would revisit -> act like a jump
                if fixed_index is None:
                    fixed_index = pc
                pc += operand
            else:
                pc += 1
        else:  # jump
            if (pc + operand) in visited:    # lands on a visited line -> act like a next
                if fixed_index is None:
                    fixed_index = pc
                pc += 1
            else:
                pc += operand
    return RepairResult(acc, fixed_index)


def solve(lines: Iterable[str]) -> int:
    """Parse raw instruction lines and return the repaired terminal accumulator."""
    return run_repaired(parse_program(lines)).acc


if __name__ == "__main__":  # pragma: no cover - tiny CLI for "given a file" inputs
    path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
    with open(path) as fh:
        prog = parse_program(fh)
    plain = run(prog)
    repaired = run_repaired(prog)
    if plain.looped:
        print(f"loop detected: first line revisited = {plain.loop_index} "
              f"(accumulator there = {plain.acc})")
        print(f"repaired bad instruction at line {repaired.fixed_index}")
    else:
        print("program terminates without looping")
    print(f"terminal accumulator = {repaired.acc}")
