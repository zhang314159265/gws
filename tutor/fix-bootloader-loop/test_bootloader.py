"""Tests for the bootloader instruction simulator (detect the loop, repair it).

Run with:  pytest test_bootloader.py

These tests run against the reference solution. To practice, write your own
``bootloader_impl.py`` exposing the same API and switch the import below to
``import bootloader_impl as bl``.
"""

import pytest

import bootloader_impl as bl


# ---------------------------------------------------------------------------
# Parsing: <op> <signed-int> lines, with validation of the alphabet.
# ---------------------------------------------------------------------------

class TestParseProgram:
    def test_parses_ops_and_signed_operands(self):
        prog = bl.parse_program(["plus 1", "next -2", "jump +3", "jump -4"])
        assert prog == [("plus", 1), ("next", -2), ("jump", 3), ("jump", -4)]

    def test_skips_blank_and_whitespace_lines(self):
        prog = bl.parse_program(["", "  plus 5 ", "   ", "next 0"])
        assert prog == [("plus", 5), ("next", 0)]

    def test_rejects_unknown_op(self):
        with pytest.raises(ValueError):
            bl.parse_program(["acc 1"])

    def test_rejects_non_integer_operand(self):
        with pytest.raises(ValueError):
            bl.parse_program(["plus x"])

    def test_rejects_missing_operand(self):
        with pytest.raises(ValueError):
            bl.parse_program(["plus"])


# ---------------------------------------------------------------------------
# Part 1 — run the program as written and detect the loop.
#
# `next` is a no-op move (operand ignored), `jump` adds the operand to the pc,
# `plus` accumulates. The run halts on the first line it would visit twice.
# ---------------------------------------------------------------------------

class TestRunAndDetectLoop:
    def test_terminates_by_running_off_the_end(self):
        # plus5 -> next (op ignored) -> plus3 -> off the end.
        prog = bl.parse_program(["plus 5", "next 0", "plus 3"])
        result = bl.run(prog)
        assert result == bl.RunResult(acc=8, looped=False, loop_index=None)
        assert bl.detect_loop(prog) is None

    def test_next_operand_is_ignored_for_control_flow(self):
        # The -99 on the `next` must NOT move the pc; it just advances by one.
        prog = bl.parse_program(["next -99", "plus 4"])
        assert bl.run(prog) == bl.RunResult(acc=4, looped=False, loop_index=None)

    def test_jump_off_the_front_also_terminates(self):
        # A jump that lands before line 0 leaves the listing -> halt, no loop.
        prog = bl.parse_program(["jump -1"])
        assert bl.run(prog) == bl.RunResult(acc=0, looped=False, loop_index=None)

    def test_detects_loop_and_reports_first_revisited_line(self):
        # 0:plus1 -> 1:plus2 -> 2:jump-2 -> back to 0 (already visited).
        prog = bl.parse_program(["plus 1", "plus 2", "jump -2", "plus 4", "jump 2", "plus 99"])
        result = bl.run(prog)
        assert result.looped is True
        assert result.loop_index == 0          # line 0 is the first visited twice
        assert result.acc == 3                 # accumulator just before the repeat
        assert bl.detect_loop(prog) == 0

    def test_empty_program(self):
        assert bl.run([]) == bl.RunResult(acc=0, looped=False, loop_index=None)


# ---------------------------------------------------------------------------
# Part 2 — repair the run. When a move would revisit a line, flip that one
# instruction: a revisiting `jump` becomes a `next`, a revisiting `next`
# becomes a `jump`. Return the TERMINAL accumulator.
# ---------------------------------------------------------------------------

class TestRunRepaired:
    def test_repairs_a_jump_that_loops_back(self):
        # The bug: line 2 should be `next -2` but was written `jump -2`, looping
        # to line 0. Repair treats the revisiting jump as a next -> advance to 3.
        #   0:plus1(acc1) 1:plus2(acc3) 2:[jump-2->next] 3:plus4(acc7) 4:jump2->off end
        prog = bl.parse_program(["plus 1", "plus 2", "jump -2", "plus 4", "jump 2", "plus 99"])
        result = bl.run_repaired(prog)
        assert result.acc == 7
        assert result.fixed_index == 2

    def test_repairs_a_next_that_loops(self):
        # The bug: line 3 should be `jump 3` but was written `next 3`. After a
        # back-jump lands on line 3, the natural `next` move (-> line 4) would
        # revisit, so it is treated as a jump using operand 3 -> off the end.
        #   0:plus1(acc1) 1:jump3->4 4:plus2(acc3) 5:jump-2->3 3:[next3->jump]->6 end
        prog = bl.parse_program(["plus 1", "jump 3", "plus 9", "next 3", "plus 2", "jump -2"])
        result = bl.run_repaired(prog)
        assert result.acc == 3
        assert result.fixed_index == 3

    def test_no_repair_needed_when_program_already_terminates(self):
        prog = bl.parse_program(["plus 5", "next 0", "plus 3"])
        assert bl.run_repaired(prog) == bl.RepairResult(acc=8, fixed_index=None)

    def test_terminal_accumulator_is_not_the_sum_of_all_plus_operands(self):
        # The reported clarification: return the value reached when the repaired
        # program halts, NOT the total of every `plus` in the file. Here the
        # `plus 99` on line 5 is never executed.
        prog = bl.parse_program(["plus 1", "plus 2", "jump -2", "plus 4", "jump 2", "plus 99"])
        sum_of_all_plus = 1 + 2 + 4 + 99
        assert bl.solve(["plus 1", "plus 2", "jump -2", "plus 4", "jump 2", "plus 99"]) == 7
        assert 7 != sum_of_all_plus

    def test_solve_end_to_end_from_raw_lines(self):
        assert bl.solve(["plus 1", "jump 3", "plus 9", "next 3", "plus 2", "jump -2"]) == 3

    def test_always_halts_on_a_pathological_program(self):
        # A program whose single inline flip does NOT fully resolve the loop
        # (the AoC day-8 example). run_repaired must still terminate, returning
        # the partial accumulator reached rather than hanging.
        prog = bl.parse_program([
            "next 0",   # 0
            "plus 1",   # 1
            "jump 4",   # 2
            "plus 3",   # 3
            "jump -3",  # 4  <- flipped to a `next` on the revisit of line 1
            "plus -99", # 5
            "plus 1",   # 6
            "jump -4",  # 7
            "plus 6",   # 8
        ])
        """
            * "next 0",   # 0
            * "plus 1",   # 1
            * "jump 4",   # 2
            * "plus 3",   # 3
            * "jump -3",  # 4  <- flipped to a `next` on the revisit of line 1
            * "plus -99", # 5 <===
            * "plus 1",   # 6
            * "jump -4",  # 7
            "plus 6",   # 8

        """
        result = bl.run_repaired(prog)
        assert isinstance(result.acc, int)     # it halted (did not hang)
        assert result.fixed_index == 4
        assert result.acc == -94               # documented partial-progress value
