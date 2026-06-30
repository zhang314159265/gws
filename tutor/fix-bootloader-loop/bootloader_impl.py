import dataclasses

@dataclasses.dataclass
class RunResult:
    acc: int = 0
    looped: bool = False
    loop_index: int | None = None

@dataclasses.dataclass
class RepairResult:
    acc: int = 0
    fixed_index: int | None = None

def parse_program(text_list):
    out = []
    for text in text_list:
        text = text.strip()
        if not text:
            continue
        parts = text.split()
        instr, operand = parts

        if instr not in ["plus", "next", "jump"]:
            raise ValueError(f"Invalid instruction {instr}")

        out.append((instr, int(operand)))
    return out

def run(prog):
    if len(prog) == 0:
        return RunResult()

    visited = set()

    idx = 0
    accum = 0
    while idx >= 0 and idx < len(prog):
        assert idx not in visited
        visited.add(idx)
        instr, operand = prog[idx]

        # compute next idx and accumulate
        if instr == "plus":
            accum += operand
            idx += 1
        elif instr == "next":
            idx += 1
        elif instr == "jump":
            idx += operand
        else:
            raise ValueError(f"Invalid instruction {instr}")

        if idx in visited:
            return RunResult(acc=accum, loop_index=idx, looped=True)

    return RunResult(acc=accum)

def detect_loop(prog):
    res = run(prog)
    return res.loop_index

def run_repaired(prog):
    if len(prog) == 0:
        return RunResult()

    visited = set()

    idx = 0
    accum = 0
    repair_idx = None
    while idx >= 0 and idx < len(prog):
        assert idx not in visited
        visited.add(idx)
        instr, operand = prog[idx]
        old_idx = idx

        # compute next idx and accumulate
        if instr == "plus":
            accum += operand
            idx += 1
        elif instr == "next":
            idx += 1
        elif instr == "jump":
            idx += operand
        else:
            raise ValueError(f"Invalid instruction {instr}")

        if idx in visited:
            if repair_idx is not None:
                break
            assert instr in ["next", "jump"]
            if instr == "next":
                idx = old_idx + operand
            else:
                idx = old_idx + 1
            repair_idx = old_idx

    return RepairResult(accum, repair_idx)

def solve(lines) -> int:
    return run_repaired(parse_program(lines)).acc
