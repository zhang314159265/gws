from dataclasses import dataclass

@dataclass
class Sample:
    ts: float
    stack: list

@dataclass
class Event:
    kind: str
    ts: float
    name: str

def parse_events(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = tuple(line.split())
        if len(parts) != 2 or parts[0] not in ["enter", "exit"]:
            raise ValueError(f"Invalid line {line}")
        out.append(parts)
    return out

def reconstruct_stacks(events):
    stack = []
    out = []
    for ev in events:
        if ev[0] == "enter":
            stack.append(ev[1])
        elif ev[0] == "exit":
            if len(stack) == 0 or ev[1] != stack[-1]:
                raise ValueError(f"Unbalanced events")
            stack.pop()
        else:
            raise ValueError(f"Invalid event {ev[0]}")

        out.append(tuple(stack))
    return out

def compress(stacks):
    out = []
    if not stacks:
        return out

    item = stacks[0]
    count = 1
    for newitem in stacks[1:]:
        if newitem == item:
            count += 1
        else:
            out.append((item, count))
            item = newitem
            count = 1

    out.append((item, count))
    return out

def common_prefix_len(seq1, seq2):
    out = 0
    for lhs, rhs in zip(seq1, seq2):
        if lhs != rhs:
            break
        out += 1
    return out

def common_suffix_len(seq1, seq2):
    return common_prefix_len(reversed(seq1), reversed(seq2))

def suffix_compatible(seq1, seq2):
    return common_suffix_len(seq1, seq2) == min(len(seq1), len(seq2))

def dedup_suffix_samples(samples):
    out = []
    if not samples:
        return out

    item = samples[0]
    count = 1
    for newitem in samples[1:]:
        if suffix_compatible(item, newitem):
            item = item if len(item) >= len(newitem) else newitem
            count += 1
        else:
            out.append((item, count))
            item = newitem
            count = 1

    out.append((item, count))
    return out

def denoise(samples, max_elapsed):
    out = []
    if not samples:
        return out

    start_ts = end_ts = samples[0][0]
    stack = samples[0][1]
    count = 1

    for ts, newstack in samples[1:]:
        if ts - start_ts > max_elapsed or newstack != stack:
            out.append((stack, start_ts, end_ts, count))
            start_ts = end_ts = ts
            count = 1
            stack = newstack
        else:
            count += 1
            end_ts = ts

    out.append((stack, start_ts, end_ts, count))
    return out

def convert_samples_to_events(samples):
    out = []
    stack = []
    for sample in samples:
        ts = sample.ts
        newstack = sample.stack

        prefix_len = common_prefix_len(stack, newstack)
        for item in reversed(stack[prefix_len:]):
            out.append(Event(kind="end", ts=ts, name=item))

        for item in newstack[prefix_len:]:
            out.append(Event(kind="start", ts=ts, name=item))

        stack = newstack
    return out

def convert_samples_to_debounced_events(samples, n):
    # stack item [ts, name, cnt]
    stack = []
    out = []
    for sample in samples:
        ts, newstack = sample.ts, sample.stack

        # walk the prefix first
        prefix_len = 0
        for accum_item, newitem in zip(stack, newstack):
            if accum_item[1] != newitem:
                break
            prefix_len += 1
            accum_item[2] += 1
            if accum_item[2] == n:
                out.append(Event(kind="start", ts=accum_item[0], name=accum_item[1]))

        # leave
        for accum_item in reversed(stack[prefix_len:]):
            if accum_item[2] >= n:
                out.append(Event(kind="end", ts=ts, name=accum_item[1]))

        stack = stack[:prefix_len]
        # new item
        for item in newstack[prefix_len:]:
            accum_item = [ts, item, 1]
            stack.append(accum_item)
            if n == 1:
                out.append(Event(kind="start", ts=accum_item[0], name=accum_item[1]))
    return out
