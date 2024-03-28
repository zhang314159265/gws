"""
Here is a nice description of the trace file format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit#heading=h.yr4qxyxotyw 
"""

import json
import re
import io
from dataclasses import dataclass, field
import contextlib

chrome_json_path = "/tmp/chrome.json"

class config:
    global_min_ts = None
    list_min_ts = None

    duration_diff_threshold = 5
    compare_window = 10
    print_threshold = [None, None] # don't print event in the diff view if duration is not in the range

@contextlib.contextmanager
def set_list_min_ts(event_list):
    min_ts = event_list.compute_min_ts()
    try:
        prior, config.list_min_ts = config.list_min_ts, min_ts
        yield
    finally:
        config.list_min_ts = prior 

@dataclass
class Event:
    ph: str # phase type
    name: str
    pid: int
    tid: int
    ts: int # time stamp in us granularity
    s: str = None
    cat: str = None
    dur: int = None # the unit is microseconds confirmed by the captured trace
    args: dict = field(default_factory=dict) # extra args
    id: int = None
    bp: str = None

    @property
    def rel_ts(self):
        return self.ts - config.global_min_ts

    def __str__(self):
        if config.list_min_ts:
            list_rel_ts = f"{self.ts - config.list_min_ts:6} "
        else:
            list_rel_ts = ""
        return f"{self.rel_ts:6} {list_rel_ts}{self.dur:6} {self.name[:128]}"

class EventList:
    def __init__(self, event_list):
        self.event_list = event_list

    def __len__(self):
        return len(self.event_list)

    def append(self, obj):
        self.event_list.append(obj)

    def filter_by_name_re(self, name_re):
        return EventList([
            e for e in self.event_list if re.search(name_re, e.name)
        ])

    def only_gpu_event(self):
        return EventList([
            e for e in self.event_list if e.cat == "kernel"
        ])

    def find_by_rel_ts(self, rel_ts):
        """
        Find an event by rel_ts
        """
        out = []
        for idx, event in enumerate(self.event_list):
            if event.rel_ts == rel_ts:
                out.append((idx, event))
   
        assert len(out) == 1, f"{out}"
        return out[0]

    def filter_by_rel_ts_range(self, start_rel_ts, end_rel_ts):
        """
        Find events between [start_rel_ts, end_rel_ts].
        """
        start_idx, _ = self.find_by_rel_ts(start_rel_ts)
        end_idx, _ = self.find_by_rel_ts(end_rel_ts)
        return EventList(self.event_list[start_idx: end_idx + 1])

    def compute_min_ts(self):
        return min(e.ts for e in self.event_list if e is not None)

    def __str__(self):
        buf = io.StringIO()
        for idx, event in enumerate(self.event_list):
            buf.write(f"{idx:4} {str(event)}\n")
        return buf.getvalue()

    def __getitem__(self, idx):
        return self.event_list[idx]

def diff_event_list(event_list_lhs, event_list_rhs, alternative_print=True):
    lhs_print_list = []
    rhs_print_list = []
    def print_event(idx, lhs):
        event_list = event_list_lhs if lhs else event_list_rhs
        event = event_list[idx]

        thres = config.print_threshold
        if thres[0] is not None and event.dur < thres[0]:
            return
        if thres[1] is not None and event.dur > thres[1]:
            return
        mark = '-' if lhs else '+'
        with set_list_min_ts(event_list):
            msg = f"{mark} {idx:4} {str(event)}"
            if alternative_print:
                print(msg)
            else:
                if lhs:
                    lhs_print_list.append(msg + "\n")
                else:
                    rhs_print_list.append(msg + "\n")

    lhs_idx, rhs_idx = 0, 0

    while lhs_idx < len(event_list_lhs) and rhs_idx < len(event_list_rhs):
        # does alternatively checking left or right helpful?
        stay_event = event_list_lhs[lhs_idx]
        niter = min(config.compare_window, len(event_list_rhs) - rhs_idx)

        found_match = False
        for i in range(niter):
            move_idx = rhs_idx + i
            move_event = event_list_rhs[move_idx]
            if stay_event.name == move_event.name and abs(stay_event.dur - move_event.dur) <= config.duration_diff_threshold:
                # match
                # skip rhs event list between [rhs_idx, move_idx)
                for j in range(rhs_idx, move_idx):
                    print_event(j, False)
                found_match = True
                lhs_idx += 1
                rhs_idx = move_idx + 1
                break

        if not found_match:
            # no match skip the lhs one
            print_event(lhs_idx, True)
            lhs_idx += 1
            # no change to rhs

    while lhs_idx < len(event_list_lhs):
        print_event(lhs_idx, True)
        lhs_idx += 1

    while rhs_idx < len(event_list_rhs):
        print_event(rhs_idx, False)
        rhs_idx += 1

    if not alternative_print:
        print("".join(lhs_print_list))
        print("".join(rhs_print_list))

with open(chrome_json_path) as f:
    chrome_json_obj = json.load(f)

event_list = EventList([])
for event_json_obj in chrome_json_obj["traceEvents"]:
    # print(event_json_obj)
    try:
        event_obj = Event(**event_json_obj)
    except Exception as e:
        print(event_json_obj)
        raise

    event_list.append(event_obj)

config.global_min_ts = event_list.compute_min_ts()
print(len(event_list))
event_list = event_list.only_gpu_event()
print(len(event_list))
# event_list = event_list.filter_by_name_re(r"triton_red_fused__native_batch_norm_legit_functional_2")
# print(event_list)

event_list_lhs = event_list.filter_by_rel_ts_range(426356, 459317)
event_list_rhs = event_list.filter_by_rel_ts_range(475196, 506634)

if True:
    with set_list_min_ts(event_list_lhs): print(event_list_lhs)

    print("\n=============\n")

    with set_list_min_ts(event_list_rhs): print(event_list_rhs)

diff_event_list(event_list_lhs, event_list_rhs, alternative_print=False)

print("bye")
