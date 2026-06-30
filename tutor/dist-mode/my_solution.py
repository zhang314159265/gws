from collections import Counter

def all2all(ctx, data_list):
    for idx, data in enumerate(data_list):
        ctx.send(idx, data)
    recv_list = []
    for _ in range(len(data_list)):
        recv_list.append(ctx.recv())

    ctx.barrier()
    return recv_list

def gather(ctx, dst, data):
    out = []
    if ctx.worker_id == dst:
        out.append(data)
        for _ in range(ctx.WORKER_NUM - 1):
            out.append(ctx.recv())
    else:
        ctx.send(dst, data)

    ctx.barrier()
    return out

def broadcast(ctx, src, data):
    if ctx.worker_id == src:
        for idx in range(ctx.WORKER_NUM):
            if idx == src:
                continue
            ctx.send(idx, data)
    else:
        data = ctx.recv()
    ctx.barrier()
    return data

def distributed_mode(ctx):
    local_data = ctx.read()
    sharded_stat = [Counter() for _ in range(ctx.WORKER_NUM)]
    for val in local_data:
        sharded_stat[val % ctx.WORKER_NUM][val] += 1

    gathered_stat = all2all(ctx, sharded_stat)
    combined_stat = Counter()
    for stat in gathered_stat:
        combined_stat.update(stat)

    best_freq, best_val = -1, None
    for val, freq in combined_stat.items():
        if freq > best_freq or (freq == best_freq and val < best_val):
            best_freq, best_val = freq, val

    # gather on worker 0
    all_best_freq_val = gather(ctx, 0, (best_freq, best_val))
    if ctx.worker_id == 0:
        for cand_freq_val in all_best_freq_val:
            if cand_freq_val[0] < 0:
                continue
            if cand_freq_val[0] > best_freq or (cand_freq_val[0] == best_freq and cand_freq_val[1] < best_val):
                best_freq, best_val = cand_freq_val

    best_freq, best_val = broadcast(ctx, 0, (best_freq, best_val))
    return best_val

def mode_naive(ctx):
    stat = dict(Counter(ctx.read()))
    all_stat = gather(ctx, 0, stat)
    best_freq, best_val = -1, None
    if ctx.worker_id == 0:
        combined = Counter()
        for stat in all_stat:
            combined.update(stat)
        for val, freq in combined.items():
            if freq > best_freq or (freq == best_freq and val < best_val):
                best_freq, best_val = freq, val

    best_val = broadcast(ctx, 0, best_val)
    return best_val

def count_le(ctx, local_data, pivot):
    local_cnt = sum(x <= pivot for x in local_data)
    all_cnt = gather(ctx, 0, local_cnt)
    gcnt = 0
    if ctx.worker_id == 0:
        gcnt = sum(all_cnt)
    gcnt = broadcast(ctx, 0, gcnt)
    return gcnt

def findkth(ctx, local_data, gmin, gmax, k):
    left, right = gmin, gmax
    while left <= right:
        mid = left + (right - left) // 2
        if count_le(ctx, local_data, mid) >= k + 1:
            right = mid - 1
        else:
            left = mid + 1
    # print(f"findkth k={k}, val={left}")
    return left

def distributed_median(ctx):
    local_data = ctx.read()
    local_min, local_max, local_cnt = min(local_data, default=2**64), max(local_data, default=-2**64), len(local_data)
    all_stat = gather(ctx, 0, (local_min, local_max, local_cnt))

    global_min, global_max, global_cnt = None, None, 0
    if ctx.worker_id == 0:
        global_min, global_max, global_cnt = local_min, local_max, 0
        for other_min, other_max, other_cnt in all_stat:
            global_min = min(global_min, other_min)
            global_max = max(global_max, other_max)
            global_cnt += other_cnt

    global_min, global_max, global_cnt = broadcast(ctx, 0, (global_min, global_max, global_cnt))

    if global_cnt == 0:
        return None
    elif global_cnt % 2 == 1:
        return findkth(ctx, local_data, global_min, global_max, (global_cnt - 1) // 2)
    else:
        return (findkth(ctx, local_data, global_min, global_max, (global_cnt - 1) // 2) + findkth(ctx, local_data, global_min, global_max, (global_cnt + 1) // 2)) / 2
