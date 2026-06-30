from collections import defaultdict, Counter

def all2all(ctx, local_shards):
    for idx, shard in enumerate(local_shards):
        ctx.send(idx, shard)

    recv_list = []
    for _ in range(len(local_shards)):
        recv_list.append(ctx.recv())

    ctx.barrier()
    return recv_list

def gather(ctx, dst_rank, obj):
    out = []
    if ctx.worker_id != dst_rank:
        ctx.send(dst_rank, obj)
    else:
        out.append(obj)
        for _ in range(ctx.WORKER_NUM - 1):
            out.append(ctx.recv())

    ctx.barrier()
    return out

def broadcast(ctx, src_rank, obj):
    out = None
    if ctx.worker_id == src_rank:
        for rank in range(ctx.WORKER_NUM):
            if rank != src_rank:
                ctx.send(rank, obj)
        out = obj
    else:
        out = ctx.recv()
    ctx.barrier()
    return out

def distributed_mode(ctx):
    nworker = ctx.WORKER_NUM

    local_data = ctx.read()
    local_stat_shards = [defaultdict(int) for _ in range(nworker)]
    for val in local_data:
        # can do a hash for val depending on how val is generated
        local_stat = local_stat_shards[val % nworker]
        local_stat[val] += 1

    # all2all
    all_shards = all2all(ctx, local_stat_shards)

    maxfreq = -1
    ans = None

    combined = defaultdict(int)
    for shard in all_shards:
        for val, freq in shard.items():
            combined[val] += freq
    for key, freq in combined.items():
        if freq > maxfreq or (freq == maxfreq and key < ans):
            maxfreq = freq
            ans = key

    # gather to rank0
    stat_list = gather(ctx, 0, (maxfreq, ans))

    if ctx.worker_id == 0:
        for freq, val in stat_list:
            if freq < 0:
                continue
            if freq > maxfreq or (freq == maxfreq and val < ans):
                maxfreq = freq
                ans = val

    maxfreq, ans = broadcast(ctx, 0, (maxfreq, ans))
    return ans

def count_le(ctx, local_data, pivot):
    local_cnt = sum(x <= pivot for x in local_data)
    cntlist = gather(ctx, 0, local_cnt)
    global_count = None
    if ctx.worker_id == 0:
        global_count = sum(cntlist)
    return broadcast(ctx, 0, global_count)

def findkth(ctx, local_data, gmin, gmax, k):
    left, right = gmin, gmax
    while left <= right:
        mid = left + (right - left) // 2
        cnt = count_le(ctx, local_data, mid)
        # print(f"{mid=} {k=} {mid=} {cnt=}")
        if cnt >= k + 1:
            right = mid - 1
        else:
            left = mid + 1
    return left

def distributed_median(ctx):
    local_data = ctx.read()
    minval = min(local_data, default=2**64)
    maxval = max(local_data, default=-2**64)
    countval = len(local_data)

    minlist = gather(ctx, 0, minval)
    ctx.barrier()
    maxlist = gather(ctx, 0, maxval)
    ctx.barrier()
    countlist = gather(ctx, 0, countval)
    ctx.barrier()
    # print(f"{ctx.worker_id=} {countlist=}")

    if ctx.worker_id == 0:
        gmin = min(minlist)
        gmax = max(maxlist)
        gtot = sum(countlist)
    else:
        gmin = gmax = gtot = None

    gmin, gmax, gtot = broadcast(ctx, 0, (gmin, gmax, gtot))

    # print(f"{ctx.worker_id=} {gtot=}")

    if gtot == 0:
        return None
    elif gtot % 2 == 1:
        # print(f"{ctx.worker_id=} {sorted(local_data)=}, {sorted(local_data)[200]=}")
        return findkth(ctx, local_data, gmin, gmax, gtot // 2)
    else:
        return (
            findkth(ctx, local_data, gmin, gmax, gtot // 2) +
            findkth(ctx, local_data, gmin, gmax, gtot // 2 - 1)
        ) / 2



def mode_naive(ctx):
    local_data = ctx.read()
    local_stat = dict(Counter(local_data))
    all_stat = gather(ctx, 0, local_stat)

    ans = None
    if ctx.worker_id == 0:
        stat = Counter()

        for shard in all_stat:
            stat.update(shard)
        maxfreq = -1
        for val, freq in stat.items():
            if freq > maxfreq or (freq == maxfreq and val < ans):
                maxfreq = freq
                ans = val

    ans = broadcast(ctx, 0, ans)
    return ans
