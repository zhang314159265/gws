"""
NOTE: the script right now only benchmark the latency of the attention
kernel itself. The following things are excluded
- add new key/value to the cache
- setup BlockMask for flex-attention
- etc.

"""

import math
import torch
import triton
from torch.nn.attention.flex_attention import flex_attention, BlockMask
import torch.nn.functional as F
import functools
from torch._inductor.utils import do_bench_using_profiling

torch.manual_seed(1337)

flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)
# flex_attention_compiled = flex_attention

class script_args:
    batch_size = 8
    kv_seq_len = 128
    page_table_block_size = 16
    # XXX 70000 is too large and cause int64 indexing. Trigger some flex
    # decoding bug
    # num_page_table_blocks = 70000  # vllm reserves about this many blocks for llama3-8B on B200
    num_page_table_blocks = 10000  # vllm reserves about this many blocks for llama3-8B on B200
    num_query_head = 32
    num_kv_head = 8
    # num_kv_head = 16 # larger kvhead to make sure flex-decoding is not bypassed
    head_dim = 128

def create_page_table():
    """
    For 4 requests, the kvcache will contain blocks like:
    0, req0_blk0, req1_blk0, req2_blk0, req3_blk0, req0_blk1, req1_blk1, req2_blk1, req3_blk1, ... unallocated blocks. So the corresponding page_table
    for the batch of (req0, req1, req2, req3) will look like
    [
        [1, 5, 9,..., 0, 0, 0],
        [2, 6, 10, ...,  0, 0],
        [3, 7, 11, ...,  0, 0],
        [4, 8, 12, ...,  0, 0],
    ]

    This assumes that prefill only uses a single kv cache block. Otherwise,
    the allocattion pattern will be slightly different.
    """
    kv_seq_len = script_args.kv_seq_len
    block_size = script_args.page_table_block_size
    batch_size = script_args.batch_size

    nblock = (kv_seq_len + block_size - 1) // block_size
    # the number 512 is get by running vllm for llama3-8B
    page_table = torch.zeros(batch_size, 512, device="cuda", dtype=torch.int32)
    assert nblock <= page_table.size(1)
    
    # very naive way to setup the page table
    for blk_id in range(1, batch_size * nblock + 1):
        row = (blk_id - 1) % batch_size
        col = (blk_id - 1) // batch_size
        # make sure a offset can be handled properly
        page_table[row, col] = blk_id + 245
    return page_table

query = torch.randn([script_args.batch_size, script_args.num_query_head, script_args.head_dim], dtype=torch.bfloat16, device="cuda")
kv_cache = torch.randn(
    2, script_args.num_page_table_blocks, script_args.page_table_block_size,
    script_args.num_kv_head,
    script_args.head_dim,
    dtype=torch.bfloat16,
    device="cuda") # .fill_(0.5) make it pass
page_table = create_page_table()

def _convert_physical_to_logical(doc_ids, physical_to_logical, q_idx, physical_kv_idx):
    q_req = doc_ids[q_idx]

    block_size = script_args.page_table_block_size

    physical_kv_block = physical_kv_idx // block_size
    physical_kv_offset = physical_kv_idx % block_size
    logical_block_idx = physical_to_logical[q_req, physical_kv_block]
    logical_kv_idx = logical_block_idx * block_size + physical_kv_offset

    # determine valid kv indices
    live_block = logical_block_idx >= 0
    within_upper_bound = logical_kv_idx < script_args.kv_seq_len
    within_lower_bound = logical_kv_idx >= 0
    is_valid = live_block & within_upper_bound & within_lower_bound

    # convert q index
    query_start_loc = torch.arange(script_args.batch_size, device="cuda", dtype=torch.int32)
    local_q_idx = q_idx - query_start_loc[q_req]
    logical_q_idx = local_q_idx + script_args.kv_seq_len - 1

    return is_valid, logical_q_idx, logical_kv_idx

def _paged_mask_mod(b, h, q_idx, physical_kv_idx, *, doc_ids, physical_to_logical):
    is_valid, logical_q_idx, logical_kv_idx = _convert_physical_to_logical(
        doc_ids, physical_to_logical, q_idx, physical_kv_idx
    )
    return torch.where(
        is_valid,
        logical_kv_idx <= logical_q_idx,
        False,
    )

def _create_flex_attn_block_mask(page_table):
    """
    Follow vllm.v1.attention.backends.flex_attention._build_block_mask_direct
    """
    from vllm.v1.attention.backends.flex_attention import unique_static_unsorted, physical_to_logical_mapping
    batch_size = script_args.batch_size

    doc_ids = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    max_seq_len = script_args.kv_seq_len
    block_size = script_args.page_table_block_size

    used_pages = page_table[doc_ids, : (max_seq_len + block_size - 1) // block_size]
    q_block_size = 16
    npad = q_block_size - (batch_size % q_block_size)
    if npad > 0:
        used_pages = F.pad(used_pages, [0, 0, 0, npad], mode="constant", value=0)
    used_pages = used_pages.reshape(used_pages.shape[0] // q_block_size, - 1)
    kv_indices = unique_static_unsorted(
        used_pages.long(), M=script_args.num_page_table_blocks,
    ).to(torch.int32)
    kv_num_blocks = (kv_indices >= 0).sum(dim=-1).to(torch.int32)

    # NOTE seq_lens means kv seq lens rather than q seq lens!
    seq_lens = torch.ones(batch_size, device="cuda", dtype=torch.int32) * script_args.kv_seq_len
    physical_to_logical = physical_to_logical_mapping(
        page_table, seq_lens, block_size, script_args.num_page_table_blocks,
    )
    return BlockMask.from_kv_blocks(
        seq_lengths=(batch_size, script_args.page_table_block_size * script_args.num_page_table_blocks),
        kv_num_blocks=kv_num_blocks[None, None],
        kv_indices=kv_indices[None, None],
        full_kv_num_blocks=None,
        compute_q_blocks=False,
        BLOCK_SIZE=(16, 16),
        mask_mod=functools.partial(_paged_mask_mod, doc_ids=doc_ids, physical_to_logical=physical_to_logical),
    )

def run_flex_attn(query, kv_cache, page_table, force_use_flex_attention=False):
    key_cache, value_cache = kv_cache.unbind(0)
    key_cache = key_cache.view(-1, script_args.num_kv_head, script_args.head_dim)
    value_cache = value_cache.view(-1, script_args.num_kv_head, script_args.head_dim)

    query, key_cache, value_cache = map(
        lambda x: x[None, ...].permute(0, 2, 1, 3),
        (query, key_cache, value_cache),
    )

    scale = 1.0 / math.sqrt(script_args.head_dim)
    enable_gqa = script_args.num_query_head != script_args.num_kv_head

    kernel_options = dict(
        BLOCK_M=16,  # BLOCK_M 16 will make flex decoding being bypassed for some query/kv head setup
        BLOCK_N=16,
        FORCE_USE_FLEX_ATTENTION=force_use_flex_attention,
    )

    block_mask = _create_flex_attn_block_mask(page_table)
    def f():
        out = flex_attention_compiled(
            query,
            key_cache,
            value_cache,
            None, # score mod
            block_mask,
            scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )
        return out[0].transpose(0, 1)
    return f

def run_flash_attn(query, kv_cache, page_table):
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func

    key_cache, value_cache = kv_cache.unbind(0)
    output = torch.empty_like(query)

    batch_size = script_args.batch_size
    kv_seq_len = script_args.kv_seq_len

    cu_seqlens_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)
    max_seqlen_q = 1
    seqused_k = torch.full([batch_size], kv_seq_len, device="cuda", dtype=torch.int32)
    max_seqlen_k = kv_seq_len
    scale = 1.0 / math.sqrt(script_args.head_dim)
    def f():
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=None,
            window_size=(-1, -1),
            block_table=page_table,
            softcap=0,
            scheduler_metadata=None,
            fa_version=2,  # vllm picks FA2 . If I force to use FA3, I get numerical errors!
            num_splits=0,
            s_aux=None,
            # qkv descale?
        )
        return output
    return f

def run_flash_infer(query, kv_cache, page_table):
    """
    vllm flashinfer backend has 4 general cases:
    1. prefill with trtllm kernels
    2. prefill with flashinfer kernels
    3. decode with trtllm kernels
    4. decode with flashinfer kernels
    This function only covers case 3 for now.
    """
    from vllm.utils.flashinfer import use_trtllm_attention
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    sa = script_args
    assert use_trtllm_attention(
        sa.num_query_head,
        sa.num_kv_head,
        sa.batch_size,
        sa.kv_seq_len,
        kv_cache_dtype="auto",
        q_dtype=query.dtype,
        is_prefill=False,
        has_sinks=False,
        has_spec=False,
        dcp_world_size=1,
    )

    decode_query = query.contiguous()
    workspace_buffer = torch.zeros(
        394 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    kv_cache_permute = kv_cache.permute([1, 0, 3, 2, 4]).contiguous()
    assert kv_cache_permute.is_contiguous()
    assert workspace_buffer.is_contiguous()
    assert page_table.is_contiguous()
    output = torch.empty_like(query)
    batch_size = script_args.batch_size
    # NOTE seq_lens means kv seq lens rather than q seq lens!
    seq_lens = torch.ones(batch_size, device="cuda", dtype=torch.int32) * script_args.kv_seq_len
    max_seq_len = script_args.kv_seq_len
    scale = 1.0 / math.sqrt(script_args.head_dim)

    def f():
        trtllm_batch_decode_with_kv_cache(
            query=decode_query,
            kv_cache=kv_cache_permute,
            workspace_buffer=workspace_buffer,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            o_sf_scale=None,
            out=output,
            q_len_per_req=1,
        )
        return output

    return f

def run_triton_attn(query, kv_cache, page_table):
    from vllm.attention.ops.triton_unified_attention import unified_attention

    # triton attn uses a different kvcache shape compared to flashattn
    # XXX: Is this conversion necessary for perf? benchmark shows the same perf
    kv_cache = kv_cache.transpose(0, 1).contiguous()
    key_cache, value_cache = kv_cache.unbind(1)
    output = torch.empty_like(query)

    batch_size = script_args.batch_size
    kv_seq_len = script_args.kv_seq_len

    cu_seqlens_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)
    max_seqlen_q = 1
    seqused_k = torch.full([batch_size], kv_seq_len, device="cuda", dtype=torch.int32)
    max_seqlen_k = kv_seq_len
    scale = 1.0 / math.sqrt(script_args.head_dim)


    def f():
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=None,
            window_size=(-1, -1),
            block_table=page_table,
            softcap=0,
            q_descale=None,
            # kv descale? it's fine to not pass in a tensor.
            # Used for fp8 dtype.
            k_descale=None,
            v_descale=None,
            sinks=None,
            output_scale=None,
        )
        return output
    return f

def run_naive(query, kv_cache, page_table):
    key_cache, value_cache = kv_cache.unbind(0)
    kv_seq_len = script_args.kv_seq_len
    block_size = script_args.page_table_block_size
    batch_size = script_args.batch_size
    num_query_head = script_args.num_query_head
    num_kv_head = script_args.num_kv_head
    head_dim = script_args.head_dim
    G = num_query_head // num_kv_head
    scale = 1.0 / math.sqrt(script_args.head_dim)
    assert kv_seq_len % block_size == 0
    nblock_per_req = kv_seq_len // block_size
    page_table = page_table[:, :nblock_per_req]

    def f():
        # key_cache [N_PHY_BLK, BLK, H, D]
        # page_table [B, N_LOG_BLK]
        # key_cache[page_table]: [B, N_LOG_BLK, BLK, H, D]
        # [B, S, H, D]
        key_tensor = key_cache[page_table].reshape(batch_size, -1, num_kv_head, head_dim)
        value_tensor = value_cache[page_table].reshape(batch_size, -1, num_kv_head, head_dim)

        if G > 1:
            key_tensor = key_tensor[:, :, :, None, :].expand(-1, -1, -1, G, -1).reshape(batch_size, -1, num_query_head, head_dim)
            value_tensor = value_tensor[:, :, :, None, :].expand(-1, -1, -1, G, -1).reshape(batch_size, -1, num_query_head, head_dim)
    
        # query [B, H, D]
        key_tensor = key_tensor.transpose(1, 2) # [B, H, S, D]
        value_tensor = value_tensor.transpose(1, 2)


        # p [B, H, 1, S]
        p = (query[:, :, None, :] @ key_tensor.transpose(-1, -2)).float() * scale
        p = p.softmax(dim=-1)
        # out [B, H, 1, D]
        out = p.to(query.dtype) @ value_tensor
        return out.reshape_as(query)
    return f

f_naive = run_naive(query, kv_cache, page_table)
# print("naive", f_naive()[0])
f_flex_attn = run_flex_attn(query, kv_cache, page_table, force_use_flex_attention=True)
# print("flexattn", f_flex_attn()[0])
f_flex_decode = run_flex_attn(query, kv_cache, page_table, force_use_flex_attention=False)
# print("flexdecode", f_flex_decode()[0])
f_flash_attn = run_flash_attn(query, kv_cache, page_table)
f_flash_infer = run_flash_infer(query, kv_cache, page_table)
# print("flashattn", f_flash_attn()[0])
f_triton_attn = run_triton_attn(query, kv_cache, page_table)
# print("tritonattn", f_triton_attn()[0])

ref = f_naive()

def verify_and_bench(label, ref, func, profile=False, cudagraph=True):
    tol = dict(atol=1e-2, rtol=1e-2)
    out = func()
    try:
        torch.testing.assert_close(ref, out, **tol)
        print(f"\033[32m{label} Accuracy test pass!\033[0m")
    except:
        print(f"\033[31m{label} Accuracy test fail!\033[0m")
    do_bench = do_bench_using_profiling
    # do_bench = triton.testing.do_bench
    ms = do_bench(func)
    if profile:
        if cudagraph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                func()
            func = lambda: graph.replay()
        with torch.profiler.profile() as p:
            func()
        path = f"/tmp/{label}.json"
        print(f"Trace writtern to {path}")
        p.export_chrome_trace(path)
    print(f"{label}: {ms:.3f} ms")

verify_and_bench("naive", ref, f_naive)

verify_and_bench("flash_attn", ref, f_flash_attn, profile=True)
verify_and_bench("flash_infer", ref, f_flash_infer, profile=True)

verify_and_bench("triton_attn", ref, f_triton_attn, profile=True)
verify_and_bench("flex_attn", ref, f_flex_attn, profile=True)
verify_and_bench("flex_decode", ref, f_flex_decode, profile=True)

print("pass")
