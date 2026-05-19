import math
import random
import argparse

import torch

import dsv4_decode_kernel


torch.manual_seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--packed", action="store_true")
parser.add_argument("--sink", action="store_true")
parser.add_argument(
    "--vllm",
    action="store_true",
    help="Also run vLLM's ROCm Triton sparse decode baseline. Packed mode only.",
)
args = parser.parse_args()

B = args.batch
H = 64
N = 128
D = 512
dtype = torch.bfloat16
scale = 1.0 / math.sqrt(D)
block_size = 128

q = torch.randn(B, H, 1, D, dtype=dtype, device="cuda")
k = torch.randn(B, 1, N, D, dtype=dtype, device="cuda")
v = torch.randn(B, 1, N, D, dtype=dtype, device="cuda")
indices = torch.stack(
    [torch.randperm(N, device="cuda", dtype=torch.int32) for _ in range(B)],
    dim=0,
).view(B, 1, N, 1).contiguous()
out = torch.empty(B, H, 1, D, dtype=dtype, device="cuda")
attn_sink = torch.linspace(-0.1, 0.1, H, dtype=torch.float32, device="cuda")


def fp8_dtype():
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def pack_fp8_ds_mla_cache(kv):
    assert kv.shape == (B * N, D)
    cache = torch.zeros(
        (B, block_size, 584),
        dtype=torch.uint8,
        device="cuda",
    )
    cache_flat = cache.view(torch.uint8).flatten()
    kv_nope_fp8 = kv[:, :448].to(fp8_dtype()).view(torch.uint8)
    kv_rope_u8 = kv[:, 448:].contiguous().view(torch.uint8)

    for slot in range(kv.shape[0]):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + 448].copy_(kv_nope_fp8[slot])
        cache_flat[token_base + 448 : token_base + 448 + 64 * 2].copy_(
            kv_rope_u8[slot]
        )
        cache_flat[scale_base : scale_base + 7].fill_(127)
    return cache


def read_fp8_ds_mla_cache(cache, slot):
    cache_flat = cache.view(torch.uint8).flatten()
    block_idx = slot // block_size
    pos = slot % block_size
    block_base = block_idx * cache.stride(0)
    token_base = block_base + pos * 576
    nope = cache_flat[token_base : token_base + 448].view(fp8_dtype()).float()
    rope = cache_flat[token_base + 448 : token_base + 448 + 64 * 2].view(
        torch.bfloat16
    ).float()
    return torch.cat([nope, rope])


def ref_attention(q, k, v):
    idx = indices[:, 0, :, 0].long()
    batch_arange = torch.arange(B, device="cuda")[:, None]
    kg = k[:, 0][batch_arange, idx]
    vg = v[:, 0][batch_arange, idx]
    scores = torch.einsum("bhld,btd->bhlt", q.float(), kg.float())
    probs = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhlt,btd->bhld", probs, vg.float()).to(dtype)


def ref_packed_attention(q, cache):
    q_f32 = q.float()
    result = torch.empty_like(q_f32)
    idx = indices[:, 0, :, 0].long()
    for b in range(B):
        kv = torch.stack(
            [read_fp8_ds_mla_cache(cache, int(slot.item())) for slot in idx[b]]
        )
        for h in range(H):
            scores = torch.mv(kv, q_f32[b, h, 0]) * scale
            if args.sink:
                scores_with_sink = torch.cat([scores, attn_sink[h].reshape(1)])
                probs = torch.softmax(scores_with_sink, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            result[b, h, 0] = torch.sum(probs[:, None] * kv, dim=0)
    return result.to(dtype)


def make_vllm_ragged():
    # vLLM's ragged decode API expects one query row per batch item:
    # q=[B,H,D], indices=[B*N], indptr=[B+1].
    main_indices = indices[:, 0, :, 0].reshape(-1).contiguous()
    main_indptr = torch.arange(
        0,
        (B + 1) * N,
        N,
        device="cuda",
        dtype=torch.int32,
    )
    return main_indices, main_indptr


def vllm_sparse_decode(q, cache):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
    )

    main_indices, main_indptr = make_vllm_ragged()
    out_bhd = _rocm_sparse_attn_decode_ragged_triton(
        q=q[:, :, 0, :].contiguous(),
        main_cache=cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=attn_sink if args.sink else None,
        nope_head_dim=448,
        rope_head_dim=64,
    )
    return out_bhd[:, :, None, :]


if args.packed:
    # vLLM's sparse decode cache is paged globally, so each batch row points at
    # its own 128-token block here.
    kv_flat = k.view(B * N, D)
    cache = pack_fp8_ds_mla_cache(kv_flat)
    indices = (indices + (torch.arange(B, device="cuda", dtype=torch.int32) * N).view(B, 1, 1, 1)).contiguous()
    ref = ref_packed_attention(q, cache)
else:
    if args.vllm:
        raise SystemExit("--vllm requires --packed")
    cache = None
    ref = ref_attention(q, k, v)

for _ in range(10):
    if args.packed:
        dsv4_decode_kernel.dispatch_packed_decode(
            q, cache, indices, attn_sink, out, scale, int(args.sink)
        )
    else:
        dsv4_decode_kernel.dispatch_decode(q, k, v, indices, out, scale)
torch.cuda.synchronize()

diff = (out.float() - ref.float()).abs()
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
print(f"max_diff={diff.max().item():.6f}")
print(f"mean_diff={diff.mean().item():.6f}")
print(f"cos={cos.item():.8f}")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

iters = args.iters
start.record()
for _ in range(iters):
    if args.packed:
        dsv4_decode_kernel.dispatch_packed_decode(
            q, cache, indices, attn_sink, out, scale, int(args.sink)
        )
    else:
        dsv4_decode_kernel.dispatch_decode(q, k, v, indices, out, scale)
end.record()
torch.cuda.synchronize()
print(f"hk_ms={start.elapsed_time(end) / iters:.6f}")

start.record()
for _ in range(iters):
    if args.packed:
        ref = ref_packed_attention(q, cache)
    else:
        ref = ref_attention(q, k, v)
end.record()
torch.cuda.synchronize()
print(f"torch_ms={start.elapsed_time(end) / iters:.6f}")

if args.vllm:
    for _ in range(10):
        vllm_out = vllm_sparse_decode(q, cache)
    torch.cuda.synchronize()

    vllm_diff = (vllm_out.float() - ref.float()).abs()
    vllm_cos = torch.nn.functional.cosine_similarity(
        vllm_out.float().flatten(), ref.float().flatten(), dim=0
    )
    print(f"vllm_max_diff={vllm_diff.max().item():.6f}")
    print(f"vllm_mean_diff={vllm_diff.mean().item():.6f}")
    print(f"vllm_cos={vllm_cos.item():.8f}")

    start.record()
    for _ in range(iters):
        vllm_out = vllm_sparse_decode(q, cache)
    end.record()
    torch.cuda.synchronize()
    print(f"vllm_ms={start.elapsed_time(end) / iters:.6f}")
