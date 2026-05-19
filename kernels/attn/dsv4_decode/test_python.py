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
args = parser.parse_args()

B = args.batch
H = 64
N = 128
D = 512
dtype = torch.bfloat16
scale = 1.0 / math.sqrt(D)

q = torch.randn(B, H, 1, D, dtype=dtype, device="cuda")
k = torch.randn(B, 1, N, D, dtype=dtype, device="cuda")
v = torch.randn(B, 1, N, D, dtype=dtype, device="cuda")
indices = torch.stack(
    [torch.randperm(N, device="cuda", dtype=torch.int32) for _ in range(B)],
    dim=0,
).view(B, 1, N, 1).contiguous()
out = torch.empty(B, H, 1, D, dtype=dtype, device="cuda")


def ref_attention(q, k, v):
    idx = indices[:, 0, :, 0].long()
    batch_arange = torch.arange(B, device="cuda")[:, None]
    kg = k[:, 0][batch_arange, idx]
    vg = v[:, 0][batch_arange, idx]
    scores = torch.einsum("bhld,btd->bhlt", q.float(), kg.float())
    probs = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhlt,btd->bhld", probs, vg.float()).to(dtype)


ref = ref_attention(q, k, v)

for _ in range(10):
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
    dsv4_decode_kernel.dispatch_decode(q, k, v, indices, out, scale)
end.record()
torch.cuda.synchronize()
print(f"hk_ms={start.elapsed_time(end) / iters:.6f}")

start.record()
for _ in range(iters):
    ref = ref_attention(q, k, v)
end.record()
torch.cuda.synchronize()
print(f"torch_ms={start.elapsed_time(end) / iters:.6f}")
