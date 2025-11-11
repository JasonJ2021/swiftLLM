# -*- coding: utf-8 -*-

import os
import math
import time
from flash_attn import flash_attn_func

import torch
import triton
# from flash_attn import flash_attn_func

from fla.ops.retention import chunk_retention, parallel_retention
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
from swiftllm.utils import get_flash_attn_flop, get_flash_attn_memory


def profile_chunk_delta_rule(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    from fla.utils import device
    requires_grad = False
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(T)
    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)

    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

    print("Fused Chunk Delta Rule")
    results = triton.testing.do_bench(lambda: fused_chunk_delta_rule(q, k, v, beta), quantiles=[0.5, 0.2, 0.8])
    print(f"Time: {results[0] / 1000} seconds")

def profile_flash_attention(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    # Kernel name: flash_fwd_kernel
    from fla.utils import device
    requires_grad = False
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(T)
    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    print("Flash Attention")
    results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=[0.5, 0.2, 0.8])
    print(f"Time: {results[0] / 1000} seconds")
    flop = get_flash_attn_flop(B, T, H, D, causal=True)
    memory_bytes = get_flash_attn_memory(B, T, H, D)
    flash_ans_time = results[0] / 1000
    flash_flops = flop / flash_ans_time / 1e12
    memory_gb = memory_bytes / (1024 * 1024 * 1024)
    flash_bandwidth = memory_gb / flash_ans_time
    print(f"Flash Attention: {flash_ans_time * 1e6:4.0f} us, {flash_flops:.3f} TFlops")
    print(f"Flash Attention Bandwidth: {flash_bandwidth:.3f} GB/s")

if __name__ == '__main__':
    # profile_flash_attention(B=1, T=4096, H=128, D=256, dtype=torch.bfloat16)
    profile_chunk_delta_rule(B=1, T=4096, H=128, D=256, dtype=torch.bfloat16)
