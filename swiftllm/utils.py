def cdiv(a: int, b: int):
    return (a + b - 1) // b

KB = 1024
MB = 1024*1024
GB = 1024*1024*1024
TB = 1024*1024*1024*1024


def get_flash_attn_flop(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def get_flash_attn_memory(B, T, H, D, dtype_bytes=2) -> float:
    # 读取 Q, K, V 和写入 O
    memory_bytes = 4 * B * T * H * D * dtype_bytes
    return memory_bytes