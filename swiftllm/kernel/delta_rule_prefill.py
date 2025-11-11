import torch

from typing import Tuple


def prepare_wu(k: torch.Tensor,  # [b, s, h_k, d_qk]
               v: torch.Tensor,  # [b, s, h_k, d_v]
               beta: torch.Tensor,  # [b, s, h_q]
               chunk_size: int = 64
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the W and U for the delta rule prefill.
    Args:
        k: [b, s, h_k, d_qk]
        v: [b, s, h_k, d_v]
        beta: [b, s, h_q]
    Returns:
        W: [b, s, h_q, d_qk]
        U: [b, s, h_q, d_v]
    """
    b, s, h_k, d_qk = k.shape
    A_t = torch.zeros((b, h_k, s, s), device=k.device, dtype=torch.float32)
    # Step1: compute A
    k = k.transpose(1, 2)  # [b, h_k, s, d_qk]
    v = v.transpose(1, 2)  # [b, h_k, s, d_v]
    beta_transposed = beta.transpose(-1, -2)  # [b, h_q, s]
    K_KT = k.to(torch.float32) @ (k.transpose(-1, -2).to(torch.float32))  # [b, h_k, s, s]  # NOTE: We should use float32 here
    # print(K_KT.shape)
    # print(beta_transposed.unsqueeze(-1).shape)
    A_t = torch.tril(-beta_transposed.unsqueeze(-1) * K_KT, diagonal=-1)  # [b, h_q, s, s]

    # Step2: compute T
    I = torch.eye(s, device=k.device, dtype=torch.float32)  # [s, s]
    I_minus_A_t = I.unsqueeze(0).unsqueeze(0) - A_t  # [b, h_q, s, s]
    T_t = torch.linalg.inv(I_minus_A_t)  # [b, h_q, s, s]

    # Step3: compute w
    beta_K_t = beta_transposed.unsqueeze(-1) * k  # [b, h_q, s, d_qk]
    W_t: torch.Tensor = T_t.to(k.dtype) @ beta_K_t  # [b, h_q, s, d_qk]

    # Step4: compute u
    beta_v_t = beta_transposed.unsqueeze(-1) * v  # [b, h_q, s, d_v]
    U_t: torch.Tensor = T_t.to(v.dtype) @ beta_v_t  # [b, h_q, s, d_v]

    return W_t.transpose(1, 2), U_t.transpose(1, 2)


def compute_output(q: torch.Tensor,  # [b, s, h_q, d_qk]
                   k: torch.Tensor,  # [b, s, h_k, d_qk]
                   v: torch.Tensor,  # [b, s, h_k, d_v]
                   state: torch.Tensor,  # [b, h_q, d_v, d_qk]
                   W_t: torch.Tensor,  # [b, s, h_q, d_qk]
                   U_t: torch.Tensor,  # [b, s, h_q, d_v]
                   ) -> torch.Tensor:
    q = q.transpose(1, 2)  # [b, h_q, s, d_qk]
    k = k.transpose(1, 2)  # [b, h_k, s, d_qk]
    v = v.transpose(1, 2)  # [b, h_k, s, d_v]
    W_t = W_t.transpose(1, 2)  # [b, h_q, s, d_qk]
    U_t = U_t.transpose(1, 2)  # [b, h_q, s, d_v]

    Q_ST = q @ state.transpose(-1, -2)  # [b, h_q, s, d_v]
    Q_KT = q @ k.transpose(-1, -2)  # [b, h_q, s, s]
    masked_Q_KT = Q_KT.masked_fill(torch.tril(torch.ones_like(Q_KT)) == 0, 0)
    output = Q_ST + masked_Q_KT @ (U_t - W_t @ state.transpose(-1, -2))  # [b, h_q, s, d_v]
    return output.transpose(1, 2)  # [b, h_q, s, d_v]


def reference_torch_delta_rule_prefill(q: torch.Tensor,  # [b, s_q, h_q, d_qk]
                                       k: torch.Tensor,  # [b, s_kv, h_kv, d_qk]
                                       v: torch.Tensor,  # [b, s_kv, h_kv, d_v]
                                       beta: torch.Tensor,  # [b, s_q, h_q]
                                       head_first: bool = False,
                                       output_final_state: bool = False
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    CHUNK_SIZE = 64
    b, s, h_q, d_qk = q.shape
    h_k = k.shape[-2]
    d_v = v.shape[-1]
    assert s % CHUNK_SIZE == 0, "We assume the sequence length is a multiple of the chunk size now"
    assert h_q == h_k, "We assume the number of heads in Q and K are the same"
    # print(b, s, h_q, d_qk)
    state = torch.zeros((b, h_q, d_v, d_qk), device=q.device, dtype=q.dtype)
    output = torch.empty((b, s, h_q, d_v), device=q.device, dtype=q.dtype)
    q = q * (d_qk ** -0.5)

    for t in range(0, s // CHUNK_SIZE):
        s_start = t * CHUNK_SIZE
        s_end = min(s_start + CHUNK_SIZE, s)
        q_tile = q[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_q, d_qk]
        k_tile = k[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_k, d_qk]
        v_tile = v[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_k, d_v]
        beta_tile = beta[:, s_start:s_end, :]  # [b, CHUNK_SIZE, h_q]

        W_t, U_t, _ = prepare_wu(k_tile, v_tile, beta_tile)

        output[:, s_start:s_end, :, :] = compute_output(q_tile, k_tile, v_tile, state, W_t, U_t)

        state = state + ((U_t - W_t @ state.transpose(-1, -2)) @ k_tile.transpose(1, 2))

    return output, state
