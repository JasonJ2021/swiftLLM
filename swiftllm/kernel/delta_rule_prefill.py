import torch

from typing import Tuple


def prepare_wu(k: torch.Tensor,  # [b, s, h_k, d_qk]
               v: torch.Tensor,  # [b, s, h_k, d_v]
               beta: torch.Tensor,  # [b, s, h_q]
               chunk_size: int = 64
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    A_t = torch.zeros((b, h_k, s, chunk_size), device=k.device, dtype=torch.float32)
    T_t = torch.zeros((b, h_k, s, chunk_size), device=k.device, dtype=torch.float32)
    I = torch.eye(chunk_size, device=k.device, dtype=torch.float32)  # [s, s]
    k = k.transpose(1, 2)  # [b, h_k, s, d_qk]
    v = v.transpose(1, 2)  # [b, h_k, s, d_v]
    W_t = torch.empty_like(k)
    U_t = torch.empty_like(v)

    beta_transposed = beta.transpose(-1, -2)  # [b, h_q, s]
    for t in range(0, s // chunk_size):
        s_start = t * chunk_size
        s_end = min(s_start + chunk_size, s)
        k_tile = k[:, :, s_start:s_end, :]  # [b, h_k, chunk_size, d_qk]
        beta_tile = beta_transposed[:, :, s_start:s_end]  # [b, h_q, chunk_size]
        # [b, h_k, chunk_size, chunk_size]  # NOTE: We should use float32 here
        K_KT = k_tile.to(torch.float32) @ (k_tile.transpose(-1, -2).to(torch.float32))

        # Step1: compute A_t and T_t
        A_t[:, :, s_start:s_end, :] = torch.tril(-beta_tile.unsqueeze(-1) * K_KT, diagonal=-1)  # [b, h_k, chunk_size, chunk_size]

        # Step2: compute T
        I_minus_A_t = I.unsqueeze(0).unsqueeze(0) - A_t[:, :, s_start:s_end, :]  # [b, h_k, chunk_size, chunk_size]
        T_t[:, :, s_start:s_end, :] = torch.linalg.inv(I_minus_A_t)  # [b, h_k, chunk_size, chunk_size]

        # Step3: compute w
        beta_K_t = beta_transposed[:, :, s_start:s_end].unsqueeze(-1) * k[:, :, s_start:s_end, :]  # [b, h_q, chunk_size, d_qk]
        W_t[:, :, s_start:s_end, :] = T_t[:, :, s_start:s_end, :].to(k.dtype) @ beta_K_t  # [b, h_k, chunk_size, d_qk]

        # Step4: compute u
        beta_v_t = beta_transposed[:, :, s_start:s_end].unsqueeze(-1) * v[:, :, s_start:s_end, :]  # [b, h_k, chunk_size, d_v]
        U_t[:, :, s_start:s_end, :] = T_t[:, :, s_start:s_end, :].to(v.dtype) @ beta_v_t  # [b, h_k, chunk_size, d_v]

    return W_t.transpose(1, 2), U_t.transpose(1, 2), T_t.transpose(1, 2)


def compute_output(q: torch.Tensor,  # [b, s, h_q, d_qk]
                   k: torch.Tensor,  # [b, s, h_k, d_qk]
                   v: torch.Tensor,  # [b, s, h_k, d_v]
                   state: torch.Tensor,  # [b, h_q, d_qk, d_v]
                   W_t: torch.Tensor,  # [b, s, h_q, d_qk]
                   U_t: torch.Tensor,  # [b, s, h_q, d_v]
                   ) -> torch.Tensor:
    orig_dtype = q.dtype
    q = q.transpose(1, 2)  # [b, h_q, s, d_qk]
    k = k.transpose(1, 2)  # [b, h_k, s, d_qk]
    v = v.transpose(1, 2)  # [b, h_k, s, d_v]
    W_t = W_t.transpose(1, 2)  # [b, h_q, s, d_qk]
    U_t = U_t.transpose(1, 2)  # [b, h_q, s, d_v]

    # Compute Q @ state with higher precision
    Q_ST = q.float() @ state  # [b, h_q, s, d_v]
    
    # Compute intra-chunk attention with float32 precision
    Q_KT = (q.float() @ k.float().transpose(-1, -2))  # [b, h_q, s, s]
    mask = torch.triu(torch.ones(Q_KT.shape[-2], Q_KT.shape[-1], dtype=torch.bool, device=Q_KT.device), diagonal=1)
    masked_Q_KT = Q_KT.masked_fill(mask, 0)  # Keep lower triangular (including diagonal)
    
    # Compute u_i = U_t - W_t @ state with higher precision
    u_i = U_t.float() - (W_t.float() @ state)  # [b, h_q, s, d_v]
    
    # Final output computation in float32, then convert back
    output = (Q_ST + masked_Q_KT @ u_i).to(orig_dtype)  # [b, h_q, s, d_v]
    return output.transpose(1, 2)  # [b, s, h_q, d_v]


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
    state = torch.zeros((b, h_q, d_qk, d_v), device=q.device, dtype=torch.float32)
    output = torch.empty((b, s, h_q, d_v), device=q.device, dtype=q.dtype)
    q = q * (d_qk ** -0.5)
    W, U, _A = prepare_wu(k, v, beta) 

    for t in range(0, s // CHUNK_SIZE):
        s_start = t * CHUNK_SIZE
        s_end = min(s_start + CHUNK_SIZE, s)
        q_tile = q[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_q, d_qk]
        k_tile = k[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_k, d_qk]
        v_tile = v[:, s_start:s_end, :, :]  # [b, CHUNK_SIZE, h_k, d_v]

        # W_t, U_t, _ = prepare_wu(k_tile, v_tile, beta_tile)  # [b, chunk_size, h_q, d_qk] and [b, chunk_size, h_q, d_v]
        W_t = W[:, s_start:s_end, :, :]
        U_t = U[:, s_start:s_end, :, :]

        output[:, s_start:s_end, :, :] = compute_output(q_tile, k_tile, v_tile, state, W_t, U_t)
        
        # Update S 
        U_t = U_t.transpose(1, 2) # [b, h_q, chunk_size, d_v]
        W_t = W_t.transpose(1, 2) # [b, h_q, chunk_size, d_qk]
        k_t = k_tile.transpose(1, 2) # [b, h_q, chunk_size, d_qk]
        u_i = U_t - W_t @ state.to(W_t.dtype)  # [b, h_q, chunk_size, d_v]
        state = state + (k_t.transpose(-1, -2) @ u_i).to(state.dtype)  # [b, h_q, d_qk, d_v]

    return output, state
