import torch

from typing import Tuple


def reference_torch_delta_rule_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, output_final_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h_q, d_qk = q.shape
    CHUNK_SIZE = 64
    print(b, s, h_q, d_qk)

     

    return None, None
