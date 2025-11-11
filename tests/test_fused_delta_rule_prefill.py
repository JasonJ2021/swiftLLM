import math
import time
from typing import Tuple
import random
import dataclasses

import torch
import triton

from lib import check_is_allclose
from fla.utils import assert_close
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.delta_rule.naive import delta_rule_chunkwise
from swiftllm.kernel.delta_rule_prefill import reference_torch_delta_rule_prefill


@dataclasses.dataclass
class TestParam:
    b: int
    s_q: int
    s_kv: int
    h_q: int = 8
    h_kv: int = 8
    d_qk: int = 128
    d_v: int = 128
    seed: int = 0
    check_correctness: bool = True
    benchmark: bool = False


@dataclasses.dataclass
class Testcase:
    t: TestParam
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor  # Required by the delta rule


def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)
    q = torch.randn((t.b, t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16) / 10
    k = torch.randn((t.b, t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16) / 10
    v = torch.randn((t.b, t.s_kv, t.h_kv, t.d_v), dtype=torch.bfloat16) / 10
    beta = torch.rand((t.b, t.s_q, t.h_q), device=device, dtype=torch.bfloat16).sigmoid()

    q.clamp_(-10, 10)
    k.clamp_(-10, 10)
    v.clamp_(-10, 10)

    return Testcase(
        t=t,
        q=q,
        k=k,
        v=v,
        beta=beta
    )


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)

    def run_ans(t: Testcase):
        # TODO: Replace this with OUR Prefil kernel
        return reference_torch_delta_rule_prefill(t.q, t.k, t.v, t.beta, output_final_state=True)
        # return chunk_delta_rule(t.q, t.k, t.v, t.beta)

    ans_out, ans_final_state = run_ans(t)
    torch.cuda.synchronize()

    if p.benchmark:
        pass

    if p.check_correctness:
        torch.cuda.synchronize()
        # TODO: Implement correctness check
        ref_out, ref_final_state = chunk_delta_rule(t.q, t.k, t.v, t.beta, output_final_state=True)
        torch.cuda.synchronize()
        # print(f"out max diff: {(ref_out - ans_out).abs().max().item()}")
        # print(f"out min diff: {(ref_out - ans_out).abs().min().item()}")
        # print(f"state max diff: {(ref_final_state - ans_final_state).abs().max().item()}")
        # print(f"state min diff: {(ref_final_state - ans_final_state).abs().min().item()}")
        is_correct = True
        try:
            assert_close("out", ref_out, ans_out, 0.006)
            assert_close("final_state", ref_final_state, ans_final_state, 0.006)
        except Exception as e:
            is_correct = False
            
        # is_correct &= check_is_allclose("out", ans_out, ref_out, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=9e-6)
        # is_correct &= check_is_allclose("final_state", ans_final_state, ref_final_state,
        #                                 abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=7e-6)
        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        TestParam(b=1, s_q=s_q, s_kv=s_q, h_q=128, h_kv=128, d_qk=128, d_v=128, seed=0, check_correctness=True, benchmark=False)
        for s_q in [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ]

    corner_cases = [
    ]

    performance_cases = [
    ]

    testcases = correctness_cases + corner_cases + performance_cases

    failed_cases = []
    for test in testcases:
        if test.benchmark:
            time.sleep(0.2)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
