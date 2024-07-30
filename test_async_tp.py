import os
from typing import List

import torch
import torch.distributed as dist

from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    restride_A_shard_for_fused_all_gather_matmul,
    enable_symm_mem_for_group,
)


def test_all_gather_matmul(group: dist.ProcessGroup):
    BATCH = 8
    M = 4096
    N = 1024
    K = 8192

    rank = group.rank()
    A_shard = torch.rand(
        (BATCH, M // group.size(), K), dtype=torch.bfloat16, device="cuda"
    ).normal_(mean=0, std=1)
    Bs = [
        torch.rand((N, K), dtype=torch.bfloat16, device="cuda").normal_(mean=0, std=1).T
        for _ in range(3)
    ]

    res_0 = _fused_all_gather_matmul_fallback(A_shard, Bs, 1, group.group_name)
    res_1 = torch.ops.symm_mem.fused_all_gather_matmul(A_shard, Bs, 1, group.group_name)


def test_matmul_reduce_scatter(group: dist.ProcessGroup):
    BATCH = 8
    M = 4096
    N = 8192
    K = 3584

    rank = group.rank()
    A = torch.rand((BATCH, M, K), dtype=torch.bfloat16, device="cuda").normal_()
    B = torch.rand((N, K), dtype=torch.bfloat16, device="cuda").normal_().T

    res_0 = _fused_matmul_reduce_scatter_fallback(A, B, "avg", 1, group.group_name)
    res_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        A, B, "avg", 1, group.group_name
    )


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(f"cuda:{local_rank}")
    torch.manual_seed(42 + rank)

    dist.init_process_group("nccl")
    enable_symm_mem_for_group("0")

    test_all_gather_matmul(dist.group.WORLD)
    test_matmul_reduce_scatter(dist.group.WORLD)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
