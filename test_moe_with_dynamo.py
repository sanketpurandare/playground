# Run Command: TORCH_LOGS="aot_graphs" torchrun --standalone --nproc_per_node=4 <file_name>.py
import os
import time

import torch
import torch.nn.functional as F

from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile(fullgraph=True)
def test_moe_token_dispatch_and_combine(
    mesh: DeviceMesh,
    x: torch.Tensor,
    gate: torch.Tensor,
    up_proj_experts: torch.Tensor,
    down_proj_experts: torch.Tensor,
):
    b, s, d = x.shape
    x = x.reshape(b * s, d)
    num_ep_ranks = mesh.shape[0]
    num_experts = gate.shape[-1]
    experts_per_rank = num_experts // num_ep_ranks
    scores = F.softmax(x @ gate, dim=-1)
    topk_scores, selected_expert_indices = scores.topk(1, dim=-1)
    num_tokens_per_expert = torch.histc(
        selected_expert_indices.view(-1), bins=8, min=0, max=num_experts - 1
    )
    sorted_token_indices = torch.argsort(selected_expert_indices.view(-1), stable=True)
    sorted_x = x[sorted_token_indices, :]
    sorted_topk_scores = topk_scores.view(-1)[sorted_token_indices]
    input_offsets = num_tokens_per_expert
    output_offsets = all_to_all_single(
        input_offsets, None, None, group=mesh.get_group()
    )
    input_splits = torch.sum(
        input_offsets.reshape(num_ep_ranks, experts_per_rank), dim=1
    )
    output_splits = torch.sum(
        output_offsets.reshape(num_ep_ranks, experts_per_rank), dim=1
    )

    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()

    y = all_to_all_single_autograd(
        sorted_x * sorted_topk_scores.unsqueeze(-1),
        output_splits_list,
        input_splits_list,
        mesh.get_group(),
    )
    offsets = (
        output_offsets.reshape(num_ep_ranks, experts_per_rank).sum(dim=0).cumsum(dim=0)
    )
    y = torch._grouped_mm(
        y, up_proj_experts.transpose(-2, -1), offs=offsets.to(torch.int32)
    )
    y = F.silu(y)
    y = torch._grouped_mm(
        y,
        down_proj_experts.transpose(-2, -1),
        offs=offsets.to(torch.int32),
    )
    z = all_to_all_single_autograd(
        y, input_splits_list, output_splits_list, mesh.get_group()
    )
    z_unsorted = torch.zeros_like(z)
    z_unsorted[sorted_token_indices, :] = z
    z = z.reshape(x.shape)
    return z.sum()


if __name__ == "__main__":
    ep_size = int(os.environ["WORLD_SIZE"])
    gpu_id = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{gpu_id}"
    torch.distributed.init_process_group(backend="nccl", device_id=gpu_id)
    mesh = init_device_mesh("cuda", (ep_size,))
    dim = 128
    slen = 64
    bsz = 32
    num_experts = 8
    experts_per_rank = num_experts // ep_size
    gate = torch.rand(
        (dim, num_experts), device="cuda", requires_grad=True, dtype=torch.bfloat16
    )
    up_proj_experts = torch.rand(
        (experts_per_rank, dim * 4, dim),
        device="cuda",
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    down_proj_experts = torch.rand(
        (experts_per_rank, dim, dim * 4),
        device="cuda",
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    for _ in range(10):
        x = torch.rand(
            (bsz, slen, dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        z = test_moe_token_dispatch_and_combine(
            mesh, x, gate, up_proj_experts, down_proj_experts
        )
        z.backward()

    torch.distributed.barrier()
    time.sleep(2)

    if mesh:
        torch.distributed.destroy_process_group()
