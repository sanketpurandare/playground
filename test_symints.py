import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

if __name__ == "__main__":
    world_size = 4
    rank = 0
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    store = FakeStore()
    dist.init_process_group("fake", rank=rank, world_size=world_size, store=store)
    mesh = init_device_mesh("cuda", (world_size,))
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    with fake_mode:
        dim = 128
        slen = 64
        bsz = 32
        num_experts = 8
        ep_size = mesh.shape[0]
        experts_per_rank = num_experts // ep_size
        x = torch.rand((bsz, slen, dim), device="cuda")
        gate = torch.rand((dim, num_experts), device="cuda")
        scores = F.softmax(x @ gate, dim=-1)
        top_scores, selected_expert_indices = scores.topk(1, dim=-1)
        num_tokens_per_expert = torch.histc(
            selected_expert_indices.view(-1), bins=8, min=0, max=7
        )
        lis = num_tokens_per_expert.tolist()
        new_t = torch._refs.tensor(lis, device="cuda")
        print(lis)
        alignment = 8
        total_tokens_per_expert = num_tokens_per_expert.view(
            ep_size, experts_per_rank
        ).sum(0)
        # pad out empty experts to alignment requirement
        total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

        # align the chunk sizes (cdiv)
        m_sizes = (
            (total_tokens_per_expert + alignment - 1) // alignment * alignment
        ).to(torch.int32)
        input_offsets = num_tokens_per_expert
        output_offsets = all_to_all_single(
            input_offsets, None, None, group=mesh.get_group()
        )
        input_offsets = input_offsets.reshape(ep_size, experts_per_rank).sum(dim=1)
        output_offsets = output_offsets.reshape(ep_size, experts_per_rank).sum(dim=1)
        input_splits = input_offsets.tolist()
        output_splits = output_offsets.tolist()

        torch._check(input_splits[0] == output_splits[0])
        torch._check(sum(input_splits) == (bsz * slen))
        print(f"Input Splits: {input_splits} Output Splits: {output_splits}")
        y = all_to_all_single_autograd(
            x.reshape((bsz * slen), dim), output_splits, input_splits, mesh.get_group()
        )
        print(f"Token Dispatch: {y.shape}")
        z = all_to_all_single_autograd(y, input_splits, output_splits, mesh.get_group())
        print(f"Token Combine: {z.shape}")
        torch._check(z.shape[0] == (bsz * slen))
        z = z.reshape(x.shape)
        print(f"Enforcing Static Shape after Token Combine {z.shape}")
