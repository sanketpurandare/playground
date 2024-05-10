"""
torchrun --standalone --nproc_per_node=4 fsdp_test.py
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 fsdp_test.py
"""

import functools
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.distributed._tools.fsdp2_memory_tracker import FSDPMemTracker
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from test_model import GPT, GPTConfig, Block


def meta_init_model(config: GPTConfig) -> nn.Module:
    torch.manual_seed(42)
    with torch.device("meta"):
        model = GPT(config)
    return model


def apply_fsdp_wrapping(
    model: nn.Module,
    use_activation_checkpoint: bool,
    use_cpu_offload: bool,
    use_compile: bool,
    mesh: DeviceMesh
):
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # offload_policy = OffloadPolicy("cpu" if use_cpu_offload else None)
    if use_activation_checkpoint and use_compile:
        apply_activation_checkpointing(
            model, auto_wrap_policy=ModuleWrapPolicy((Block,))
        )
    fully_shard_fn = functools.partial(
        fully_shard,
        mp_policy=mp_policy,
        mesh=mesh,
        # offload_policy=offload_policy
    )
    for i, module in enumerate(model.transformer.h):
        if use_compile:
            module.forward = torch.compile(module.forward)
        if use_activation_checkpoint and not use_compile:
            # TODO: This does not work with compile! P872011846
            checkpoint(module)
        fully_shard_fn(
            module, reshard_after_forward=(i < len(model.transformer.h) - 1)
        )
    model = fully_shard_fn(model)
    return model


vocab_size = 50304
n_layer = 4


def test_memory_tracking(
    use_activation_checkpoint: bool,
    use_cpu_offload: bool,
    use_compile: bool,
    mesh: DeviceMesh,
):

    try:
        rank = dist.get_rank()
    except:
        rank = 0
    # torch.cuda.memory._record_memory_history()
    config = GPTConfig(block_size=2048, n_layer=n_layer, vocab_size=vocab_size)

    model = meta_init_model(config)
    if rank == 0:
        print(f"peak active before model init: {torch.cuda.memory_allocated()/1024**2} MB")
    model = apply_fsdp_wrapping(
        model, use_activation_checkpoint, use_cpu_offload, use_compile, mesh
    )
    model.to_empty(device="cuda")
    if rank == 0:
        print(f"peak active after model init: {torch.cuda.memory_allocated()/1024**2} MB")

    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    torch.manual_seed(rank + 1)
    bsz, seq_len = 32, 1024
    src = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    tgt = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    inp = (src, tgt)

    def inner(num_iters: int):
        for _ in range(num_iters):
            optim.zero_grad()
            loss = model(*inp)
            loss.backward()
            optim.step()

    torch.cuda.synchronize()
    # torch._C._cuda_clearCublasWorkspaces()
    if rank == 0:
        print(f"peak active after 1st iter: {torch.cuda.memory_allocated()/1024**2} MB")
    # import pickle
    # snapshot = torch.cuda.memory._snapshot()
    # with open(f"snapshot_{dist.get_rank()}.pickle", "wb") as f:
    #     pickle.dump(snapshot, f)

    display_modulewise_stats = True if rank == 0 else False
    memory_tracker = FSDPMemTracker(
        mod=model,
        optm=optim,
        inputs=inp,
        display_modulewise_stats=display_modulewise_stats,
        units="MB",
        display_peak_stats=False
    )
    num_iters = 1
    with memory_tracker:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        inner(num_iters)
        end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    iter_time = start.elapsed_time(end)
    if rank == 0:
        print(f"Time per iter: {iter_time/num_iters:.3f} ms")

    mem_stats = torch.cuda.memory_stats()
    peak_active_gb = mem_stats["active_bytes.all.peak"] / (1024**3)
    peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / (1024**3)
    num_retries = mem_stats["num_alloc_retries"]
    if rank == 0:
        print(
            f"peak active: {peak_active_gb} GB | peak reserved:"
            f" {peak_reserved_gb} GB | num_retries: {num_retries}"
        )
        print(
            f"Tracker Max: {memory_tracker.get_peak_memory() / (1024 ** 3)} GB"
        )
    dist.barrier()


if __name__ == "__main__":
    try:
        dist.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dims = (world_size,)
        names = ("dp",)
        world_mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)
    except:
        world_mesh = DeviceMesh("cuda", [1])
        gpu_id = 0
        world_size = 1
    if gpu_id == 0:
        print(f"world_size: {world_size}")
        print(f"world_mesh: {world_mesh}")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    # TODO: Use argparse for the different args plus profiler / memory trace.
    # use_cpu_offload = True
    use_cpu_offload = False
    # use_activation_checkpoint = False
    use_activation_checkpoint = True
    # use_compile = True
    use_compile = False
    if use_compile:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = n_layer + 2
    test_memory_tracking(
        use_activation_checkpoint, use_cpu_offload, use_compile, world_mesh,
    )
    try:
        dist.destroy_process_group()
    except:
        pass
