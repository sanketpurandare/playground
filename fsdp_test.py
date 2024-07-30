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
from torch.distributed._tensor.api import DTensor
from torch.distributed._composable import checkpoint
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy
)
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from test_fake_pg import bypass_collectives
from torch.testing._internal.distributed.fake_pg import FakeStore
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
    offload_policy = CPUOffloadPolicy(pin_memory=False) if use_cpu_offload else None
    if use_activation_checkpoint and use_compile:
        apply_activation_checkpointing(
            model, auto_wrap_policy=ModuleWrapPolicy((Block,))
        )
    fully_shard_fn = functools.partial(
        fully_shard,
        mp_policy=mp_policy,
        mesh=mesh,
        offload_policy=offload_policy
    )
    for i, module in enumerate(model.transformer.h):
        if use_compile:
            module.forward = torch.compile(module.forward)
        if use_activation_checkpoint and not use_compile:
            # TODO: This does not work with compile! P872011846
            checkpoint(module, preserve_rng_state=False)
        fully_shard_fn(
            module, reshard_after_forward=(i < len(model.transformer.h) - 1)
        )
    model = fully_shard_fn(model)
    return model


vocab_size = 8192
n_layer = 2


def test_memory_tracking(
    use_activation_checkpoint: bool,
    use_cpu_offload: bool,
    use_compile: bool,
    mesh: DeviceMesh,
):
    IN_FAKE_MODE = True if active_fake_mode() else False
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

    model.to_empty(device="cpu" if use_cpu_offload else "cuda")
    if rank == 0:
        print(f"peak active after model init: {torch.cuda.memory_allocated()/1024**2} MB")

    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    torch.manual_seed(rank + 1)
    bsz, seq_len = 32, 2048
    src = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    tgt = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    inp = (src, tgt)

    def train_step():
        loss = model(*inp)
        loss.backward()
        optim.step()
        optim.zero_grad()

    torch.cuda.synchronize()
    # torch._C._cuda_clearCublasWorkspaces()
    if rank == 0:
        print(f"peak active after 1st iter: {torch.cuda.memory_allocated()/1024**2} MB")
    # import pickle
    # snapshot = torch.cuda.memory._snapshot()
    # with open(f"snapshot_{dist.get_rank()}.pickle", "wb") as f:
    #     pickle.dump(snapshot, f)
    memory_tracker = FSDPMemTracker(
        mod=model,
        optm=optim,
    )
    memory_tracker.track_inputs(inp)
    num_iters = 2
    with memory_tracker:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        for i in range(num_iters):
            train_step()
            if i < (num_iters - 1) and num_iters > 1:
                memory_tracker.reset_mod_stats()
        end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    iter_time = start.elapsed_time(end)
    if rank == 0:
        print(f"Time per iter: {iter_time/num_iters:.3f} ms")

    mem_stats = torch.cuda.memory_stats()
    peak_active = mem_stats["active_bytes.all.peak"]
    peak_reserved = mem_stats["reserved_bytes.all.peak"]
    num_retries = mem_stats["num_alloc_retries"]
    dev = torch.device(torch.cuda.current_device())
    tracker_peak = memory_tracker.get_tracker_snapshot("peak")[dev]["Total"]
    if rank == 0:
        memory_tracker.display_modulewise_snapshots(depth=4, units="MiB", tabulate=True)
        memory_tracker.display_snapshot("peak", units="MiB", tabulate=True)
        print(
            f"peak active: {peak_active / (1024**3)} GiB | Tracker Max: {tracker_peak / (1024 ** 3)} GiB | "
            f"Accuracy: {tracker_peak/peak_active} | "
            f"peak reserved: {peak_reserved / (1024**3)} GiB | num_retries: {num_retries}"
        )
        print(
            f"Tracker Max: {tracker_peak / (1024 ** 3)} GiB"
        )
    if not IN_FAKE_MODE:
        dist.barrier()


if __name__ == "__main__":
    try:
        dist.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dims = (world_size,)
        names = ("dp",)
        world_mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)
    except Exception:
        gpu_id = 0
        world_size = 4
        dims = (world_size,)
        names = ("dp",)
        store = FakeStore()
        dist.init_process_group(
            "fake", rank=gpu_id, world_size=world_size, store=store
        )
        world_mesh = DeviceMesh("cuda", torch.arange(0, world_size))

    if gpu_id == 0:
        print(f"world_size: {world_size}")
        print(f"world_mesh: {world_mesh}")
        print(f"peak active after cuda-mesh init: {torch.cuda.memory_allocated()/1024**2} MB")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    # TODO: Use argparse for the different args plus profiler / memory trace.
    # use_cpu_offload = True
    use_cpu_offload = True
    # use_activation_checkpoint = False
    use_activation_checkpoint = True
    # use_compile = True
    use_compile = False
    if use_compile:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = n_layer + 2
    with nullcontext():
    # with FakeTensorMode():
        test_memory_tracking(
            use_activation_checkpoint, use_cpu_offload, use_compile, world_mesh,
        )
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(e)
