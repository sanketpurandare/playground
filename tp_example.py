import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import MemTracker
from torch.distributed._tools.fake_collectives import CollDistMode
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
from contextlib import nullcontext
import time


"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

from torch.testing._internal.distributed.fake_pg import FakeStore


dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
_world_size = 8
store = FakeStore()
torch.distributed.init_process_group(
    "fake", rank=0, world_size=_world_size, store=store
)

tp_size = 8

# understand world topology
_rank = 0

print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
import numpy as np

device_mesh = DeviceMesh("cuda", np.reshape(np.arange(_world_size), (dp_size, tp_size)), mesh_dim_names=("dp", "tp"))

tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
simple_llama2_config = ModelArgs(dim=2048, n_layers=16, n_heads=32, vocab_size=32000)
fake_mode = True
with FakeTensorMode() if fake_mode else nullcontext():

    with torch.device(dev):
        model = Transformer.from_model_args(simple_llama2_config)

    # init model weights
    model.init_weights()
    gib = (2 ** 30)
    print(f"Memory before parallel: {torch.cuda.memory_allocated()/gib:.2f}")
    # print(f"Model before parallelization {model=}\n")

    # parallelize the first embedding and the last linear out projection
    with CollDistMode():
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "norm": SequenceParallel(),
                "output": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate()
                ),
            }
        )

        for layer_id, transformer_block in enumerate(model.layers):
            layer_tp_plan = {
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            }

            # Adjust attention module to use the local number of heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

            # Custom parallelization plan for the model
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )

    # Init FSDP using the dp device mesh
    # sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
    sharded_model = model
    print(f"Memory after parallel: {torch.cuda.memory_allocated()/gib:.2f}")
    # print(f"Model after parallelization {sharded_model=}\n")

    # Create an optimizer for the parallelized and sharded model.
    lr = 3e-3
    print(f"Creating AdamW optimizer with learning rate {lr}")
    optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

    # Training loop:
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    print("\nStarting 2D training...")
    num_iterations = 2
    batch_size = 2
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    mem_tracker = MemTracker()
    mem_tracker.track_external(sharded_model, optimizer)
    for i in range(num_iterations):
        # seeding with dp_rank to ensure identical inputs for TP groups
        with mem_tracker:
            torch.manual_seed(i + dp_rank)
            inp = torch.randint(32000, (8, 512), device=dev)

            output = sharded_model(inp)
            output.sum().backward()
            optimizer.step()
        if i == 0:
            mem_tracker.reset_mod_stats()

    mem_tracker.display_snapshot("peak", units="GiB", tabulate=True)
    mem_stats = torch.cuda.memory_stats()
    peak_active = mem_stats["active_bytes.all.peak"]
    peak_reserved = mem_stats["reserved_bytes.all.peak"]

    print(
        f"peak active: {peak_active / gib:.2f} GiB | "
        f"peak reserved: {peak_reserved / gib:.2f} GiB"
    )