"""
torchrun --standalone --nproc_per_node=4 fsdp2_test.py
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 fsdp2_test.py
"""

import functools
import os
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._composable import checkpoint
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
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# NOTE: We take the GPT2 implementation from nanoGPT: https://github.com/karpathy/nanoGPT


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        (B, T, C) = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTMLP(nn.Module):  # renamed to avoid name conflict
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPTMLP(config)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    checkpoint_activations: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        wte = nn.Embedding(config.vocab_size, config.n_embd)
        wpe = nn.Embedding(config.block_size, config.n_embd)
        torch.nn.init.normal_(wte.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(wpe.weight, mean=0.0, std=0.02)
        blocks: List[Block] = []
        for _ in range(config.n_layer):
            block = Block(config)
            blocks.append(block)
        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wpe=wpe,
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(blocks),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Supports at most {self.config.block_size} but got {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            if self.config.checkpoint_activations:
                # We only support composition with non-reentrant AC
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return loss



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
    reduce_dtype = torch.float16
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    offload_policy = CPUOffloadPolicy(pin_memory=True) if use_cpu_offload else None
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


vocab_size = 50304
n_layer = 4


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

    model.to_empty(device="cpu")
    if rank == 0:
        print(f"peak active after model init: {torch.cuda.memory_allocated()/1024**2} MB")

    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    torch.manual_seed(rank + 1)
    bsz, seq_len = 32, 1024
    src = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    tgt = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    inp = (src, tgt)

    def train_step():
        loss = model(*inp)
        loss.backward()
        optim.step()
        optim.zero_grad()

    # train_step()
    torch.cuda.synchronize()
    # torch._C._cuda_clearCublasWorkspaces()
    if rank == 0:
        print(f"peak active after 1st iter: {torch.cuda.memory_allocated()/1024**2} MB")
    # import pickle
    # snapshot = torch.cuda.memory._snapshot()
    # with open(f"snapshot_{dist.get_rank()}.pickle", "wb") as f:
    #     pickle.dump(snapshot, f)

    num_iters = 2

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream())
    for i in range(num_iters):
        train_step()
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
    use_cpu_offload = False
    # use_activation_checkpoint = False
    use_activation_checkpoint = False
    # use_compile = True
    use_compile = False
    if use_compile:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = n_layer + 2
    with FakeTensorMode():
        test_memory_tracking(
            use_activation_checkpoint, use_cpu_offload, use_compile, world_mesh,
        )
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(e)
