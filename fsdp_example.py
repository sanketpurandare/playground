if __name__== "__main__":
    from contextlib import nullcontext
    from functools import partial
    import torch
    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )
    from torch.distributed._tensor import DeviceMesh
    from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
    )
    from torch.testing._internal.distributed.fake_pg import FakeStore
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    world_size = 4
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=0, world_size=world_size, store=store
    )
    mesh = DeviceMesh("cuda", torch.arange(0, world_size))
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    use_fake_mode = False
    with FakeTensorMode() if use_fake_mode else nullcontext():
        vocab_size = 8192
        bsz, seq_len = 32, 1024
        with torch.device(dev):
            model_args = ModelArgs(
                n_layers=2,
                n_heads=16,
                vocab_size=vocab_size,
                max_seq_len=seq_len,
                dropout_p=0.1,
            )
            model = Transformer(model_args)
        foreach = True
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        offload_policy = CPUOffloadPolicy(pin_memory=not use_fake_mode)
        reshard_after_forward = True
        fsdp_config = {

        }
        fully_shard_fn = partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                checkpoint(module, preserve_rng_state=not use_fake_mode)
                fully_shard_fn(module)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

        torch.manual_seed(42)
        inp = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        fmt = FSDPMemTracker(model, optim)
        fmt.track_inputs((inp,))
        with fmt:
            for iter_idx in range(2):
                loss = model(inp).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    fmt.reset_mod_stats()
    mem_stats = torch.cuda.memory_stats()
    tracker_peak = fmt.get_tracker_snapshot("peak")[dev]["Total"]
    cuda_peak_active = mem_stats["active_bytes.all.peak"]
    fmt.display_modulewise_snapshots(depth=4, units="MiB", tabulate=True)
    fmt.display_snapshot("peak", units="MiB", tabulate=True)
    print(
        f"peak active: {cuda_peak_active / (1024**3)} GiB | "
        f"Tracker Max: {tracker_peak / (1024 ** 3)} GiB"
    )
    if not use_fake_mode:
        print(f"Accuracy: {tracker_peak/cuda_peak_active}")

    try:
        torch.distributed.destroy_process_group()
    except Exception as e:
        print(e)
