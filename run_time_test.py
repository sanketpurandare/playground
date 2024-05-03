from contextlib import nullcontext
from typing import Union, cast

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from fsdp_test import GPT, GPTConfig

aten = torch.ops.aten


class MyDispatchMode(TorchDispatchMode):

    def __init__(self, fake_mode: Union[nullcontext, FakeTensorMode]):
        self.is_fake_mode = isinstance(fake_mode, FakeTensorMode)
        if self.is_fake_mode:
            self.fake_mode = cast(FakeTensorMode, fake_mode)
            # aten.lift_fresh
            # aten.t
            # aten.transpose
            # aten.view
            # aten.detach
            # aten._unsafe_view
            # aten.split
        self.ignore_ops = {
            aten.randint,
            aten.view,
            aten.randn,
            aten.rand,
            aten.randn_like,
            aten.rand_like,
            aten.randint_like,
            aten.detach,
            aten._unsafe_view,
            aten.arange,
            aten.ones_like,
            aten.zeros_like,
            aten.t,
            aten.transpose,
            aten.split,
            aten.lift_fresh,
        }
        self.no_fallback_kernel = set()
        self.total_time: float = 0.0

    # NB: returns fake tensors
    def maybe_run_and_benchmark_fallback_kernel(
        self,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be umodified
        if torch.Tag.inplace_view in func.tags:
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):
                if self.fake_mode.is_our_fake(e):
                    if e.dtype in (
                        torch.float32,
                        torch.float64,
                        torch.float16,
                        torch.bfloat16,
                    ):
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            num_iters = 2
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                r = None
                r = func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            total_op_time = start_event.elapsed_time(end_event)
            mean_op_time = total_op_time / num_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return (
                        self.fake_mode.fake_tensor_converter.from_real_tensor(
                            self.fake_mode, e
                        )
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        def _check(t: torch.Tensor):
            if not isinstance(t, FakeTensor):
                print(t.device, t.dtype, t.shape, t.requires_grad)
                # return self.fake_mode.from_tensor(t)
            return t

        if self.is_fake_mode and func._overloadpacket not in self.ignore_ops:
            try:

                res, mean_op_time = (
                    self.maybe_run_and_benchmark_fallback_kernel(
                        func,
                        args,
                        kwargs,
                        NotImplementedError,
                    )
                )
                self.total_time += mean_op_time

            except NotImplementedError:
                self.no_fallback_kernel.add(func._overloadpacket)
                # pytree.tree_map_only(torch.Tensor, _check, args)
                res = func(*args, **kwargs or {})
        else:
            res = func(*args, **kwargs or {})
        # pytree.tree_map_only(torch.Tensor, _check, res)
        return res

    def __enter__(self):
        self.total_time = 0.0
        super().__enter__()
        return self

    def __exit__(self, *args):
        print("no_fallback_kernel: ", list(self.no_fallback_kernel))
        return super().__exit__(*args)


if __name__ == "__main__":
    USE_FAKE_TENSOR_MODE = True
    fake_tensor_mode = (
        FakeTensorMode() if USE_FAKE_TENSOR_MODE else nullcontext()
    )
    my_dispatch_mode = MyDispatchMode(fake_tensor_mode)

    with fake_tensor_mode:
        n_layer = 12
        vocab_size = 50304
        config = GPTConfig(
            block_size=256, n_layer=n_layer, vocab_size=vocab_size
        )
        with torch.device("cuda"):
            model = GPT(config)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(1)
        bsz, seq_len = 32, 256
        src = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        inp = (src, tgt)

        def inner(num_iters: int):
            for _ in range(num_iters):
                optim.zero_grad()
                loss = model(*inp)
                loss.backward()
                optim.step()

        inner(1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        num_iters = 2

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with my_dispatch_mode:
            start.record(torch.cuda.current_stream())
            inner(num_iters)
            end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

        iter_time = start.elapsed_time(end)
        print(
            f"Dispatcher total_time: {my_dispatch_mode.total_time/num_iters:.3f} ms"
        )
        print(f"Actual total_time: {iter_time/num_iters:.3f} ms")

        mem_stats = torch.cuda.memory_stats()
        peak_active_gb = mem_stats["active_bytes.all.peak"] / (1024**3)
        peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / (1024**3)
        print(
            f"peak active: {peak_active_gb} GB | peak reserved:"
            f" {peak_reserved_gb} GB"
        )
