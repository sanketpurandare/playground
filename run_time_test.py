from contextlib import nullcontext
from os import write
from typing import Callable
import time
import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch._guards import active_fake_mode
from torch.utils.flop_counter import flop_registry
from fsdp_test import GPT, GPTConfig
from torch._inductor.utils import get_gpu_dram_gbps, get_device_tflops

aten = torch.ops.aten


class EstimateMode(TorchDispatchMode):

    def __init__(self):
        self.fake_mode: FakeTensorMode
        self._dispatch: Callable
        self.estimate_mode_type: str
        # No fall-back kernel needed for view ops
        self.ignore_ops = {
            aten.lift_fresh,
            aten.t,
            aten.transpose,
            aten.view,
            aten.detach,
            aten._unsafe_view,
            aten.split,
            aten.adjoint,
            aten.as_strided,
            aten.diagonal,
            aten.expand,
            aten.expand_as,
            aten.movedim,
            aten.permute,
            aten.select,
            aten.squeeze,
            aten.mT,
            aten.mH,
            aten.real,
            aten.imag,
            aten.view_as,
            aten.unflatten,
            aten.unfold,
            aten.unbind,
            aten.unsqueeze,
            aten.vsplit,
            aten.hsplit,
            aten.split_with_sizes,
            aten.swapaxes,
            aten.swapdims,
            aten.chunk,
        }
        self.ignore_ops_extended = {
            aten.randint,
            aten.randn,
            aten.rand,
            aten.randn_like,
            aten.rand_like,
            aten.randint_like,
            aten.arange,
            aten.ones_like,
            aten.zeros_like,
        }
        self.ignore_ops_extended.update(self.ignore_ops)
        self.gpu_memory_bandwidth = get_gpu_dram_gbps()
        self.float_types = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        self.no_fallback_kernel = set()
        self.total_time: float = 0.0

    # NB: returns fake tensors
    def _maybe_run_and_benchmark_fallback_kernel(
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
                    if e.dtype in self.float_types:
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
            cpu_start = time.time()
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                r = None
                r = func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            cpu_end = time.time()
            torch.cuda.synchronize()
            cpu_time = (cpu_end - cpu_start) / 1000
            total_op_time = start_event.elapsed_time(end_event) - cpu_time
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

    def _dispatch_benchmark_estimate(self, func, args, kwargs):
        if func._overloadpacket not in self.ignore_ops_extended:
            try:

                res, mean_op_time = (
                    self._maybe_run_and_benchmark_fallback_kernel(
                        func,
                        args,
                        kwargs,
                        NotImplementedError,
                    )
                )
                self.total_time += mean_op_time
                return res
            except NotImplementedError:
                self.no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return res

    def _dispatch_inductor_estimate(self, func, args, kwargs):
        def get_num_bytes(t: torch.Tensor) -> int:
            st = t.untyped_storage()
            num_bytes = st.size() * st.element_size()
            return num_bytes

        def get_compute_time(func_packet, args, kwargs, out, out_dtypes):
            if func_packet in flop_registry:
                assert (
                    len(out_dtypes) == 1
                ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
                dtype = out_dtypes.pop()
                factor = 1.0
                gpu_flops = get_device_tflops(dtype) * 1e12
                flop_count_func = flop_registry[func_packet]
                flop_count = flop_count_func(*args, **kwargs, out_val=out)
                compute_time = (factor * flop_count / gpu_flops) * 1e6
                return compute_time
            return 0.0

        def get_transfer_time(flat_args_kwargs, flat_outs):
            read_bytes = sum(
                get_num_bytes(t)
                for t in flat_args_kwargs
                if isinstance(t, torch.Tensor)
            )
            write_bytes = sum(
                get_num_bytes(t)
                for t in flat_outs
                if isinstance(t, torch.Tensor)
            )
            counted_bytes = read_bytes + write_bytes
            transfer_time = counted_bytes / self.gpu_memory_bandwidth
            return transfer_time

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet not in self.ignore_ops:

            flat_args_kwargs, args_spec = pytree.tree_flatten((args, kwargs))
            flat_outs, out_spec = pytree.tree_flatten(out)
            transfer_time = get_transfer_time(flat_args_kwargs, flat_outs)

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in self.float_types
            }

            args, kwargs = pytree.tree_unflatten(flat_args_kwargs, args_spec)
            out = pytree.tree_unflatten(flat_outs, out_spec)

            compute_time = get_compute_time(
                func_packet, args, kwargs, out, out_dtypes
            )
            op_time = compute_time/10**6
            # op_time = max(transfer_time, compute_time)/10**6
            self.total_time += op_time

        return out

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = self._dispatch(func, args, kwargs)
        return res

    def __call__(self, estimate_mode_type: str):
        if estimate_mode_type == "benchmark":
            self._dispatch = self._dispatch_benchmark_estimate
        elif estimate_mode_type == "inductor":
            self._dispatch = self._dispatch_inductor_estimate
        elif estimate_mode_type == "actual":
            return nullcontext()
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self.estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self):
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        self.fake_mode = fake_mode
        self.total_time = 0.0
        super().__enter__()
        return self

    def __exit__(self, *args):
        print(
            f"Estimated ({self.estimate_mode_type})"
            f"total_time: {self.total_time:.3f} ms"
        )
        if len(self.no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self.no_fallback_kernel))
        return super().__exit__(*args)


def test(
    estimate_mode: EstimateMode,
    estimate_mode_type: str = "actual",
):
    if estimate_mode_type == "actual":
        num_iters = 2
        maybe_fake_tensor_mode = nullcontext()
    else:
        num_iters = 1
        maybe_fake_tensor_mode = FakeTensorMode()

    with maybe_fake_tensor_mode:
        n_layer = 16
        vocab_size = 50304
        config = GPTConfig(
            block_size=512, n_layer=n_layer, vocab_size=vocab_size
        )
        with torch.device("cuda"):
            model = GPT(config)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(1)
        bsz, seq_len = 32, 512
        src = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        inp = (src, tgt)

        def inner(num_iters: int):
            for _ in range(num_iters):
                optim.zero_grad()
                loss = model(*inp)
                loss.backward()
                optim.step()

        inner(1)  # Initializing optimizer states
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with estimate_mode(estimate_mode_type=estimate_mode_type):
            start.record(torch.cuda.current_stream())
            inner(num_iters)
            end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

        iter_time = start.elapsed_time(end)

        if estimate_mode_type == "actual":
            print(f"Actual run_time : {iter_time/num_iters:.3f} ms")
        else:
            print(
                f"Estimation process total_time: {iter_time/num_iters:.3f} ms"
            )

        mem_stats = torch.cuda.memory_stats()
        peak_active_gb = mem_stats["active_bytes.all.peak"] / (1024**3)
        peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / (1024**3)
        print(
            f"peak active: {peak_active_gb} GB | peak reserved:"
            f" {peak_reserved_gb} GB"
        )


if __name__ == "__main__":

    estimate_mode = EstimateMode()
    test(estimate_mode, estimate_mode_type="inductor")
    test(estimate_mode, estimate_mode_type="actual")
    test(estimate_mode, estimate_mode_type="benchmark")
