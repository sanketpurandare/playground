import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._C._distributed_c10d import Work
from torch._guards import active_fake_mode
from torch.distributed._composable.fsdp import FSDP
from torch.distributed._composable.fsdp._fsdp_param_group import (
    FSDPCommContext, FSDPParamGroup)
from torch.distributed._tensor.api import DTensor
from torch.futures import Future
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdKeyDictionary

_PYTORCH_MIN_ALLOCATE = 2**9

__all__ = ["MemoryTrackingMode"]


class _RefType(str, Enum):
    sharded_parameter = "sharded_parameter"
    unsharded_parameter = "unsharded_parameter"
    buffer = "buffer"
    sharded_gradient = "sharded_gradient"
    unsharded_gradient = "unsharded_gradient"
    activation = "activation"
    all_gather = "all_gather"
    all_gather_copy_in = "all_gather_copy_in"
    reduce_scatter = "reduce_scatter"
    optstate = "optstate"
    inputs = "inputs"


@dataclass
class _WeakRefInfo:
    def __init__(
        self, size: int, element_size: int, reftype: _RefType
    ) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        return (
            math.ceil((self.size * self.element_size) / _PYTORCH_MIN_ALLOCATE)
            * _PYTORCH_MIN_ALLOCATE
        )

    def get_mem_consumed(self, st: torch.UntypedStorage) -> int:
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed


class _FSDPParamGroupSavedMethods(NamedTuple):
    pre_forward: Callable
    post_forward: Callable
    pre_backward: Callable
    post_backward: Callable


class _CollectiveSavedMethods(NamedTuple):
    all_gather_into_tensor: Callable
    reduce_scatter_tensor: Callable
    all_reduce: Callable
    barrier: Callable


class MemoryTrackingMode(TorchDispatchMode):
    def __init__(
        self,
        mod: Optional[torch.nn.Module] = None,
        optm: Optional[torch.optim.Optimizer] = None,
        inputs: Optional[Any] = None,
        units: str = "B",
        display_modulewise_stats: bool = True,
    ):
        self.mod = mod
        self.optm = optm
        self.inputs = inputs
        self.units = units
        self.display_modulewise_stats = display_modulewise_stats
        self.pre_forward_order: List[FSDPParamGroup] = []
        self.pre_forward_order_indices: List[int] = []
        self.memory_tracking: Dict[str, Dict[str, Dict[str, int]]] = (
            defaultdict(lambda: defaultdict(defaultdict))
        )
        self.parents: List[str] = []
        self._MEMORY_MAX: int = 0
        self.FIRST_OPT_ITER: bool = True
        self._RECORD_PRE_FORWARD_ORDER: bool = False
        self._IN_FAKE_MODE: bool = False
        self._fsdp_param_group_to_saved_methods: Dict[
            FSDPParamGroup, _FSDPParamGroupSavedMethods
        ] = {}
        self._collective_saved_methods: _CollectiveSavedMethods
        self._unsharded_param_to_grad_hook_handles: Dict[
            nn.Parameter, RemovableHandle
        ] = {}
        self._optimizer_hook_handle: Union[RemovableHandle, None] = None
        self.WINFO = WeakIdKeyDictionary()

    def _update_and_maybe_create_winfo(
        self, t: torch.Tensor, reftype: _RefType, existing: bool = False
    ) -> bool:
        if isinstance(t, DTensor):
            t = t._local_tensor
        st = t.untyped_storage()
        winfo = self.WINFO.get(st, None)
        if winfo is None:
            if existing:
                return False
            winfo = _WeakRefInfo(st.size(), st.element_size(), reftype)
            self.WINFO[st] = winfo
        else:
            winfo.reftype = reftype
        return True

    def _update_stats(self):
        curr_use: int = 0
        for st, winfo in self.WINFO.items():
            curr_use += winfo.get_mem_consumed(st)

        if self._MEMORY_MAX < curr_use:
            self._MEMORY_MAX = curr_use

    def _track(self, t: torch.Tensor):
        if isinstance(t, DTensor):
            t = t._local_tensor
        st = t.untyped_storage()
        if self.WINFO.get(st, None) is not None:
            return
        winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.activation)
        self.WINFO[st] = winfo
        self._update_stats()

    def _get_current_memory_allocated(self) -> Dict[str, int]:
        mem_stats = defaultdict(int)
        for reftype in _RefType:
            mem_stats[reftype.name] = 0
        for st, winfo in self.WINFO.items():
            mem_stats[winfo.reftype.name] += winfo.get_mem_consumed(st)
        mem_stats["TRACKER_total"] = sum([m for m in mem_stats.values()])
        if torch.cuda.is_available():
            mem_stats["CUDA_total"] = torch.cuda.memory_stats()[
                "active_bytes.all.current"
            ]
        return mem_stats

    def print_mem_stats(self, stats: Optional[Dict[str, int]] = None):
        if stats is None:
            stats = self._get_current_memory_allocated()

        def rounding_fn(value, divisor, precision) -> Union[float, int]:
            if divisor == 1:
                return value
            return round(value / divisor, precision)

        divisor = 1
        if self.units == "GB":
            divisor = 2**30
        elif self.units == "MB":
            divisor = 2**20
        elif self.units == "KB":
            divisor = 2**10

        for mem_type, mem_val in stats.items():
            print(
                f"\t{mem_type}:"
                f" {rounding_fn(mem_val, divisor, 2)} {self.units}"
            )

    def get_max_memory(self) -> int:
        return self._MEMORY_MAX

    def _display_mem_stats(self):
        for mod in self.memory_tracking.keys():
            print(f"Module:  {mod}")
            for state, stats in self.memory_tracking[mod].items():
                print(f"{state}")
                self.print_mem_stats(stats)
            print()

    def _enter_module(self, name: str, state: str):
        self.parents.append(name)
        self.memory_tracking[name][
            state
        ] = self._get_current_memory_allocated()

    def _exit_module(self, name: str, state: str):
        assert self.parents[-1] == name, f"{self.parents[-1]} is not {name}"
        self.memory_tracking[name][
            state
        ] = self._get_current_memory_allocated()
        self.parents.pop()

    def _instrument_fsdp_param_group(self, fsdp_param_group: FSDPParamGroup):
        def _unsharded_grad_hook(param: nn.Parameter):
            if param.grad is not None:
                assert self._update_and_maybe_create_winfo(
                    param.grad, _RefType.unsharded_gradient, existing=True
                ), "grad tensor not found in WINFO"

        for fsdp_param in fsdp_param_group.fsdp_params:
            assert isinstance(fsdp_param.sharded_param, nn.Parameter), (
                f"{fsdp_param._module_info.param_name} sharded param is not a"
                " nn.Parameter"
            )
            self._update_and_maybe_create_winfo(
                fsdp_param.sharded_param, _RefType.sharded_parameter
            )

            assert isinstance(fsdp_param._unsharded_param, nn.Parameter), (
                f"{fsdp_param._module_info.param_name} unsharded param is not"
                " a nn.Parameter"
            )
            self._update_and_maybe_create_winfo(
                fsdp_param._unsharded_param, _RefType.unsharded_parameter
            )
            self._unsharded_param_to_grad_hook_handles[
                fsdp_param._unsharded_param
            ] = fsdp_param._unsharded_param.register_post_accumulate_grad_hook(
                _unsharded_grad_hook
            )

    def record_pre_forward_order(self, fsdp_param_group: FSDPParamGroup):
        self.pre_forward_order.append(fsdp_param_group)
        self.pre_forward_order_indices.append(len(self.pre_forward_order) - 1)

    def _fsdp_param_group_pre_forward(
        self,
        orig_fsdp_param_group_pre_forward: Callable,
        fsdp_param_group: FSDPParamGroup,
        name: str,
    ) -> Callable:
        @wraps(orig_fsdp_param_group_pre_forward)
        def inner(*args, **kwargs):
            self._enter_module(name, "Before Pre-Forward")
            args, kwargs = orig_fsdp_param_group_pre_forward(*args, **kwargs)
            self._exit_module(name, "After Pre-Forward")
            return args, kwargs

        return inner

    def _fsdp_param_group_post_forward(
        self,
        orig_fsdp_param_group_post_forward: Callable,
        fsdp_param_group: FSDPParamGroup,
        name: str,
    ) -> Callable:
        @wraps(orig_fsdp_param_group_post_forward)
        def inner(*args, **kwargs):
            self._enter_module(name, "Before Post-Forward")
            output = orig_fsdp_param_group_post_forward(*args, **kwargs)
            self._exit_module(name, "After Post-Forward")
            return output

        return inner

    def _fsdp_param_group_pre_backward(
        self,
        orig_fsdp_param_group_pre_backward: Callable,
        fsdp_param_group: FSDPParamGroup,
        name: str,
    ) -> Callable:
        @wraps(orig_fsdp_param_group_pre_backward)
        def inner(*args, **kwargs):
            self._enter_module(name, "Before Pre-Backward")
            ret_val = orig_fsdp_param_group_pre_backward(*args, **kwargs)
            self._exit_module(name, "After Pre-Backward")
            return ret_val

        return inner

    def _fsdp_param_group_post_backward(
        self,
        orig_fsdp_param_group_post_backward: Callable,
        fsdp_param_group: FSDPParamGroup,
        name: str,
    ) -> Callable:
        @wraps(orig_fsdp_param_group_post_backward)
        def inner(*args, **kwargs):
            self._enter_module(name, "Before Post-Backward")
            ret_val = orig_fsdp_param_group_post_backward(*args, **kwargs)
            for fsdp_param in fsdp_param_group.fsdp_params:
                sharded_grad = fsdp_param.sharded_param.grad
                if sharded_grad is not None:
                    assert self._update_and_maybe_create_winfo(
                        sharded_grad, _RefType.sharded_gradient
                    ), "sharded grad failed"
            self._exit_module(name, "After Post-Backward")
            return ret_val

        return inner

    def _instrument_fsdp_module(self, mod: nn.Module):
        self.root_module = mod
        prefix = type(mod).__name__
        for name, module in mod.named_modules():
            if isinstance(module, FSDP):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    local_prefix = type(module).__name__
                    if name == "":
                        name = prefix
                    else:
                        local_prefix = type(module).__name__
                        name = ".".join([prefix, local_prefix + "_" + name])

                    self._instrument_fsdp_param_group(fsdp_param_group)
                    self._fsdp_param_group_to_saved_methods[
                        fsdp_param_group
                    ] = _FSDPParamGroupSavedMethods(
                        fsdp_param_group.pre_forward,
                        fsdp_param_group.post_forward,
                        fsdp_param_group.pre_backward,
                        fsdp_param_group.post_backward,
                    )
                    fsdp_param_group.pre_forward = (
                        self._fsdp_param_group_pre_forward(
                            fsdp_param_group.pre_forward,
                            fsdp_param_group,
                            name,
                        )
                    )
                    fsdp_param_group.post_forward = (
                        self._fsdp_param_group_post_forward(
                            fsdp_param_group.post_forward,
                            fsdp_param_group,
                            name,
                        )
                    )
                    fsdp_param_group.pre_backward = (
                        self._fsdp_param_group_pre_backward(
                            fsdp_param_group.pre_backward,
                            fsdp_param_group,
                            name,
                        )
                    )
                    fsdp_param_group.post_backward = (
                        self._fsdp_param_group_post_backward(
                            fsdp_param_group.post_backward,
                            fsdp_param_group,
                            name,
                        )
                    )
        for buffer in mod.buffers():
            self._update_and_maybe_create_winfo(buffer, _RefType.buffer)

    def _instrument_optimizer(self, optim: torch.optim.Optimizer):
        def _opt_state(
            optimizer: torch.optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            if self.FIRST_OPT_ITER:
                for states in optimizer.state.values():
                    for val in states.values():
                        if isinstance(val, torch.Tensor):
                            self._update_and_maybe_create_winfo(
                                val, _RefType.optstate
                            )
                self.FIRST_OPT_ITER = False

        _opt_state(optim, None, None)
        self.FIRST_OPT_ITER = True
        opt_hook_handle = optim.register_step_post_hook(_opt_state)
        self._optimizer_hook_handle = opt_hook_handle

    def _register_module_and_optimizer_hooks(self):
        if self.mod is not None:
            self._instrument_fsdp_module(self.mod)
        if self.optm is not None:
            self._instrument_optimizer(self.optm)

    def _deregister_module_and_optimizer_hooks(self):
        for (
            fsdp_param_group,
            saved_methods,
        ) in self._fsdp_param_group_to_saved_methods.items():
            fsdp_param_group.pre_forward = saved_methods.pre_forward
            fsdp_param_group.post_forward = saved_methods.post_forward
            fsdp_param_group.pre_backward = saved_methods.pre_backward
            fsdp_param_group.post_backward = saved_methods.post_backward

        for (
            unsharded_grad_hook_handle
        ) in self._unsharded_param_to_grad_hook_handles.values():
            unsharded_grad_hook_handle.remove()

        if self._optimizer_hook_handle is not None:
            self._optimizer_hook_handle.remove()

    def _instrument_and_maybe_bypass_collectives(self):
        self._collective_saved_methods = _CollectiveSavedMethods(
            dist.all_gather_into_tensor,
            dist.reduce_scatter_tensor,
            dist.all_reduce,
            dist.barrier,
        )

        class FakeWork(Work):

            def __init__(self):
                super().__init__()

            def get_future(self) -> Future:
                future = Future()
                future.set_result(None)
                return future

            def wait(self, timeout: timedelta = ...) -> bool:
                return True

        @wraps(dist.all_gather_into_tensor)
        def all_gather_into_tensor(
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            group=None,
            async_op=False,
        ):
            assert self._update_and_maybe_create_winfo(
                input_tensor, _RefType.all_gather_copy_in, existing=True
            ), "all_gather_in_tensor not found in WINFO"
            assert self._update_and_maybe_create_winfo(
                output_tensor, _RefType.all_gather, existing=True
            ), "all_gather_out_tensor not found in WINFO"

            if self._IN_FAKE_MODE:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._collective_saved_methods.all_gather_into_tensor(
                    output_tensor, input_tensor, group, async_op
                )

        @wraps(dist.reduce_scatter_tensor)
        def reduce_scatter_tensor(
            output: torch.Tensor,
            input: torch.Tensor,
            op: dist.ReduceOp = dist.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            assert self._update_and_maybe_create_winfo(
                input, _RefType.reduce_scatter, existing=True
            ), "reduce_scatter_in_tensor not found in WINFO"

            if self._IN_FAKE_MODE:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._collective_saved_methods.reduce_scatter_tensor(
                    output, input, op, group, async_op
                )

        @wraps(dist.all_reduce)
        def all_reduce(
            tensor: torch.Tensor,
            op: dist.ReduceOp = dist.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            if self._IN_FAKE_MODE:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._collective_saved_methods.all_reduce(
                    tensor, op, group, async_op
                )

        @wraps(dist.barrier)
        def barrier():
            if self._IN_FAKE_MODE:
                return None
            else:
                return self._collective_saved_methods.barrier()

        dist.all_gather_into_tensor = all_gather_into_tensor
        dist.reduce_scatter_tensor = reduce_scatter_tensor
        dist.all_reduce = all_reduce
        dist.barrier = barrier

    def _restore_collectives(self):
        dist.all_gather_into_tensor = (
            self._collective_saved_methods.all_gather_into_tensor
        )
        dist.reduce_scatter_tensor = (
            self._collective_saved_methods.reduce_scatter_tensor
        )
        dist.all_reduce = self._collective_saved_methods.all_reduce
        dist.barrier = self._collective_saved_methods.barrier

    def _mark_inputs(self):
        if self.inputs is not None:

            def _track_inputs(t: torch.Tensor):
                self._update_and_maybe_create_winfo(t, _RefType.inputs)

            tree_map_only(torch.Tensor, _track_inputs, self.inputs)

    def __enter__(self):
        self._register_module_and_optimizer_hooks()
        self._mark_inputs()
        self._IN_FAKE_MODE = True if active_fake_mode() else False
        self._instrument_and_maybe_bypass_collectives()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display_modulewise_stats:
            self._display_mem_stats()
        self._deregister_module_and_optimizer_hooks()
        self._restore_collectives()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = func(*args, **kwargs or {})
        tree_map_only(torch.Tensor, self._track, res)
        return res


def test():
    class DummyModel(nn.Module):
        def __init__(self, layers: int, dim: int):
            super(DummyModel, self).__init__()
            self._module_list = []
            for _ in range(layers):
                self._module_list.extend([nn.Linear(dim, dim), nn.ReLU()])
            self.module = nn.Sequential(*self._module_list)

        def forward(self, x):
            return self.module(x)

    batch_size = 100
    layers = 5
    dim = 10000
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.cuda.reset_peak_memory_stats()

    model = DummyModel(layers, dim)
    optim = torch.optim.Adam(model.parameters(), fused=True)
    mem_tracker = MemoryTrackingMode(
        model, optim, display_modulewise_stats=True
    )
    with mem_tracker as mt:
        input_batch = torch.randn(batch_size, dim)
        print("After Model and mini-batch init:")
        mt.print_mem_stats()
        output = model(input_batch)
        print("After Forward:")
        mt.print_mem_stats()
        output.sum().backward()
        output = None
        print("After Backward:")
        mt.print_mem_stats()
        optim.step()
        print("After Opt Step:")
        mt.print_mem_stats()
        optim.zero_grad()
        print("After Zero Grad:")
        mt.print_mem_stats()
        MAX_MEMORY = mt.get_max_memory()

    print(f"Tracker measured: {MAX_MEMORY}")
    if torch.cuda.is_available():
        CUDA_MEMORY_MAX = torch.cuda.max_memory_allocated()
        print(f"Cuda measured: {CUDA_MEMORY_MAX}")
        print(f"Peak comparison ratio: {MAX_MEMORY/CUDA_MEMORY_MAX}")


if __name__ == "__main__":
    test()
