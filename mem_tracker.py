import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Union,
    Tuple,
)
from functools import partial, wraps
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdKeyDictionary
from torch.distributed._composable.fsdp._fsdp_param_group import (
    FSDPCommContext,
    FSDPParamGroup,
)
from torch.distributed._composable.fsdp import FSDP

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


class MemoryTrackingMode(TorchDispatchMode):
    def __init__(
        self,
        mod: Optional[torch.nn.Module] = None,
        optm: Optional[torch.optim.Optimizer] = None,
        inputs: Optional[Any] = None,
        depth: int = 2,
        units: str = "B",
        display_modulewise_stats: bool = True,
    ):
        self.mod = mod
        self.optm = optm
        self.inputs = inputs
        self.depth = depth
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
        self._fsdp_param_group_to_saved_methods: Dict[
            FSDPParamGroup, _FSDPParamGroupSavedMethods
        ] = {}
        self._sharded_param_to_grad_hook_handles: Dict[
            nn.Parameter, RemovableHandle
        ] = {}
        self._unsharded_param_to_grad_hook_handles: Dict[
            nn.Parameter, RemovableHandle
        ] = {}
        self._optimizer_hook_handle: Union[RemovableHandle, None] = None
        self.WINFO = WeakIdKeyDictionary()

    def _update_stats(self):
        curr_use: int = 0
        for st, winfo in self.WINFO.items():
            curr_use += winfo.get_mem_consumed(st)

        if self._MEMORY_MAX < curr_use:
            self._MEMORY_MAX = curr_use

    def _track(self, t: torch.Tensor):
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
            mem_stats["CUDA_total"] = torch.cuda.memory_allocated()
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

    def _display_mem_stats(self, depth=None):
        if depth is None:
            depth = self.depth
        for mod in self.memory_tracking.keys():
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue
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
        def _grad_hook(reftype: _RefType, param: nn.Parameter):
            if param.grad is not None:
                st = param.grad.untyped_storage()
                winfo = self.WINFO.get(st, None)
                assert winfo is not None, "grad tensor not found in WINFO"
                winfo.reftype = reftype

        _sharded_grad_hook = partial(_grad_hook, _RefType.sharded_gradient)
        _unsharded_grad_hook = partial(_grad_hook, _RefType.unsharded_gradient)

        for fsdp_param in fsdp_param_group.fsdp_params:
            assert isinstance(fsdp_param.sharded_param, nn.Parameter), (
                f"{fsdp_param._module_info.param_name} sharded param is not a"
                " nn.Parameter"
            )
            st = fsdp_param.sharded_param.untyped_storage()
            winfo = self.WINFO.get(st, None)
            if winfo is None:
                winfo = _WeakRefInfo(
                    st.size(), st.element_size(), _RefType.sharded_parameter
                )
                self.WINFO[st] = winfo
            self._sharded_param_to_grad_hook_handles[
                fsdp_param.sharded_param
            ] = fsdp_param.sharded_param.register_post_accumulate_grad_hook(
                _sharded_grad_hook
            )

            assert isinstance(fsdp_param._unsharded_param, nn.Parameter), (
                f"{fsdp_param._module_info.param_name} unsharded param is not"
                " a nn.Parameter"
            )
            st = fsdp_param._unsharded_param.untyped_storage()
            winfo = self.WINFO.get(st, None)
            if winfo is None:
                winfo = _WeakRefInfo(
                    st.size(), st.element_size(), _RefType.unsharded_parameter
                )
                self.WINFO[st] = winfo
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
            if (
                all_gather_result := fsdp_param_group._all_gather_result
            ) is not None:
                assert isinstance(
                    all_gather_result, torch.Tensor
                ), "Expected all gather result to be a tensor"
                winfo = self.WINFO[all_gather_result.untyped_storage()]
                assert (
                    winfo is not None
                ), "all gather tensor not found in WINFO"
                winfo.reftype = _RefType.all_gather
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
                    if name == "":
                        name = prefix
                    else:
                        name = ".".join([prefix, name])

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
            st = buffer.untyped_storage()
            winfo = self.WINFO.get(st, None)
            if winfo is None:
                winfo = _WeakRefInfo(
                    st.size(), st.element_size(), _RefType.buffer
                )
                self.WINFO[st] = winfo

    def _instrument_optimizer(self, optim: torch.optim.Optimizer):
        def _opt_state(
            optimizer: torch.optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            if self.FIRST_OPT_ITER:
                for states in optimizer.state.values():
                    for val in states.values():
                        if isinstance(val, torch.Tensor):
                            st = val.untyped_storage()
                            winfo = self.WINFO.get(st, None)
                            if winfo is None:
                                winfo = _WeakRefInfo(
                                    st.size(),
                                    st.element_size(),
                                    _RefType.optstate,
                                )
                                self.WINFO[st] = winfo
                            else:
                                winfo.reftype = _RefType.optstate
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
            sharded_grad_hook_handle
        ) in self._sharded_param_to_grad_hook_handles.values():
            sharded_grad_hook_handle.remove()

        for (
            unsharded_grad_hook_handle
        ) in self._unsharded_param_to_grad_hook_handles.values():
            unsharded_grad_hook_handle.remove()

        if self._optimizer_hook_handle is not None:
            self._optimizer_hook_handle.remove()

    def _mark_inputs(self):
        if self.inputs is not None:

            def _track_inputs(t: torch.Tensor):
                st = t.untyped_storage()
                winfo = self.WINFO.get(st, None)
                if winfo is None:
                    winfo = _WeakRefInfo(
                        st.size(), st.element_size(), _RefType.inputs
                    )
                    self.WINFO[st] = winfo

            tree_map_only(torch.Tensor, _track_inputs, self.inputs)

    def __enter__(self):
        self._register_module_and_optimizer_hooks()
        self._mark_inputs()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display_modulewise_stats:
            self._display_mem_stats()
        self._deregister_module_and_optimizer_hooks()
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
