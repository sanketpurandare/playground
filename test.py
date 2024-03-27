from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only, tree_map
import torchvision
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref
from enum import Enum, auto

MB = 2 ** 20
KB = 2 ** 10
MEMORY_USE = WeakIdKeyDictionary()
FIRST_OPT_ITER = True
parents = ['Global']
memory_tracking = defaultdict(lambda: defaultdict(lambda: defaultdict()))

class RefType(str, Enum):
    parameter = 'parameter'
    gradient = 'gradient'
    activation = 'activation'
    optstate = 'optstate'
@dataclass
class WeakRefInfo():
    def __init__(self, numel:int, element_size:int, reftype: RefType) -> None:
        self.numel = numel
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = self.numel * self.element_size

    def get_mem_consumed(self)->int:
        return self.mem_consumed

WINFO:Dict[weakref.ref, WeakRefInfo] = WeakIdKeyDictionary()

def track(t): 
    reftype = RefType.activation
    if isinstance(t, nn.Parameter):
        reftype = RefType.parameter  
    wt = weakref.ref(t)
    winfo = WeakRefInfo(t.nelement(), t.element_size(), reftype)
    WINFO[t] = winfo
    MEMORY_USE[t] = wt

class MemoryTrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):

        res = func(*args, **kwargs or {})
        # if isinstance(res, torch.Tensor):
        #     print("Function name: ", str(func.__name__))
        #     print("Result type: ", type(res))
        #     print("Result size", res.size())
        #     print("Result element size", res.element_size())
        #     print("Result device: ", res.device)
        tree_map_only(torch.Tensor, track, res)
        return res

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

def get_current_memory_allocated()->Dict[str, float]:
    global MEMORY_USE, WINFO
    mem_stats = defaultdict(float)
    mem_stats[RefType.parameter] = 0
    mem_stats[RefType.gradient] = 0
    mem_stats[RefType.optstate] = 0
    mem_stats[RefType.activation] = 0
    for k, v in MEMORY_USE.items():
        winfo = WINFO[k]
        mem = k.nelement() * k.element_size()
        assert(mem == winfo.get_mem_consumed()), "Failed assert"
        mem_stats[winfo.reftype] += winfo.get_mem_consumed()
    mem_stats['total'] = sum([m for m in mem_stats.values()])
    return mem_stats

def get_fqn()->str:
    fqn = '.'.join(parents)
    return fqn


def create_backwards_push(name):
    class PushState(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any) -> Any:
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args
        
        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            global parents
            parents.append(name)
            fqn = get_fqn()
            memory_tracking[fqn]["Before Backward"] = get_current_memory_allocated()
            return grad_outputs
    return PushState.apply

def create_backwards_pop(name):
    class PopState(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any) -> Any:
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args
        
        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            global parents
            fqn = get_fqn()
            memory_tracking[fqn]["After Backward"] = get_current_memory_allocated()
            assert(parents[-1] == name)
            parents.pop()
            return grad_outputs
    return PopState.apply

def enter_module_forward(name:str):
    def f (module:nn.Module, args:Any):
        global parents
        parents.append(name)
        fqn = get_fqn()
        memory_tracking[fqn]["Before Forward"] = get_current_memory_allocated()
        args = normalize_tuple(args)
        out = create_backwards_pop(name)(*args)
        return out
    return f

def exit_module_forward(name:str):
    def f (module:nn.Module, args: Any, outputs: Any):
        global parents
        assert(parents[-1] == name)
        fqn = get_fqn()
        memory_tracking[fqn]["After Forward"] = get_current_memory_allocated()
        parents.pop()
        outputs = normalize_tuple(outputs)
        return create_backwards_push(name)(*outputs)
    return f

def final_call():
        fqn = get_fqn()
        memory_tracking[fqn]["After Backward"] = get_current_memory_allocated()
        parents.pop()


def instrument_module(mod: nn.Module):
    for name, module in mod.named_children():
        module.register_forward_pre_hook(enter_module_forward(name))
        module.register_forward_hook(exit_module_forward(name))

def instrument_optimizer(optim:torch.optim.Optimizer):

    def outopt(optimizer:torch.optim.Optimizer, args:Any, kwargs:Any)->None:
        global FIRST_OPT_ITER, WINFO, MEMORY_USE
        print("This fired.")
        if FIRST_OPT_ITER:
            for param, states in optimizer.state.items():
                for val in states.values():
                    if isinstance(val, torch.Tensor):
                        winfo = WINFO[val]
                        assert winfo is not None, "None winfo object"
                        winfo.reftype = RefType.optstate
            FIRST_OPT_ITER = False            
    post_handle = optim.register_step_post_hook(outopt)

def display_mem_stats():
    for mod in memory_tracking.keys():
        print(f"Module: ", mod)
        for k,stats in memory_tracking[mod].items():
            print(f"{k}")
            for type, mem in stats.items():
                print(f"{type}: {round(mem/MB, 2)} MBs")
        print()

def experiment():
    fake_mode = FakeTensorMode()
    mem_tracker = MemoryTrackingMode()
    torch.set_default_device("cuda")
    with mem_tracker:
        print(torch.cuda.memory_allocated())
        model = torchvision.models.resnet18()
        print(torch.cuda.memory_allocated())
        optim = torch.optim.Adam(model.parameters(), fused=True)
        instrument_module(model)
        instrument_optimizer(optim)
        input = torch.randn(256, 3, 224, 224)
        print(torch.cuda.memory_allocated())
        output = model(input)
        print(torch.cuda.memory_allocated())
        output.sum().backward()
        print(torch.cuda.memory_allocated())
        optim.step()
        optim.zero_grad()
        final_call()
    print(torch.cuda.memory_allocated())
    display_mem_stats()

if __name__ == "__main__":
    experiment()

