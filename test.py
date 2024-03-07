from collections import defaultdict
from typing import Any
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only, tree_map
import torchvision
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref

MB = 2 ** 20
MEMORY_USE = WeakIdKeyDictionary()
parents = ['Global']
memory_tracking = defaultdict(lambda: defaultdict(int))



def track(t):    
    wt = weakref.ref(t)
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

def get_current_memory_allocated()->int:
    curr_use = 0
    for k, v in MEMORY_USE.items():
        curr_use += k.nelement() * k.element_size()
    return curr_use

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


def display_mem_stats():
    for mod in memory_tracking.keys():
        print(f"Module: ", mod)
        for k,v in memory_tracking[mod].items():
            print(f"{k}: {round(v/MB,2)} MBs")
        print()

def experiment():
    fake_mode = FakeTensorMode()
    mem_tracker = MemoryTrackingMode()
    torch.set_default_device("cuda")
    with fake_mode, mem_tracker:
        model = torchvision.models.resnet18()
        instrument_module(model)
        input = torch.randn(256, 3, 224, 224)
        output = model(input)
        output.sum().backward()
        final_call()
    display_mem_stats()

if __name__ == "__main__":
    experiment()

