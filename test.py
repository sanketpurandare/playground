from collections import defaultdict
from typing import Any
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
import torchvision
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref

MEMORY_USE = WeakIdKeyDictionary()
parents = ['Global']
memory_tracking = defaultdict(lambda: defaultdict(int))
# def update_stats():
#     global parents
#     curr_use = 0
#     for k, v in MEMORY_USE.items():
#         curr_use += k.nelement() * k.element_size()
#     for par in parents:
#         memory_tracking[par]


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
        global parents
        curr_use = 0
        for k, v in MEMORY_USE.items():
            curr_use += k.nelement() * k.element_size()
        for par in parents:
            memory_tracking[par][func.__name__] = curr_use


        return res

def enter_module(name:str):
    def f (module:nn.Module, args:Any):
        global parents
        parents.append(name)
    return f

def exit_module(name:str):
    def f (module:nn.Module, args: Any, output: Any):
        global parents
        assert(parents[-1] == name)
        parents.pop()
    return f

def enter_module_backward(name:str):
    def f (module:nn.Module, grad_output: Any):
        global parents
        parents.append(name)
    return f

def exit_module_backward(name:str):
    def f (module:nn.Module, grad_input: Any, grad_output: Any):
        global parents
        assert(parents[-1] == name)
        parents.pop()
    return f



def instrument_module(mod: nn.Module):
    for name, module in mod.named_children():
        module.register_forward_pre_hook(enter_module(name))
        module.register_forward_hook(exit_module(name))
        # module.register_full_backward_pre_hook(enter_module_backward(name))
        # module.register_full_backward_hook(exit_module_backward(name))

def display_mem_stats():
    for mod in memory_tracking.keys():
        print(f"Module: ", mod)
        for k,v in memory_tracking[mod].items():
            print(k, v)
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
    display_mem_stats()

if __name__ == "__main__":
    experiment()

