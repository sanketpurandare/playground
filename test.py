import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
import torchvision
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref

MEMORY_USE = WeakIdKeyDictionary()

def print_stats():
    for k, v in MEMORY_USE.items():
        print(k , v)

def track(t):
    def callback(_):
        print_stats()
    
    wt = weakref.ref(t, callback)
    MEMORY_USE[t] = wt
    print_stats()

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



def instrument_module(mod: nn.Module):
    for name, module in mod.named_children():
        module.register_forward_pre_hook(enter_module(name))
        module.register_forward_hook(exit_module(name))

def experiment():
    fake_mode = FakeTensorMode()
    mem_tracker = MemoryTrackingMode()
    torch.set_default_device("cuda")
    with fake_mode, mem_tracker:
        model = torchvision.models.resnet18()
        input = torch.randn(256, 3, 224, 224)
        output = model(input)
        output.sum().backward()

if __name__ == "__main__":
    experiment()

