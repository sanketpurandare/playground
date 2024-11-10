import contextlib
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.utils._pytree as pytree
from torch.utils.module_tracker import ModuleTracker
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class TestMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), args)
        print(func)    
        return out
    
class MLP(torch.nn.Module):
    def __init__(self, dim: int, device: torch.device) -> None:
        super().__init__()
        self.mlp = torch.nn.ModuleList()
        self.mlp.extend(
            [
                torch.nn.Linear(dim, 4 * dim, device=device),
                torch.nn.GELU(),
                torch.nn.Linear(4 * dim, dim, device=device),
                torch.nn.GELU()
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    dim = 1024
    use_fake_tensors = True
    dev = torch.cuda.current_device()
        
    with FakeTensorMode() if use_fake_tensors else contextlib.nullcontext():
        inp = torch.randn(dim, device=dev)
        mod = MLP(dim, dev) 
        with TestMode():
            mod(inp).sum().backward()
