import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.composite_compliance import is_view_fn
from test_model import GPT, GPTConfig, loss_fn

_PYTORCH_MIN_ALLOCATE = 512
_MB = 2 ** 20
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
def get_tensor_mem( x: torch.Tensor) -> int:
    return (math.ceil(x.untyped_storage().size() / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE) / _MB

class MemDispatch(TorchDispatchMode):

    def __init__(self, mod_tracker: ModTracker):
        super().__init__()
        self.mod_tracker = mod_tracker
        self.tracked_mem = 0

    def print_tensor_meta(self, x: torch.Tensor):
        mem = get_tensor_mem(x)
        self.tracked_mem += mem
        print(f"Size: {x.size()}")
        print(f"Mem: {mem}")
  
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs if kwargs else {}
        res = func(*args, **kwargs)
        if not is_view_fn(func):
            print(f"Active Modules ({func.__name__}): {self.mod_tracker.parents}")
            pytree.tree_map_only(torch.Tensor, self.print_tensor_meta, res)
        return res

if __name__ == "__main__":
    mt = ModTracker()
    saved_tensor_mem = 0

    def pack(x: torch.Tensor):
        global saved_tensor_mem
        print(f"Active Modules (SavedTensorHooks): {mt.parents}")
        mem = get_tensor_mem(x)
        saved_tensor_mem += mem
        print(f"Size: {x.size()}")
        print(f"Mem: {mem}")
        return x

    def unpack(x: torch.Tensor):
        return x
    
    # dim = 1024
    # dev = torch.cuda.current_device()
    # inp = torch.randn(dim, device=dev)
    # mod = MLP(dim, dev)
    dev = torch.device(torch.cuda.current_device())
    n_layer = 1
    vocab_size = 8192
    config = GPTConfig(
        block_size=512,
        n_layer=n_layer,
        dropout=0.01,
        vocab_size=vocab_size,
        checkpoint_activations=False,
    )
    with torch.device(dev):
        model = GPT(config)
    torch.manual_seed(1)
    bsz, seq_len = 64, 512
    src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
    tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
    with mt, torch.autograd.graph.saved_tensors_hooks(pack, unpack), MemDispatch(mt):
        out = model(src)
        loss = loss_fn(out, tgt)

    print(f"Total saved tensor mem: {saved_tensor_mem}")
