from typing import Dict, List
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.distributed._tools.mod_tracker import ModTracker

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.ModuleList([
            torch.nn.Linear(10, 20),
            torch.nn.GELU(),
            torch.nn.Linear(20, 10),
            torch.nn.GELU(),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)
        return x
    
class MyDispatchMode(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.mod_tracker = ModTracker()
        self.mod_functions: Dict[str, Dict[str, List[str]]] = {}
        self.pre_fw_order: List[str] = []
        self.pre_bw_order: List[str] = []

    def __enter__(self):
        self.mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.pre_fw_order.append(
                self.mod_tracker.get_known_fqn(mod)
                ),
            pre_bw_hook=lambda mod, g_out: self.pre_bw_order.append(
                self.mod_tracker.get_known_fqn(mod)
                )
        )
        self.mod_tracker.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mod_tracker.__exit__()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs or {}
        pass_type = "bw" if self.mod_tracker.is_bw else "fw"      
        for mod_name in self.mod_tracker.parents:
            self.mod_functions.setdefault(mod_name, {"fw": [], "bw": []})
            self.mod_functions[mod_name][pass_type].append(func.__name__)
        return func(*args, **kwargs)

if __name__ == "__main__":
    my_mod = MyModel().cuda()
    inp = torch.rand(10, 10, device="cuda")

    with MyDispatchMode() as md:
        my_mod(inp).sum().backward()

    print("Fw execution order: ", md.pre_fw_order)
    print("Bw execution order: ", md.pre_bw_order)
    for mod_name, functions in md.mod_functions.items():
        print(mod_name)
        print(functions) 