import torch
import torch.utils._pytree as pytree
class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.ModuleList([
            torch.nn.Linear(10, 20, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(20, 10),
            torch.nn.GELU(),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)
        return x


if __name__ == "__main__":
    with torch.device(torch.cuda.current_device()):
        mod = MyModel()
        inp1 = torch.rand(4, 10)
        inp2 = torch.rand(4, 10)

        mod.mlp[0].register_forward_pre_hook(lambda mod, inp: pytree.tree_map_only(torch.Tensor, lambda x: torch.zeros_like(x), inp))
        mod.mlp[1].register_forward_pre_hook(lambda mod, inp: print(inp))
        mod.mlp[2].register_forward_hook(lambda mod, inp, out: pytree.tree_map_only(torch.Tensor, lambda x: torch.ones_like(x), out))
        mod.mlp[3].register_forward_pre_hook(lambda mod, inp: print(inp))

        mod(inp1)

