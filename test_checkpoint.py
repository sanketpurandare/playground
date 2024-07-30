from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools import ModTracker


class MLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp_block = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim), nn.ReLU())

    def forward(self, x):
        return self.mlp_block(x)


class Foo(nn.Module):
    def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
        super().__init__()
        self.linears = nn.ModuleList()
        self.use_ac = use_ac
        for _ in range(n_layers):
            self.linears.append(nn.Linear(dim, dim))

    def forward(self, x):
        for i, block in enumerate(self.linears):
            if i >= 1 and self.use_ac:
                x = checkpoint(block, x, preserve_rng_state=True, use_reentrant=False)
            else:
                x = block(x)
            # x = nn.functional.relu(x)
        return x


if __name__ == "__main__":
    torch.set_default_device('cuda')
    batch_size = 2
    dim = 8
    n_layers = 2
    use_fake_mode = False
    with FakeTensorMode() if use_fake_mode else nullcontext():
        test_op = []

        def hook(mod, mt, hook_name):
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))
        mt = ModTracker()
        mt.register_user_hooks(lambda m, i: hook(m, mt, "pre_fw"), lambda m, i, o: hook(m, mt, "post_fw"), lambda m, go: hook(m, mt, "pre_bw"), lambda m, gi: hook(m, mt, "post_bw"))
        model = Foo(n_layers, dim, True).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(batch_size, dim, dim, device="cuda", dtype=torch.bfloat16)
        with mt:
            model(x).sum().backward()
        print(test_op)
        # expected_op = [('pre_fw', 'Foo', True, False), ('pre_fw', 'Foo.linears.0', True, False), ('post_fw', 'Foo.linears.0', True, False), ('pre_fw', 'Foo.linears.1', True, False), ('post_fw', 'Foo.linears.1', True, False), ('post_fw', 'Foo', True, False), ('pre_bw', 'Foo', True, True), ('pre_bw', 'Foo.linears.1', True, True), ('pre_fw', 'Foo.linears.1', True, True), ('post_fw', 'Foo.linears.1', True, True), ('post_bw', 'Foo.linears.1', True, True), ('pre_bw', 'Foo.linears.0', True, True)]
    print(torch.cuda.max_memory_allocated())
