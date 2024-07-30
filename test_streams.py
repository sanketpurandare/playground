import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.storage import _StorageBase
from contextlib import contextmanager
from functools import partial
from torch.utils.weak import weakref, WeakIdKeyDictionary

def print_type(x):
    print(type(x))

class MyModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear(x)
        return x.relu_()

class MyModel(torch.nn.Module):
    def __init__(self, change_shape_in_recomp: bool):
        super().__init__()
        self.change_shape_in_recomp = change_shape_in_recomp
        self.a = torch.nn.Linear(2, 2)

    def forward(self, x):
        if self.change_shape_in_recomp:
            x.relu_()
        random_tensor = torch.randn(1, 2)
        x = random_tensor + self.a(x)
        return x

class MyDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"function: {func.__name__}")
        # print(torch.cuda.current_stream() == torch.cuda.default_stream())
        print("Args")
        tree_map(print_type, args)
        res = func(*args, **kwargs or {})
        print("Res")
        tree_map(print_type, res)
        print()
        return res


@contextmanager
def capture_resize():
    orig_resize = torch.UntypedStorage.resize_

    def resize_(st: torch.UntypedStorage, size: int):
        print("resize")
        return orig_resize(st, size)

    torch.UntypedStorage.resize_ = resize_

    try:
        yield
    finally:
        torch.UntypedStorage.resize_ = orig_resize

def callback_fn(message: str, w: weakref.ref):
    print(message)


if __name__ == "__main__":
    d = WeakIdKeyDictionary()
    with MyDispatchMode(), capture_resize():
        x = torch.randn(100, device="cuda")
        y = torch.randn(10, 10, device="cuda")
        z = torch.randn(10, device="cuda")
        wx = weakref.ref(x, partial(callback_fn, "x"))
        wy = weakref.ref(y, partial(callback_fn, "y"))
        print("Hello")
        del y
        del x
