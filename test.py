import torch

from torch.utils.benchmark import timer
from functools import partial, wraps


class MyClass:
    def __init__(self, name):
        self.name = name
        self.t2 = torch.randn(8192, 8192, device="cuda", requires_grad=True)

    def fn(self, t1):
        print("Actual function for : ", self.name)
        t3 = torch.mm(t1, self.t2)
        return t3


def dec(func, obj, obj_no):
    print("Decorator function")
    
    @wraps(func)
    def inner(*args, **kwargs):
        print(obj)
        print(inner.__name__)
        print(f"Decorator function args len {len(args)}, kwargs len {len(kwargs)}")
        print("Decorator function inner for obj no: ", obj_no)
        return func(*args, **kwargs)
    return inner


if __name__ == "__main__":
    t1 = torch.randn(8192, 8192, device="cuda")
    mc = MyClass("mc")
    mc1 = MyClass("mc1")
    mc.fn = dec(mc.fn, mc, 0)
    mc1.fn = dec(mc1.fn, mc1, 1)
    mc1.fn(t1)
    mc.fn(t1)
