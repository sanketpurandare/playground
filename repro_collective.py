import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.distributed._functional_collectives import all_reduce
from torch.utils._python_dispatch import TorchDispatchMode
from functools import wraps
from contextlib import contextmanager, nullcontext
import logging
from datetime import timedelta
from typing import cast, Optional, overload

class FakeWork(Work):
    def __init__(self):
        super().__init__()

    def get_future(self) -> Future:
        future = Future()
        future.set_result(None)
        return future

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True

def _all_reduce_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (args[0], fakework_script_obj)

def _barrier_meta(*args):
    fakework = FakeWork()
    fakework.__setattr__("getFuture", fakework.get_future)
    fakework_script_obj = fakework.boxed()
    return (fakework_script_obj)

if not torch._running_with_deploy():
    # Library MUST be defined at module scope or it doesn't work
    # Creating a "DEF" Library always crashes torch::deploy so we create our
    # Library instances here guarded against running inside it
    lib_impl = torch.library.Library("c10d", "IMPL")
    lib_impl.impl("allreduce_", _all_reduce_meta, "Meta")
    lib_impl.impl("barrier", _barrier_meta, "Meta")

class IgnoreDistMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        logging.info(f'Function name: {str(func.__name__)}')
        logging.info(f'Function type: {type(func)}')
        logging.info(f'Func: {func}')

        res = func(*args, **kwargs or {})
        return res

def run_test():
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    logging.getLogger().setLevel(
        logging.DEBUG if rank == 0 else logging.CRITICAL
    )

    # with nullcontext():
    with FakeTensorMode():
        with IgnoreDistMode():

            test_tensor = torch.randn(10000, device="cuda")
            output = all_reduce(test_tensor, reduceOp="avg", group=dist.group.WORLD)
            

            dist.all_reduce(test_tensor)
        dist.barrier()



if __name__ == "__main__":
    gpu_id = 0
    world_size = 4
    dims = (world_size,)
    names = ("dp",)
    store = FakeStore()
    dist.init_process_group(
        "fake", rank=gpu_id, world_size=world_size, store=store
    )
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    try:
        run_test()
    finally:
        dist.destroy_process_group()