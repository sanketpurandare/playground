import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
from torch.distributed._functional_collectives import all_gather_tensor, all_reduce
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
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

        if func == torch.ops.c10d._allgather_base_.default:
            logging.info(f'Function name: {str(func.__name__)}')
            logging.info(f'Function type: {type(func)}')
            logging.info(f'Func: {func}')
            logging.info(f"Arg types: {[type(arg) for arg in args]}")
            logging.info(f"Arg 0 size: {args[0].size()}")
            logging.info(f"Arg 1 size: {args[1].size()}")
            logging.info(f"Torch Script Inp Obj: {ProcessGroup.unbox(args[2])}")
            # func = torch.ops._c10d_functional.all_gather_into_tensor.default
            res = func(*args, **kwargs or {})
            # res = all_gather_tensor_inplace(args[0], args[1], ProcessGroup.unbox(args[2]))
        elif func == torch.ops.c10d.allreduce_.default:
            logging.info(f'Function name: {str(func.__name__)}')
            logging.info(f'Function type: {type(func)}')
            logging.info(f'Func: {func}')
            logging.info(f"Arg types: {[type(arg) for arg in args]}")
            logging.info(f"Args: {args}")

            res = func(*args, **kwargs or {})
        else:
            res = func(*args, **kwargs or {})
        if isinstance(res, tuple):
            logging.info(res)
            print(res[1]._method_names())
            logging.info(f" Res types: {[type(r) for r in res]}")
            work = FakeWork.unbox(res[1])
            print(type(work))
            print(work.__class__)
            # logging.info(f"Torch Script Op Obj: {work.__dir__()}")
            # logging.info(f"Future: {work.get_future().__dir__()}")
            # logging.info(f"Future value: {work.get_future().value()}")
            # logging.info(f"Future done: {work.get_future().done()}")
            # logging.info(f"Future value type: {type(work.get_future().value())}")
            # logging.info(f"Future value size: {work.get_future().value()}")
            # logging.info(f"Tensor Size: {res[0][0].size()}")

        # if isinstance(res, torch.Tensor):
        #     print("Function name: ", str(func.__name__))
        #     print("Result type: ", type(res))
        #     print("Result size", res.size())
        #     print("Result element size", res.element_size())
        #     print("Result device: ", res.device)
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
            output_tensor = torch.empty(
                test_tensor.numel() * world_size, device="cuda"
            )
            # work = all_gather_tensor(output_tensor, test_tensor, dist.group.WORLD)
            # work = dist.all_gather_into_tensor(output_tensor, test_tensor, None, True)
            # res = all_reduce(test_tensor, reduceOp="avg", group=dist.group.WORLD)
            # res = res.wait()
            # print(res.untyped_storage().data_ptr()==test_tensor.untyped_storage().data_ptr())
            work = dist.all_reduce(test_tensor)
            print(type(work))

            # if work is not None:
            #     if rank == 0:
            #         print(type(work))
            #         future = work.get_future()
            #         print(future.done())
            #         print(type(future.value()))
            #         print(future.value()[0].size())
            #         print(future.value()[0].untyped_storage() == output_tensor.untyped_storage())
            #     print(work.wait())
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

