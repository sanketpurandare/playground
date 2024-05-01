from datetime import timedelta
import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed._composable.fsdp
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
from contextlib import contextmanager, nullcontext
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._python_dispatch import TorchDispatchMode
import logging

aten = torch.ops.aten
from torch.distributed._functional_collectives import all_gather_tensor_inplace
import torch.distributed._functional_collectives_impl as func_col_impl

func_col_impl._use_native_funcol = True


class IgnoreDistMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        logging.info(str(func.__name__))
        logging.info(type(func))
        logging.info(func)

        if func == torch.ops.c10d._allgather_base_.default:
            logging.info(str(func.__name__))
            logging.info(type(func))
            logging.info(func)
            logging.info(f"Arg types: {[type(arg) for arg in args]}")
            logging.info(f"Arg 0 size: {args[0].size()}")
            logging.info(f"Arg 1 size: {args[1].size()}")
            logging.info(f"Torch Script Inp Obj: {ProcessGroup.unbox(args[2])}")
            # func = torch.ops._c10d_functional.all_gather_into_tensor.default
            res = func(*args, **kwargs or {})
            # res = all_gather_tensor_inplace(args[0], args[1], ProcessGroup.unbox(args[2]))
        else:
            res = func(*args, **kwargs or {})
        if isinstance(res, tuple):
            # logging.info(res)
            logging.info(f" Res types: {[type(r) for r in res]}")
            work = Work.unbox(res[1])
            logging.info(f"Torch Script Op Obj: {work.__dir__()}")
            logging.info(f"Future: {work.get_future().__dir__()}")
            logging.info(f"Future value: {work.get_future().value()}")
            logging.info(f"Future done: {work.get_future().done()}")
            logging.info(f"Future value type: {type(work.get_future().value())}")
            logging.info(f"Future value size: {work.get_future().value()[0].size()}")
            # logging.info(f"Tensor Size: {res[0].size()}")

        # if isinstance(res, torch.Tensor):
        #     print("Function name: ", str(func.__name__))
        #     print("Result type: ", type(res))
        #     print("Result size", res.size())
        #     print("Result element size", res.element_size())
        #     print("Result device: ", res.device)
        return res


class FakeWork(Work):

    def __init__(self):
        super().__init__()

    def get_future(self) -> Future:
        future = Future()
        future.set_result(None)
        return future

    def wait(self, timeout: timedelta = ...) -> bool:
        return True


def all_gather_into_tensor(out_tensor: torch.Tensor, in_tensor: torch.Tensor, group=None, async_op=False):
    if async_op:
        return FakeWork()
    return None


@contextmanager
def bypass_collectives():
    saved_all_gather = dist.all_gather_into_tensor

    try:
        dist.all_gather_into_tensor = all_gather_into_tensor
        yield
    finally:
        dist.all_gather_into_tensor = saved_all_gather


def run_worker(rank, world_size):
    logging.getLogger().setLevel(
        logging.DEBUG if rank == 0 else logging.CRITICAL
    )
    # logging.getLogger().setLevel(logging.DEBUG)
    store = FakeStore()
    # dist.init_process_group(
    #     "fake", rank=rank, world_size=world_size, store=store
    # )
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )
    logging.info(f"Number of visible devices:  {torch.cuda.device_count()}")
    torch.cuda.set_device(rank)

    # with FakeTensorMode() as fake_mode:
    with nullcontext():
        with IgnoreDistMode():

            test_tensor = torch.randn(10000, device="cuda")
            output_tensor = torch.empty(
                test_tensor.numel() * world_size, device="cuda"
            )
            # all_gather_tensor_inplace(output_tensor, test_tensor, dist.group.WORLD)
            work = dist.all_gather_into_tensor(output_tensor, test_tensor, None, True)

            if work is not None:
                if rank == 0:
                    print(type(work))
                    future = work.get_future()
                    print(future.done())
                    print(type(future.value()))
                    print(future.value()[0].size())
                    print(future.value()[0].untyped_storage() == output_tensor.untyped_storage())
                print(work.wait())
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
