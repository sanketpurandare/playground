
import torch
import os
from torch.utils.benchmark import timer
from functools import partial, wraps
from torch._C._distributed_c10d import ProcessGroup, Work
from torch.futures import Future
import torch.distributed as dist
from contextlib import contextmanager
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import detect_fake_mode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Shard
from torch.distributed._tensor.api import distribute_tensor
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack


class MyDispatchMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        if func.__name__ == "_allgather_base_.default":
            print("all_gather_into_tensor")
            return None
        res = func(*args, **kwargs or {})
        return res




if __name__ == "__main__":
    # with FakeTensorMode(), MyDispatchMode():
    #     print(is_fake_mode())
    #     x = torch.randn(1000, 1000)
    #     y = torch.randn(1000, 1000)
    #     z = x + y
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    dist.barrier()



    # with torch.device(device):
    #     with MyDispatchMode() and FakeTensorMode():
    #         world_size = dist.get_world_size()
    #         input_tensor = torch.randn(collective_size, dtype=torch.bfloat16)
    #         output_tensor = torch.randn(collective_size * world_size, dtype=torch.bfloat16)
    #         dist.all_gather_into_tensor(output_tensor, input_tensor)




    # dist.init_process_group(backend="nccl")
    # Create a mesh topology with the available devices.
    mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
    big_tensor = torch.randn(100000, 88)
    if gpu_id == 0:
        print(torch.cuda.memory_allocated())

    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    if gpu_id == 0:
        print(torch.cuda.memory_allocated())
    local_tensor = my_dtensor._local_tensor
    if gpu_id == 0:
        print(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    if gpu_id == 0:
        print(big_tensor.untyped_storage().size()*big_tensor.untyped_storage().element_size())
        print(type(my_dtensor))
        print(my_dtensor.size())
        print(my_dtensor.untyped_storage().size()*my_dtensor.untyped_storage().element_size())
        print(type(local_tensor))
        print(local_tensor.size())
        print(local_tensor.untyped_storage().size()*local_tensor.untyped_storage().element_size())
        print(local_tensor.untyped_storage() == my_dtensor.untyped_storage())
        print(big_tensor.untyped_storage() == my_dtensor.untyped_storage())
        print(my_dtensor.placements)

    dist.barrier()
    dist.destroy_process_group()
