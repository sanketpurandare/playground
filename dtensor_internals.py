# Run Command: torchrun --standalone --nproc_per_node=4 <file_name>.py

import os
import time
from typing import Callable, Optional

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.utils._python_dispatch import TorchDispatchMode

gpu_id = int(os.environ["LOCAL_RANK"])
DEVICE = f"cuda:{gpu_id}"
torch.distributed.init_process_group(backend="nccl", device_id=gpu_id)


class MyDispatchMode(TorchDispatchMode):

    def __init__(self, _dispatch_key=None):
        super().__init__(_dispatch_key)
        self.orig_dtensor_dispatch: Optional[Callable] = None

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        print(f"Rank: {gpu_id} func: {func}")
        return func(*args, **kwargs)

    def __enter__(self):
        if self.orig_dtensor_dispatch is None:
            self.orig_dtensor_dispatch = DTensor._op_dispatcher.dispatch

            def dispatch_wrapper(*args, **kwargs):
                with self:
                    res = self.orig_dtensor_dispatch(*args, **kwargs)
                return res

            DTensor._op_dispatcher.dispatch = dispatch_wrapper
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        DTensor._op_dispatcher.dispatch = self.orig_dtensor_dispatch
        return super().__exit__(exc_type, exc_val, exc_tb)


my_dispatch = MyDispatchMode()


def print_dtensor(dt, mesh):
    print(
        f"Rank: {mesh.get_rank()} Placements: {dt._spec.placements}: Shape: {dt.shape} Local Shape: {dt._local_tensor.shape}"
        f"\n{dt._local_tensor}"
    )
    torch.distributed.barrier()


mesh_dim_names = ("dp", "ep")
mesh_dims = (2, 2)
mesh = init_device_mesh("cuda", mesh_shape=mesh_dims, mesh_dim_names=mesh_dim_names)


t = torch.arange(32, dtype=torch.int32, device=DEVICE).unsqueeze(1)
t = t.reshape(8, 4)


with my_dispatch:
    placements = [Shard(0), Shard(0)]
    dt = distribute_tensor(t, mesh, placements)
    print_dtensor(dt, mesh)
    dt = dt.redistribute(mesh, [Replicate(), Shard(0)])
    print_dtensor(dt, mesh)
    dt = dt.redistribute(mesh, [Replicate(), Replicate()])
    print_dtensor(dt, mesh)


torch.distributed.barrier()
time.sleep(2)

if mesh:
    torch.distributed.destroy_process_group()
