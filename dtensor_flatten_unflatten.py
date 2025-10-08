# Run Command: torchrun --standalone --nproc_per_node=4 <file_name>.py

import os
import time

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.utils._python_dispatch import TorchDispatchMode


class MyDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        print(f"func: {func}")
        return func(*args, **kwargs)


my_dispatch = MyDispatchMode()

orig_dtensor_dispatch = DTensor._op_dispatcher.dispatch


def dispatch_wrapper(*args, **kwargs):
    with my_dispatch:
        res = orig_dtensor_dispatch(*args, **kwargs)
    return res


DTensor._op_dispatcher.dispatch = dispatch_wrapper


def print_dtensor(dt, mesh):
    print(
        f"Rank: {mesh.get_rank()} Placements: {dt._spec.placements}: Shape: {dt.shape} Local Shape: {dt._local_tensor.shape}"
        f"\n{dt._local_tensor}"
    )
    torch.distributed.barrier()


gpu_id = int(os.environ["LOCAL_RANK"])
DEVICE = f"cuda:{gpu_id}"
torch.cuda.set_device(DEVICE)
torch.distributed.init_process_group(backend="nccl", device_id=gpu_id)
mesh_dim_names = ("dp", "ep")
mesh_dims = (2, 2)
mesh = init_device_mesh("cuda", mesh_shape=mesh_dims, mesh_dim_names=mesh_dim_names)

# Unflatten Fail - Sharding not propagated to unflattened dimensions

t2 = torch.arange(32, dtype=torch.int32, device=DEVICE).unsqueeze(1)
t2 = torch.expand_copy(t2, (32, 4)).reshape(32, 4)

placements = [Shard(0), Shard(0)]

dt2 = distribute_tensor(t2, mesh, placements)

print_dtensor(dt2, mesh)

with my_dispatch:
    dt2 = dt2.unflatten(0, (8, 4))

print_dtensor(dt2, mesh)

# Flatten Fail

t = torch.arange(32, dtype=torch.int32, device=DEVICE).unsqueeze(1)
t = torch.expand_copy(t, (32, 4)).reshape(8, 4, 4)


placements = [Shard(0), Shard(1)]
dt = distribute_tensor(t, mesh, placements)

print_dtensor(dt, mesh)

with my_dispatch:
    dt = dt.flatten(0, 1)

print_dtensor(dt, mesh)


torch.distributed.barrier()
time.sleep(2)

if mesh:
    torch.distributed.destroy_process_group()
