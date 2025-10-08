import torch
import torch.distributed as dist
import torch.nn.functional as F

import torch.utils._pytree as pytree
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


class TestMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        print("Op:", func)
        print("Args:")
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), args)
        print("Out:")
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), out)
        print(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        )
        return out


if __name__ == "__main__":
    world_size = 4
    rank = 0
    gpu_id = 0
    DEVICE = f"cuda:{gpu_id}"
    torch.cuda.set_device(DEVICE)
    store = FakeStore()
    dist.init_process_group("fake", rank=rank, world_size=world_size, store=store)
    mesh_dim_names = ("dp", "ep")
    mesh_dims = (2, 2)
    mesh = init_device_mesh("cuda", mesh_shape=mesh_dims, mesh_dim_names=mesh_dim_names)

    t = torch.arange(64, dtype=torch.int32, device=DEVICE)
    t = t.reshape(8, 4, 2)
    # print(t)
    placements = [Shard(0), Shard(0)]
    dt = distribute_tensor(t, mesh, placements)
    print(
        dt.shape,
        dt._local_tensor.shape,
        dt._spec.placements,
        dt._local_tensor.stride(),
        dt._spec.tensor_meta.stride,
    )
    dt = dt.redistribute(mesh, [Replicate(), Shard(0)])
    print(
        dt.shape,
        dt._local_tensor.shape,
        dt._spec.placements,
        dt._local_tensor.stride(),
        dt._spec.tensor_meta.stride,
    )
    dt = dt.redistribute(mesh, [Replicate(), Replicate()])
    print(
        dt.shape,
        dt._local_tensor.shape,
        dt._spec.placements,
        dt._local_tensor.stride(),
        dt._spec.tensor_meta.stride,
    )

    # t = torch.arange(32, dtype=torch.int32, device=DEVICE).unsqueeze(1)
    # t = torch.expand_copy(t, (32, 4)).reshape(2, 4, 4, 4)
    # t = t.to(dtype=torch.float32)
    # print(t)
    # placements = [Shard(0), Shard(1)]
    # dt = distribute_tensor(t, mesh, placements)
    # dt = dt.reshape(dt.shape[0], -1, dt.shape[-1])

    # print(dt.shape, dt._local_tensor.shape, dt._spec.placements)
    # dt = F.softmax(dt, dim=-1)
    # dt = torch.sigmoid(dt)
    # print(dt.shape, dt._local_tensor.shape, dt._spec.placements)
    # dt = dt.reshape(2, 4, 4, 4)
    # print(dt.shape, dt._local_tensor.shape, dt._spec.placements)

    # mat1 = torch.rand(
    #     10, 32, 16, device=DEVICE, dtype=torch.float32, requires_grad=True
    # )
    # mat2 = torch.rand(48, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    # mat2_expanded = mat2.expand(mat1.shape[0], -1, -1).transpose(-2, -1)

    # def batched_mm(
    #     mat1: torch.Tensor,
    #     mat2: torch.Tensor,
    # ) -> torch.Tensor:
    #     assert mat1.ndim == 3
    #     assert mat2.ndim == 3
    #     assert mat1.shape[2] == mat2.shape[1]
    #     out = mat1 @ mat2
    #     return out

    # out = batched_mm(mat1, mat2_expanded)
    # out = out.sum()
    # with TestMode():
    #     out.backward()
    # mat1grad = mat1.grad
    # mat2grad = mat2.grad
    # mat1.grad = None
    # mat2.grad = None
    # out3 = mat1 @ mat2.transpose(-2, -1)
    # out3 = out3.sum()
    # out3.backward()
    # print(torch.allclose(out, out3))
    # print(torch.allclose(mat1.grad, mat1grad))
    # print(torch.allclose(mat2.grad, mat2grad))
