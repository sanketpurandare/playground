import torch
import torch.utils._pytree as pytree
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
    x = torch.rand(2, 64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    experts = torch.rand(
        4, 128, 64, device="cuda", dtype=torch.float32, requires_grad=True
    )
    offsets = torch.tensor(
        [[16, 32, 48, 64], [16, 32, 48, 64]], device="cuda", dtype=torch.int32
    )

    def batched_grouped_mm(
        mat1: torch.Tensor, mat2: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        assert offs.ndim == 2  # [ob, num_experts]
        assert mat1.ndim == 3  # [ob, ib, dim]
        assert mat2.ndim == 3, f"{mat2.shape}"  # [num_experts, dim, hidden_dim]
        ob1, num_experts1 = offs.shape
        ob2, _, dim = mat1.shape
        num_experts2, dim, _ = mat2.shape
        assert ob1 == ob2, f"{mat1.shape} vs {offs.shape}"
        assert num_experts1 == num_experts2, f"{mat2.shape} vs {offs.shape}"
        assert dim == dim, f"{mat1.shape} vs {mat2.shape}"
        res = []
        for m1, off in zip(mat1, offs):
            res.append(torch._grouped_mm(m1, mat2, off))
        return torch.stack(res, 0)

    # with TestMode():
    # y = batched_grouped_mm(x.bfloat16(), experts.bfloat16(), offsets).type_as(x)
    # z = torch._grouped_mm(x.bfloat16(), experts.bfloat16(), offsets).type_as(x)
    # torch.allclose(y, z)
    # print(y.shape)
    # y.sum().backward()

    mat1 = torch.rand(
        2, 64, 128, device="cuda", requires_grad=True, dtype=torch.float32
    )
    off = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)
    experts = torch.rand(
        4, 128, 64, device="cuda", dtype=torch.float32, requires_grad=True
    )
    mat2 = torch.rand(
        2, 64, 256, device="cuda", dtype=torch.float32, requires_grad=True
    )

    # out1 = torch._grouped_mm(mat1.bfloat16(), experts.bfloat16(), off).type_as(mat1)
    # out2 = (
    #     torch._grouped_mm(
    #         experts.transpose(-2, -1).bfloat16(), mat1.transpose(-2, -1).bfloat16(), off
    #     )
    #     .transpose(-2, -1)
    #     .type_as(mat1)
    # )
    # print(out1)
    # print(out2)
    # print(torch.allclose(out1, out2, rtol=1e-2, atol=1e-2))

    res = []
    for m1, m2 in zip(mat1.transpose(-2, -1), mat2):
        res.append(torch._grouped_mm(m1.bfloat16(), m2.bfloat16(), off))
    out3 = torch.stack(res, 0).type_as(mat1)
    out3 = out3.sum()
    with TestMode():
        out3.backward()
    # print(out3.shape)
