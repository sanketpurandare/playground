import torch
import torch.nn.functional as F


if __name__ == "__main__":
    a = torch.rand((16, 64, 128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.rand((64, 128), device="cuda", dtype=torch.bfloat16)
    c = a @ b
    print(c.shape)
