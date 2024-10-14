import torch
from typing import Optional


if __name__ == "__main__":

    def random_ops_fn(x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = torch.nn.functional.dropout(x, p=0.3)
        if z is None:
            z = torch.nn.functional.dropout(y, p=0.6)
        w = torch.nn.functional.dropout(z, p=0.2)

        return (w, z)
    
    torch.manual_seed(777)
    x = torch.rand(256, 256, device="cuda")
    rng_state = torch.cuda.get_rng_state()

    # produce the outputs of first call to a random function
    w, z = random_ops_fn(x)

    # Try to reproduce the same output by supplying an intermediate value that skips a part of recomputation
    torch.cuda.set_rng_state(rng_state)
    w1, _ = random_ops_fn(x, z)

    # Try to reproduce the same output by supplying only the original tensor and full recomputation
    torch.cuda.set_rng_state(rng_state)
    w2, _, = random_ops_fn(x)

    print("Full Recomputation: ", torch.allclose(w, w2))
    print("Partial Recomputation: ", torch.allclose(w, w1))


