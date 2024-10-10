from run_time_test import EstimateMode
import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

if __name__ == "__main__":
    dev = torch.cuda.current_device()
    vocab_size = 8192
    bsz, seq_len = 64, 1024
    model_args = ModelArgs(
        n_layers=4,
        n_heads=12,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dim=768,
        dropout_p=0.1,
    )

    runtime_estimator = EstimateMode()
    
    with FakeTensorMode():
        with torch.device(dev):
            model = Transformer(model_args)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev)
        with runtime_estimator("operator-level-benchmark"):
            out = model(inp)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()