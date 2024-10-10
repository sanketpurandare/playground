import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

if __name__ == "__main__":
    def _train_step(
        model: nn.Module,
        optimizer: optim.Optimizer,
        inp: torch.Tensor,
    ):
        out = model(inp)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    dev = torch.cuda.current_device()
    vocab_size = 8192
    bsz, seq_len = 32, 1024
    model_args = ModelArgs(
        n_layers=4,
        n_heads=12,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dim=768,
        dropout_p=0.1,
    )
    runtime_estimator = RuntimeEstimator()
    
    with FakeTensorMode():
        with torch.device(dev):
            model = Transformer(model_args)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev)
        with runtime_estimator("operator-level-benchmark"):
            _train_step(model, optimizer, inp)
        with runtime_estimator("operator-level-cost-model"):
            _train_step(model, optimizer, inp)

    # Actual model runtime
    with torch.device(dev):
        model = Transformer(model_args)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    inp = torch.randint(0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev)
    warmup_iters, actual_iters = 2, 5
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup_iters):
        _train_step(model, optimizer, inp)
    start_event.record()
    for _ in range(actual_iters):
        _train_step(model, optimizer, inp)
    end_event.record()
    torch.cuda.synchronize()
    measured_time = start_event.elapsed_time(end_event) / actual_iters
    print(f"Actual total_time: {measured_time:.3f} ms")