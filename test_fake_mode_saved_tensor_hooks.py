import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.utils._mode_utils import no_dispatch
from contextlib import nullcontext
from torch.distributed._tools.mem_tracker import MemTracker

if __name__ == "__main__":
    dev = torch.device("cuda:0")
    vocab_size = 8192
    bsz, seq_len = 256, 1024
    model_args = ModelArgs(
        n_layers=4,
        n_heads=12,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dim=768,
        dropout_p=0.1,
    )

    fake_mode = FakeTensorMode()

    def pack(t: torch.Tensor):
        return fake_mode.from_tensor(t)
    
    def unpack(fake_t: torch.Tensor):      
        dtype = fake_t.dtype
        size = fake_t.size()
        layout = fake_t.layout
        stride = fake_t.stride()
        device = fake_t.fake_device
        pin_memory = fake_t.is_pinned()
        requires_grad = fake_t.requires_grad       
        with no_dispatch():
            fake_t.untyped_storage().resize_(0)
            t = torch.empty_strided(
                size=size,
                stride=stride,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                ).zero_()
        return t



    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    # with nullcontext():
        with torch.device(dev):
            model = Transformer(model_args)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optimizer)
        with mem_tracker as mt:
            inp = torch.randint(0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev)
            out = model(inp)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    mt.display_snapshot("peak", units="MiB", tabulate=True)
    mt.display_modulewise_snapshots(depth=2, units="MiB", tabulate=True)
    # Check for accuracy of peak memory
    tracker_max = mt.get_tracker_snapshot('peak')[dev]['Total']
    cuda_max = torch.cuda.max_memory_allocated()
    accuracy = tracker_max / cuda_max
    print(f"Tracker Max: {tracker_max}, CUDA Max: {cuda_max}, Accuracy: {accuracy}")