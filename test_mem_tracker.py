import torch
from torch import nn
from torch.utils.mem_tracker import MemTracker


def test_cuda_tracker_equvivalence(device, dtype, batch_size, layers, dim):

    class DummyModel(nn.Module):
        def __init__(self, layers: int, dim: int):
            super(DummyModel, self).__init__()
            self.layers = nn.ModuleList()
            for _ in range(layers):
                self.layers.extend([nn.Linear(dim, dim), nn.ReLU()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = DummyModel(layers, dim).to(device=device, dtype=dtype)
    layers = model.layers
    print(layers)
    for name, submod in layers.named_children():
        print(name)
    exit()
    optim = torch.optim.Adam(model.parameters(), foreach=True)
    mem_tracker = MemTracker(
        units="B", display_modulewise_stats=True, display_peak_stats=True
    )
    mem_tracker.track_external(model, optim)
    with mem_tracker as mt:
        input_batch = torch.randn(batch_size, dim, device=device, dtype=dtype)
        output = model(input_batch)
        output.sum().backward()
        output = None
        optim.step()
        optim.zero_grad()

    # tracker_max = mt.get_peak_mem()
    # cuda_max = torch.cuda.max_memory_allocated()
    # accuracy = tracker_max / cuda_max

    # self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

if __name__ == "__main__":
    test_cuda_tracker_equvivalence("cuda", torch.float32, 2048, 10, 2048)
