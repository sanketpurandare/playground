import torch
from torch import nn
from torch.distributed._tools.mem_tracker import MemTracker
from torch._subclasses.fake_tensor import FakeTensorMode


def test_cuda_tracker_equvivalence(device, dtype, batch_size, n_layers, dim):

    class DummyModel(nn.Module):
        def __init__(self, n_layers: int, dim: int):
            super().__init__()
            self.linear = nn.ModuleList()
            for _ in range(n_layers):
                self.linear.append(nn.Linear(dim, dim))
                self.linear.append(nn.ReLU())

        def forward(self, x):
            for lin in self.linear:
                x = lin(x)
                x = lin(x)
            return x
    with torch.device(device):
        model = DummyModel(n_layers, dim).to(dtype=dtype)
    optim = torch.optim.Adam(model.parameters(), foreach=True)
    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optim)
    with mem_tracker as mt:
        for i in range(2):
            input_batch = torch.randn(batch_size, dim, device=device, dtype=dtype)
            model(input_batch).sum().backward()
            optim.step()
            optim.zero_grad()
            if i == 0:
                # to account for lazy init of optimizer state
                mt.reset_mod_stats()
    mt.display_snapshot("peak", units="MB", tabulate=True)
    mt.display_modulewise_snapshots(depth=2, units="MB", tabulate=True)
    # Check for accuracy of peak memory
    tracker_max = mt.get_tracker_snapshot('peak')[torch.device(torch.cuda.current_device())]['Total']
    cuda_max = torch.cuda.max_memory_allocated()
    accuracy = tracker_max / cuda_max
    print(f"Tracker Max: {tracker_max}, CUDA Max: {cuda_max}, Accuracy: {accuracy}")
    print(accuracy >= 0.9)


def test_tracker_attribution(device, dtype, batch_size, n_layers, dim):
    dev = torch.device(device, torch.cuda.current_device()) if device == "cuda" else torch.device(device)
    def get_param_grad_optstate_actual_bytes(model: nn.Module, opt: torch.optim.Optimizer):
        param_bytes = 0
        grad_bytes = 0
        opt_state_bytes = 0
        seen_grad = set()
        for param in model.parameters():
            param_bytes += param.numel() * param.element_size()
            if param.grad is not None and param.grad not in seen_grad:
                seen_grad.add(param.grad)
                grad_bytes += param.grad.numel() * param.grad.element_size()
        seen_grad.clear()

        for state in opt.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor) and v.device == dev:
                    opt_state_bytes += v.numel() * v.element_size()
        return param_bytes, grad_bytes, opt_state_bytes

    def get_param_grad_optstate_bytes_from_tracker(
        tracker: MemTracker,
    ):
        snapshot = tracker.get_tracker_snapshot("current")

        param_bytes = snapshot[dev]["Parameter"]
        grad_bytes = snapshot[dev]["Gradient"]
        opt_state_bytes = snapshot[dev]["Optstate"]
        return param_bytes, grad_bytes, opt_state_bytes

    def test_attribution_equivalence(
        mt: MemTracker,
        model: nn.Module,
        opt: torch.optim.Optimizer,
    ):
        actual = get_param_grad_optstate_actual_bytes(model, opt)
        tracker = get_param_grad_optstate_bytes_from_tracker(mt)
        for a, b in zip(actual, tracker):
            print(a, b)
            if a == 0:
                print(b == 0)
            else:
                print(b / a >= 0.9)

    class DummyModel(nn.Module):
        def __init__(self, n_layers: int, dim: int):
            super(__class__, self).__init__()
            self.MLP_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.MLP_layers.extend([nn.Linear(dim, 2 * dim), nn.GELU()])
                self.MLP_layers.extend([nn.Linear(2 * dim, dim), nn.GELU()])

        def forward(self, x):
            for layer in self.MLP_layers:
                x = layer(x)
            return x
    with torch.device(dev):
        model = DummyModel(n_layers, dim).to(dtype=dtype)
    optim = torch.optim.Adam(model.parameters(), foreach=True)
    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optim)
    with mem_tracker as mt:

        input_batch = torch.randn(batch_size, dim, device=device, dtype=dtype)
        # Before forward: Only parameters and input are allocated
        test_attribution_equivalence(mt, model, optim)
        output = model(input_batch)
        output.sum().backward()
        # After forward: Gradients are allocated
        test_attribution_equivalence(mt, model, optim)
        output = None
        optim.step()
        # After step: Optimizer state is allocated
        test_attribution_equivalence(mt, model, optim)
        optim.zero_grad()
        # After zero_grad: Gradients are deallocated
        test_attribution_equivalence(mt, model, optim)



if __name__ == "__main__":
    test_cuda_tracker_equvivalence("cuda", torch.float32, 2048, 10, 2048)
    # with FakeTensorMode():
    #     test_tracker_attribution("cpu", torch.float16, 1024, 4, 1024)
    # with FakeTensorMode():
    #     test_tracker_attribution("cuda", torch.float16, 1024, 4, 1024)
    # test_tracker_attribution("cpu", torch.float32, 1024, 4, 1024)
    # test_tracker_attribution("cuda", torch.float32, 1024, 4, 1024)
