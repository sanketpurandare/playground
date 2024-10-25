if __name__== "__main__":    
    import torch
    import torchvision.models as models
    from torch.distributed._tools.mem_tracker import MemTracker
    from torch.distributed._composable import checkpoint
    from torchvision.models.resnet import Bottleneck
    device, dtype = torch.device("cuda:0"), torch.float32
    with torch.device(device):
        model = models.resnet152().to(dtype=dtype)
    for module in model.modules():
        if isinstance(module, Bottleneck):
            checkpoint(module)
    print(model)
    optim = torch.optim.Adam(model.parameters(), foreach=True)
    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optim)
    with mem_tracker as mt:
        for i in range(2):
            input_batch = torch.randn(256, 3, 224, 224, device=device, dtype=dtype)
            model(input_batch).sum().backward()
            optim.step()
            optim.zero_grad()
            if i == 0:
                # to account for lazy init of optimizer state
                mt.reset_mod_stats()
    mt.display_snapshot("peak", units="MiB", tabulate=True)
    mt.display_modulewise_snapshots(depth=4, units="MiB", tabulate=True)
    # Check for accuracy of peak memory
    tracker_max = mt.get_tracker_snapshot('peak')[device]['Total']
    cuda_max = torch.cuda.max_memory_allocated()
    accuracy = tracker_max / cuda_max
    print(f"Tracker Max: {tracker_max}, CUDA Max: {cuda_max}, Accuracy: {accuracy}")