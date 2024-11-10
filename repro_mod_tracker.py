import os
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.viz._cycles import warn_tensor_cycles
from torch.distributed._composable.checkpoint_activation import checkpoint
from torch.distributed._tools.mod_tracker import ModTracker

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = ConvBlock(3, 8)
        self.conv_block2 = ConvBlock(8, 16)
        self.conv_block3 = ConvBlock(16, 32)
        self.flatten = nn.Flatten()
        self.mlp1 = nn.Linear(25088, 128)  # Adjust input features
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x

def main(rank,):    
    torch.cuda.set_device(rank)
    use_ac = True

#    warn_tensor_cycles()
    model = SimpleCNN().cuda()
    if use_ac:
        for module in model.modules():
            if isinstance(module, ConvBlock):
                checkpoint(module)
    model = FSDP(model)

    inputs = torch.randn(512, 3, 224, 224).cuda()
    # with FlopCounterMode() as flop_counter:
    with ModTracker():
        for i in range(1000):
            if i % 100 == 0 and rank == 0:
                print (f"i={i}")
                print(f"Memory: {torch.cuda.memory_allocated()/2**30} GiB")
            pred = model(inputs)
            pred.sum().backward()
   

if __name__ == "__main__":
    # Initialize the distributed process group
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    world_size = 8
    rank = 0
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=0, world_size=world_size, store=store
    )
    main(rank)
    