import logging
from functools import wraps
from contextlib import contextmanager
from typing import Callable, NamedTuple
from enum import StrEnum
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten
class ConvImpl(StrEnum):
    conv1d = "conv1d"
    conv2d = "conv2d"
    conv3d = "conv3d"

class _SavedConvs(NamedTuple):
    conv1d: Callable
    conv2d: Callable
    conv3d: Callable


class ConvTracker:
    def __init__(self):
        self._saved_convs = _SavedConvs(
            conv1d=F.conv1d,
            conv2d=F.conv2d,
            conv3d=F.conv3d
        )
        self.CurrentImpl = None

    def __enter__(self):
        @wraps(F.conv1d)
        def conv1d(*args):
            self.CurrentImpl = ConvImpl.conv1d
            return self._saved_convs.conv1d(*args)

        @wraps(F.conv2d)
        def conv2d(*args):
            self.CurrentImpl = ConvImpl.conv2d
            return self._saved_convs.conv2d(*args)

        @wraps(F.conv3d)
        def conv3d(*args):
            self.CurrentImpl = ConvImpl.conv3d
            return self._saved_convs.conv3d(*args)

        F.conv1d = conv1d
        F.conv2d = conv2d
        F.conv3d = conv3d

    def __exit__(self, *args):
        F.conv1d = self._saved_convs.conv1d
        F.conv2d = self._saved_convs.conv2d
        F.conv3d = self._saved_convs.conv3d

class IgnoreDistMode(TorchDispatchMode):
    def __init__(self):
        self.conv_tracker = ConvTracker()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == aten.convolution.default:
            logging.info(f"Conv Impl: {self.conv_tracker.CurrentImpl}")
            logging.info(f'Function name: {str(func.__name__)}')
            logging.info(f'Function type: {type(func)}')
            logging.info(f'Func: {func}')

        res = func(*args, **kwargs or {})
        return res

    def __enter__(self):
        self.conv_tracker.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        self.conv_tracker.__exit__(*args)
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # Move to CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input data
    input_1d = torch.randn(1, 3, 10).to(device)  # batch_size, channels, sequence_length
    input_2d = torch.randn(1, 3, 10, 10).to(device)  # batch_size, channels, height, width
    input_3d = torch.randn(1, 3, 10, 10, 10).to(device)  # batch_size, channels, depth, height, width

    # Weights
    weights_1d = torch.randn(2, 3, 3).to(device)  # output_channels, input_channels, kernel_size
    weights_2d = torch.randn(2, 3, 3, 3).to(device)  # output_channels, input_channels, kernel_height, kernel_width
    weights_3d = torch.randn(2, 3, 3, 3, 3).to(device)  # output_channels, input_channels, kernel_depth, kernel_height, kernel_width

    # Convolutional layers
    with IgnoreDistMode():
        output_1d = F.conv1d(input_1d, weights_1d)
        output_2d = F.conv2d(input_2d, weights_2d)
        output_3d = F.conv3d(input_3d, weights_3d)