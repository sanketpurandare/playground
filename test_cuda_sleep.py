import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

def measure_sleep(actual_iters: int, warmp_up_iters: int, cycles: int, s2: torch.cuda.Stream) -> float:
    """Delay for the given number of milliseconds
    """
    start = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]

    with torch.cuda.stream(s2):
        for _ in range(warmp_up_iters):
            torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    with torch.cuda.stream(s2):
        for i in range(actual_iters):
            start[i].record(s2)
            torch.cuda._sleep(cycles)
            end[i].record(s2)
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(start, end)) / actual_iters

def measure_mm(actual_iters: int, warmp_up_iters: int, a: torch.Tensor, b: torch.Tensor, s1: torch.cuda.Stream) -> float:
    """Measure the time to run a matrix multiplication
    """
    start = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]
    # Warm up
    with torch.cuda.stream(s1):
        for _ in range(warmp_up_iters):
            torch.mm(a, b)
    torch.cuda.synchronize()
    # Measure

    with torch.cuda.stream(s1):
        for i in range(actual_iters):
            start[i].record(s1)
            torch.mm(a, b)
            end[i].record(s1)
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(start, end)) / actual_iters

def measure_mm_sleep(actual_iters: int, warmp_up_iters: int, a: torch.Tensor, b: torch.Tensor, cycles: int, s1: torch.cuda.Stream, s2: torch.cuda.Stream) -> float:
    """Measure the time to run a matrix multiplication
    """
    start = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(actual_iters)]
    sync_event = torch.cuda.Event()
    # Warm up
    with torch.cuda.stream(s1):
        for _ in range(warmp_up_iters):
            torch.mm(a, b)
    torch.cuda.synchronize()
    # Measure

    with torch.cuda.stream(s1):
        for i in range(actual_iters):
            start[i].record(s1)
            with torch.cuda.stream(s2):
                torch.cuda._sleep(cycles)
                sync_event.record()
            torch.mm(a, b)
            s1.wait_stream(s2)
            s1.wait_event(sync_event)
            end[i].record(s1)
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(start, end)) / actual_iters

def print_type(x):
    print(type(x))

from functools import wraps
from contextlib import contextmanager, nullcontext

@contextmanager
def capture_sync_ops():
    saved_wait_stream = torch.cuda.Stream.wait_stream
    saved_wait_event = torch.cuda.Stream.wait_event
    saved_event_record = torch.cuda.Event.record
    saved_event_wait = torch.cuda.Event.wait
    saved_synchronize = torch.cuda.synchronize
    saved_stream_synchronize = torch.cuda.Stream.synchronize

    @wraps(torch.cuda.Stream.wait_stream)
    def wait_stream(self: torch.cuda.Stream, stream: torch.cuda.Stream):
        print("Caught wait stream")
        print("Self Stream id: ", self.stream_id)
        print("Arg Stream id: ", stream.stream_id)
        return saved_wait_stream(self, stream)
    
    @wraps(torch.cuda.Stream.wait_event)
    def wait_event(self: torch.cuda.Stream, event: torch.cuda.Event):
        print("Caught Wait Event")
        print("Event id: ", event.cuda_event)
        print("Stream id: ", self.stream_id)
        return saved_wait_event(self, event)
    
    @wraps(torch.cuda.Event.record)
    def event_record(self: torch.cuda.Event, stream: torch.cuda.Stream=None):
        print("Caught record Event")
        print("Event id: ", self.cuda_event)
        return saved_event_record(self, stream)
    
    @wraps(torch.cuda.Event.wait)
    def event_wait(self: torch.cuda.Event, stream: torch.cuda.Stream=None):
        print("Caught wait Event")
        print("Event id: ", self.cuda_event)
        return saved_event_wait(self, stream)
    
    @wraps(torch.cuda.synchronize)
    def synchronize(device: torch.cuda.device=None):
        print("Caught snychronize")
        return saved_synchronize(device)
    
    @wraps(torch.cuda.Stream.synchronize)
    def stream_synchronize(self: torch.cuda.Stream):
        print("Caught stream synchronize")
        print("Self Stream id: ", self.stream_id)
        return saved_stream_synchronize(self)
    
    try:
        torch.cuda.Stream.wait_stream = wait_stream
        torch.cuda.Stream.wait_event = wait_event
        torch.cuda.Event.record = event_record
        torch.cuda.Event.wait = event_wait
        torch.cuda.synchronize = synchronize
        torch.cuda.Stream.synchronize = stream_synchronize
        yield
    finally:
        torch.cuda.Stream.wait_stream = saved_wait_stream
        torch.cuda.Stream.wait_event = saved_wait_event
        torch.cuda.Event.record = saved_event_record
        torch.cuda.Event.wait = saved_event_wait
        torch.cuda.synchronize = saved_synchronize
        torch.cuda.Stream.synchronize = saved_stream_synchronize


class MyDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"function: {func.__name__}")
        print(torch.cuda.current_stream() == torch.cuda.default_stream())
        print("Args")
        res = func(*args, **kwargs or {})
        return res

if __name__ ==  "__main__":
    torch.cuda.set_device(0)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.current_stream()
    m = 256
    k = 1024
    n = 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    cycles = 10 ** 3
    with capture_sync_ops():
    # with nullcontext():
        with MyDispatchMode():
            print(measure_sleep(10, 5, cycles, s2))
            print(measure_mm(10, 5, a, b, s1))
            print(measure_mm_sleep(10, 5, a, b, cycles, s1, s2))
