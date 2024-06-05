import torch


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
            torch.mm(a, b)
            s1.wait_stream(s2)
            end[i].record(s1)
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(start, end)) / actual_iters


if __name__ ==  "__main__":
    torch.cuda.set_device(0)
    s2 = torch.cuda.Stream()
    s1 = torch.cuda.current_stream()
    a = torch.randn(1000, 10000, device="cuda", dtype=torch.float32)
    b = torch.randn(10000, 1000, device="cuda", dtype=torch.float32)
    cycles = 1000000
    print(measure_sleep(10, 5, cycles, s2))
    print(measure_mm(10, 5, a, b, s1))
    print(measure_mm_sleep(10, 5, a, b, cycles, s1, s2))
