# torchrun --nproc_per_node=8 comm_test.py
import os
from typing import Any, Callable, Dict, Tuple
from typing_extensions import Unpack
import torch
import torch.distributed as dist


def trace_handler(p):
    rank = int(os.environ["RANK"])
    if rank == 0:
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    p.export_chrome_trace(f"all_reduce_{rank}_{str(p.step_num)}.gz")

def profile_func(prof: torch.profiler.profile, func: Callable, *args: Unpack[Tuple[Any, ...]], **kwargs: Any):
    with prof:
        for _ in range(16):
            func(*args, **kwargs if kwargs else {})
            prof.step()
        torch.cuda.synchronize()

def measure_func(func: Callable, *args: Unpack[Tuple[Any, ...]], **kwargs: Any):
    s_events = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
    e_events = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
    with torch.cuda.stream(torch.cuda.current_stream()):
        for _ in range(5):
            func(*args, **kwargs if kwargs else {})  
        
        for i in range(11):
            s_events[i].record(torch.cuda.current_stream())
            func(*args, **kwargs if kwargs else {})
            e_events[i].record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    print(f"Time: {sum(s_events[i].elapsed_time(e_events[i]) for i in range(1, 11))/10}")


def worker(device_mesh: dist.device_mesh.DeviceMesh, dim_name: str):
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=5, active=10, repeat=2),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
        on_trace_ready=trace_handler,
    )
    func = dist.all_reduce
    args = (torch.randn(2**20, device='cuda'),)
    kwargs = {"group": device_mesh.get_group(mesh_dim=dim_name)}
    # profile_func(prof, func, *args, **kwargs)
    measure_func(func, *args, **kwargs)
    args = (torch.randn(2**24, device='cuda'),)
    # profile_func(prof, func, *args, **kwargs)
    measure_func(func, *args, **kwargs)

    # func = torch.mm
    # args = (torch.randn(1024, 512, device="cuda"), torch.randn(512, 2048, device="cuda"))
    # profile_func(prof, func, *args)
    # args = (torch.randn(1024, 2048, device="cuda"), torch.randn(2048, 1024, device="cuda"))
    # profile_func(prof, func, *args)


if __name__ == "__main__":
    try:
        dist.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu_id)
        world_size = int(os.environ["WORLD_SIZE"])
        dims = (world_size,)
        names = ("dp",)
        world_mesh = dist.device_mesh.init_device_mesh("cuda", dims, mesh_dim_names=names)
        dist.barrier()
        worker(world_mesh, names[0])
        dist.barrier()
    except Exception as e:
        print(e)
    finally:
        dist.destroy_process_group()
    
