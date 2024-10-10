import torch
from torch.utils.flop_counter import FlopCounterMode

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.profiler import profile, record_function

device="cuda"
q32 = torch.randn(32, 32, 32, 32, dtype=torch.float32, device=device)
q16 = torch.randn(32, 32, 32, 32, dtype=torch.float16, device=device)
qb16 = torch.randn(32, 32, 32, 32, dtype=torch.bfloat16, device=device)
attn_mask = torch.randn(32, 32, dtype=torch.float16, device=device)

contexts = {
    # "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    # "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

def fn(context, q, k, v):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v)

def fn2(context, q, k, v):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.2)

def fn3(context, q, k, v):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.2)
    
def fn4(context, q, k, v):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.2, scale=1.0)
    
def fn5(context, q, k, v, attn_mask):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v, dropout_p=0.2, attn_mask=attn_mask)

for context in contexts.values():
    print("context:", context)
    # with profile(activities=[torch.profiler.ProfilerActivity.CPU, 
    #                      torch.profiler.ProfilerActivity.CUDA]) as prof:
    with FlopCounterMode(display=True) as flop_counter_mode:
        fn(context, q16, q16, q16)
        fn2(context, q16, q16, q16)
        fn3(context, q16, q16, q16)
        fn4(context, q16, q16, q16)
        # fn5(context, q16, q16, q16, attn_mask)
        fn(context, qb16, qb16, qb16)
        # fn(context, q32, q32, q32)
    # print(prof.key_averages().table(sort_by="cuda_time_total"))