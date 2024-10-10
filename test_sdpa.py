import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.profiler import profile, record_function

from torch.utils.flop_counter import FlopCounterMode

# Set device
device = "cuda"

# Define contexts for efficient and flash attention
contexts = {
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

# Set dimensionalities and sequence lengths
d_qk = 512  # Embedding dimension for query and key projections
d_v = 512   # Embedding dimension for value projections
num_heads = 8
s_q = 20    # Query sequence length
s_kv = 10   # Key/Value sequence length
batch_size = 32

# Move tensors to the specified device
q = torch.rand(s_q, batch_size, d_qk, device=device)  # Query tensor (tgt)
kv = torch.rand(s_kv, batch_size, d_qk, device=device)  # Key and Value tensors (src)

in_proj_weight = torch.rand(3 * d_qk, d_qk, device=device)  # Weights for Q, K, V projections combined

# Optional bias for projections
in_proj_bias = torch.rand(3 * d_qk, device=device)  # Bias for Q, K, V projections (optional)

# Output projection (O)
out_proj_weight = torch.rand(d_qk, d_qk, device=device)  # Output projection weight
out_proj_bias = torch.rand(d_qk, device=device)  # Output projection bias

# Generate a causal mask for the query sequence (s_q)
causal_mask = nn.Transformer.generate_square_subsequent_mask(s_q).to(device)  # Size: (s_q, s_q)

# Adjust the mask to fit the shape (s_q, s_kv) if necessary
adjusted_mask = causal_mask[:, :s_kv]  # Crop the mask to match (s_q, s_kv)

# Multi-head attention forward pass
attn_output, attn_output_weights = F.multi_head_attention_forward(
    query=q,  # Query tensor
    key=kv,   # Key tensor
    value=kv, # Value tensor
    embed_dim_to_check=d_qk,  # Check embedding dimension
    num_heads=num_heads, 
    in_proj_weight=in_proj_weight, 
    in_proj_bias=in_proj_bias, 
    bias_k=None, 
    bias_v=None, 
    add_zero_attn=False, 
    dropout_p=0.0, 
    out_proj_weight=out_proj_weight, 
    out_proj_bias=out_proj_bias, 
    training=True, 
    key_padding_mask=None, 
    need_weights=True, 
    attn_mask=adjusted_mask,  # Pass the adjusted mask
    use_separate_proj_weight=False, 
    q_proj_weight=None, 
    k_proj_weight=None, 
    v_proj_weight=None, 
    static_k=None, 
    static_v=None, 
    is_causal=False  # Causal masking handled manually via attn_mask
)

print("Attention Output Shape:", attn_output.shape)
print("Attention Weights Shape:", attn_output_weights.shape)

# Update the `fn` function to use the correct arguments
def fn(context, q, k, v, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, attn_mask):
    with sdpa_kernel(context):
        return F.multi_head_attention_forward(
            query=q, 
            key=k, 
            value=v, 
            embed_dim_to_check=d_qk,  # Check embedding dimension
            num_heads=num_heads, 
            in_proj_weight=in_proj_weight, 
            in_proj_bias=in_proj_bias, 
            bias_k=None, 
            bias_v=None, 
            add_zero_attn=False, 
            dropout_p=0.0, 
            out_proj_weight=out_proj_weight, 
            out_proj_bias=out_proj_bias, 
            training=True, 
            key_padding_mask=None, 
            need_weights=True, 
            attn_mask=attn_mask,  # Pass the adjusted mask
            use_separate_proj_weight=False, 
            q_proj_weight=None, 
            k_proj_weight=None, 
            v_proj_weight=None, 
            static_k=None, 
            static_v=None, 
            is_causal=False  # Causal masking handled manually via attn_mask
        )

# Iterate over contexts and print the output shape for each
for context in contexts.values():
    print("Context:", context)
    # with profile(activities=[torch.profiler.ProfilerActivity.CPU, 
    #                     torch.profiler.ProfilerActivity.CUDA]) as prof:
    with FlopCounterMode(display=True) as flop_counter_mode:
        output = fn(context, q, kv, kv, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, adjusted_mask)
        print("Output Shape:", output[0].shape, output[1].shape)
    # print(prof.key_averages().table(sort_by="cuda_time_total"))