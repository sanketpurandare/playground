import torch
from torch import nn
from torch.autograd.graph import register_multi_grad_hook
from torch.utils._pytree import tree_flatten
from torch.distributed._tools.mod_tracker import ModTracker
if __name__ =="__main__":
    d = nn.Dropout(0.01)
    l = nn.Linear(16, 16)

    def pre_hook(mod, inputs):
        print("In pre-fw hook")
        flattened_ins, _ = tree_flatten(inputs)
        inp_tensors = [t for t in flattened_ins if isinstance(t, torch.Tensor) and t.requires_grad]
        if inp_tensors:
            register_multi_grad_hook(inp_tensors, lambda _: print("In post-bw hook"))

    def post_hook(mod, inputs, outputs):
        print("In post-fw hook")
        flattened_outs, _ = tree_flatten(outputs)
        out_tensors = [t for t in flattened_outs if isinstance(t, torch.Tensor) and t.requires_grad]
        if out_tensors:
            register_multi_grad_hook(out_tensors, lambda _: print("In pre-bw hook"))

    # d.register_forward_pre_hook(pre_hook)
    # d.register_forward_hook(post_hook)

    mt = ModTracker()
    mt.register_user_hooks(
        pre_fw_hook=lambda mod, inp: print(f"pre-fw: {mt.get_known_fqn(mod)}"),
        post_fw_hook=lambda mod, inp, op: print(f"post-fw: {mt.get_known_fqn(mod)}"),
        pre_bw_hook=lambda mod, gop: print(f"pre-bw: {mt.get_known_fqn(mod)}"),
        post_bw_hook=lambda mod, ginp: print(f"post-bw: {mt.get_known_fqn(mod)}"),
    )
    with mt:
        t = torch.randn(20, 16)
        lin = l(t)
        out = d(lin)
        print(lin.untyped_storage() == out.untyped_storage())
        out.sum().backward()
