import torch

if __name__ == "__main__":

    storage_ids: set[int] = set()
    nbytes = 0

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        global nbytes
        st = t.untyped_storage()
        sid = id(st)
        if sid not in storage_ids:
            nbytes += st.nbytes()
            storage_ids.add(sid)
        return t

    def fn():
        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda", requires_grad=True)
        a = a + b
        a.add_(a)
        a.cos_()
        a.sin_()
        a.cos_()
        a.sin_()

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda x: x):
        fn()

    print("Total saved memory: ", nbytes)
    print("Storages saved: ", len(storage_ids))



