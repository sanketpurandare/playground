import torch

def test_fn():
    a = torch.randn(100, 150, device="cuda")
    w1 = torch.randn(128, 100, device="cuda")
    w2 = torch.randn(256, 100, device="cuda")

    b = torch.cos(a)
    c = torch.sin(a)

    d = torch.mm(w1, b)
    e = torch.mm(w2, c)

    return torch.mm(d, e.t_())

if __name__ == "__main__":

    compiled_fn = torch.compile(test_fn, backend="inductor", options={"trace.enabled": True, "trace.graph_diagram": True})
    t = compiled_fn()
    print(t.shape)