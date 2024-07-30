import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed._tools.mod_tracker import ModTracker

def test_user_hooks():

    class Bar(nn.Module):
        def __init__(self):
            super(__class__, self).__init__()
            self.foo = nn.Linear(10, 10)

        def forward(self, x):
            return F.relu_(self.foo(x))

    mt = ModTracker()
    test_op = []

    def hook(mod, mt, hook_name):
        mfqn = mt.get_known_fqn(mod) if mod is not None else None
        test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

    mod = Bar()

    mt.register_user_hooks(lambda m, i: hook(m, mt, "pre_fw"), lambda m, i, o: hook(m, mt, "post_fw"), lambda m, go: hook(m, mt, "pre_bw"), lambda m, gi: hook(m, mt, "post_bw"))
    with mt:
        mod(torch.rand(10, 10, requires_grad=True)).sum().backward()
    expected_op = [('pre_fw', 'Bar', True, False), ('pre_fw', 'Bar.foo', True, False), ('post_fw', 'Bar.foo', True, False), ('post_fw', 'Bar', True, False), ('pre_bw', 'Bar', True, True), ('pre_bw', 'Bar.foo', True, True), ('post_bw', 'Bar', True, True), ('post_bw', 'Bar.foo', True, True)]

    print(list(zip(test_op, expected_op)))
    try:
        mt.register_user_hooks(lambda x, y: x, None, None, None)
    except Exception as e:
        print(isinstance(e, AssertionError))
    test_op.clear()
    with mt:
        loss = mod(torch.rand(10, 10, requires_grad=True)).sum()
        del mod
        loss.backward()
    expected_op = [('pre_fw', 'Bar', True, False), ('pre_fw', 'Bar.foo', True, False), ('post_fw', 'Bar.foo', True, False), ('post_fw', 'Bar', True, False), ('pre_bw', None, False, True), ('pre_bw', None, False, True), ('post_bw', None, False, True), ('post_bw', None, False, True)]
    print(list(zip(test_op, expected_op)))


if __name__ == "__main__":
    test_user_hooks()
