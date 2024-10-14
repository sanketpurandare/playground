import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.sac_estimator import SACEstimator
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


if __name__ == "__main__":
    dev = torch.cuda.current_device()
    vocab_size = 8192
    bsz, seq_len = 8, 1024
    model_args = ModelArgs(
        n_layers=4,
        n_heads=12,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dim=768,
        dropout_p=0.1,
    )
    with FakeTensorMode():
        with torch.device(dev):
            model = Transformer(model_args)
        inp = torch.randint(
            0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
        )

        sace = SACEstimator()
        with sace(estimate_mode_type='operator-level-cost-model'):
            loss = model(inp).sum()
        loss.backward()
        sace.pwlf_sac_tradeoff_curve(n_segments=2, save_tradeoff_graphs=True)
        sace.display_modulewise_sac_stats(depth=4, print_tabular=True)

