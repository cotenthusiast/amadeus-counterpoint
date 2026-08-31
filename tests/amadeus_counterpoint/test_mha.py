import torch

from amadeus_counterpoint.models.mha import MultiHeadAttention

CONFIG = {"d_model": 32, "num_heads": 4, "dropout": 0.0, "d1": 4, "d2": 8, "d3": 4}


def _build():
    d34096 = torch.empty(4096, CONFIG["d3"])
    torch.nn.init.xavier_normal_(d34096)
    return MultiHeadAttention(**CONFIG, d34096=d34096)


def test_no_bias_anywhere_in_the_qkv_and_output_projection():
    # Chessformer 79M sets omit_qkv_biases=True. Upstream builds attention
    # via torch.nn.MultiheadAttention(bias=not omit_qkv_biases); that single
    # `bias` flag controls BOTH in_proj (QKV) and out_proj bias in PyTorch's
    # implementation (see nn.MultiheadAttention.__init__: `self.out_proj =
    # NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, ...)`),
    # so the output projection must be unbiased too, matching QKV.
    mha = _build()

    assert mha.keys.bias is None
    assert mha.queries.bias is None
    assert mha.values.bias is None
    assert mha.proj.bias is None


def test_forward_and_backward_still_work_without_output_projection_bias():
    mha = _build()
    x = torch.randn(2, 64, CONFIG["d_model"], requires_grad=True)

    out = mha(x)

    assert out.shape == (2, 64, CONFIG["d_model"])
    out.sum().backward()
    assert x.grad is not None
