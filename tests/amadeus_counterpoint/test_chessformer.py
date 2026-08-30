import torch

from amadeus_counterpoint.models import Chessformer

CONFIG = dict(
    d_model=32,
    num_heads=4,
    num_layers=2,
    dropout=0.0,
    d1=4,
    d2=8,
    d3=4,
    d_ff=64,
    head_hid_dim=16,
    input_dim=96,
    elo_dim=8,
)


def build_model() -> Chessformer:
    return Chessformer(**CONFIG)


def test_forward_shapes_and_backward():
    model = build_model()
    B = 3
    x = torch.randn(B, 64, CONFIG["input_dim"])
    player_elo = torch.randint(0, 5000, (B,))
    opponent_elo = torch.randint(0, 5000, (B,))

    policy, value = model(x, player_elo, opponent_elo)

    assert policy.shape == (B, 4352)
    assert value.shape == (B, 3)

    (policy.sum() + value.sum()).backward()


def test_gab_template_bank_is_shared_across_blocks():
    model = build_model()

    banks = [block.attention.gab.d34096 for block in model.blocks]

    assert all(bank is model.d34096 for bank in banks)
    assert len(set(id(bank) for bank in banks)) == 1


def test_elo_interpolation_matches_official_ordering():
    model = build_model()

    # weight_low = elo / 5000 multiplies elo_low, so elo=0 -> elo_high
    # and elo=5000 -> elo_low (counterintuitive endpoint naming).
    at_zero = model.interpolate_elo(torch.tensor([0]))
    at_max = model.interpolate_elo(torch.tensor([5000]))

    assert torch.allclose(at_zero[0], model.elo_high.weight[0])
    assert torch.allclose(at_max[0], model.elo_low.weight[0])
