import math

import torch
import torch.nn.functional as F

from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual


# --- PlayerStyleTable ---------------------------------------------------


def test_player_style_table_shape():
    table = PlayerStyleTable(num_players=8, style_dim=32)
    player_id = torch.tensor([0, 3, 7])

    z_u = table(player_id)

    assert z_u.shape == (3, 32)


def test_player_style_table_configurable_num_players_and_dim():
    table = PlayerStyleTable(num_players=3, style_dim=16)

    z_u = table(torch.tensor([0, 1, 2]))

    assert z_u.shape == (3, 16)


def test_player_style_table_gradients_reach_only_selected_rows():
    table = PlayerStyleTable(num_players=4, style_dim=8)
    player_id = torch.tensor([1, 1, 2])

    z_u = table(player_id)
    z_u.sum().backward()

    grad = table.embeddings.weight.grad
    assert grad is not None
    assert torch.all(grad[0] == 0)
    assert torch.any(grad[1] != 0)
    assert torch.any(grad[2] != 0)
    assert torch.all(grad[3] == 0)


# --- StyleResidual ------------------------------------------------------


def test_style_residual_shape():
    residual = StyleResidual(style_dim=32, init_s=0.17)
    move_features = torch.randn(2, 5, 32)
    z_u = torch.randn(2, 32)

    out = residual(move_features, z_u)

    assert out.shape == (2, 5)


def test_style_residual_init_value_and_trainability():
    residual = StyleResidual(style_dim=32, init_s=0.17)

    assert math.isclose(residual.s.item(), 0.17, rel_tol=1e-6)
    assert residual.s.requires_grad


def test_style_residual_exact_scaling_formula():
    residual = StyleResidual(style_dim=32, init_s=0.17)
    move_features = torch.randn(2, 5, 32)
    z_u = torch.randn(2, 32)

    out = residual(move_features, z_u)

    phi_hat = F.normalize(move_features, dim=-1)
    z_hat = F.normalize(z_u, dim=-1)
    expected = 0.17 * math.sqrt(32) * torch.einsum("bkd,bd->bk", phi_hat, z_hat)

    assert torch.allclose(out, expected)


def test_style_residual_scale_invariance_from_normalization():
    residual = StyleResidual(style_dim=32, init_s=0.17)
    move_features = torch.randn(2, 5, 32)
    z_u = torch.randn(2, 32)

    out_a = residual(move_features, z_u)
    out_b = residual(move_features * 10.0, z_u * 3.0)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_style_residual_gradients_reach_move_features_z_u_and_s():
    residual = StyleResidual(style_dim=32, init_s=0.17)
    move_features = torch.randn(2, 5, 32, requires_grad=True)
    z_u = torch.randn(2, 32, requires_grad=True)

    out = residual(move_features, z_u)
    out.sum().backward()

    assert move_features.grad is not None
    assert torch.any(move_features.grad != 0)
    assert z_u.grad is not None
    assert torch.any(z_u.grad != 0)
    assert residual.s.grad is not None
    assert residual.s.grad.item() != 0


def test_style_residual_has_no_extra_parameters():
    residual = StyleResidual(style_dim=32, init_s=0.17)

    names = [name for name, _ in residual.named_parameters()]

    assert names == ["s"]
