import torch

from _helpers import CONFIG
from amadeus_counterpoint.models import Chessformer, PersonalizedChessformer
from amadeus_counterpoint.training.loss import chessformer_loss


def build_model() -> Chessformer:
    return Chessformer(**CONFIG)


def random_batch(B: int):
    x = torch.randn(B, 64, CONFIG["input_dim"])
    player_elo = torch.randint(0, 5000, (B,)).float()
    opponent_elo = torch.randint(0, 5000, (B,)).float()
    return x, player_elo, opponent_elo


def build_wrapper_and_training_batch(B: int = 3):
    """A fresh wrapper plus a fresh training batch (inputs, policy/value
    targets, and an all-legal mask) -- shared setup for the gradient tests
    below, which otherwise repeat this block verbatim."""
    model = build_model()
    wrapper = PersonalizedChessformer(model, nominal_elo=2000.0, identity="test-player")

    x, player_elo, opponent_elo = random_batch(B)
    policy_target = torch.randint(0, 4352, (B,))
    value_target = torch.randint(0, 3, (B,))
    legal_mask = torch.ones(B, 4352, dtype=torch.bool)

    return wrapper, x, player_elo, opponent_elo, policy_target, value_target, legal_mask


# --- Chessformer.forward(player_emb_override=...) -------------------------


def test_default_override_reproduces_existing_behavior():
    model = build_model()
    x, player_elo, opponent_elo = random_batch(3)

    policy_a, value_a = model(x, player_elo, opponent_elo)
    policy_b, value_b = model(x, player_elo, opponent_elo, player_emb_override=None)

    assert torch.equal(policy_a, policy_b)
    assert torch.equal(value_a, value_b)


def test_explicit_interpolated_override_matches_default_output():
    model = build_model()
    x, player_elo, opponent_elo = random_batch(3)

    policy_default, value_default = model(x, player_elo, opponent_elo)

    override = model.interpolate_elo(player_elo)
    policy_override, value_override = model(
        x, player_elo, opponent_elo, player_emb_override=override
    )

    assert torch.equal(policy_default, policy_override)
    assert torch.equal(value_default, value_override)


def test_override_shape_and_dtype_are_passed_through():
    model = build_model()
    B = 2
    x, player_elo, opponent_elo = random_batch(B)

    override = torch.randn(B, CONFIG["elo_dim"])
    policy, value = model(x, player_elo, opponent_elo, player_emb_override=override)

    assert policy.shape == (B, 4352)
    assert value.shape == (B, 3)
    assert policy.dtype == override.dtype


def test_opponent_conditioning_is_unaffected_by_override():
    model = build_model()
    B = 2
    x, player_elo, _ = random_batch(B)
    override = torch.randn(B, CONFIG["elo_dim"])

    opponent_elo_low = torch.full((B,), 500.0)
    opponent_elo_high = torch.full((B,), 4500.0)

    policy_low, _ = model(x, player_elo, opponent_elo_low, player_emb_override=override)
    policy_high, _ = model(x, player_elo, opponent_elo_high, player_emb_override=override)

    # Same override, different opponent Elo -> opponent's interpolate_elo path is still live.
    assert not torch.equal(policy_low, policy_high)


# --- PersonalizedChessformer -----------------------------------------------


def test_z_player_initialized_from_nominal_elo_interpolation():
    model = build_model()
    wrapper = PersonalizedChessformer(model, nominal_elo=2000.0, identity="test-player")

    expected = model.interpolate_elo(torch.tensor([2000.0])).squeeze(0)
    assert torch.allclose(wrapper.z_player.detach(), expected)


def test_base_parameters_are_frozen():
    model = build_model()
    wrapper = PersonalizedChessformer(model, nominal_elo=2000.0, identity="test-player")

    assert all(not p.requires_grad for p in wrapper.base.parameters())
    assert wrapper.z_player.requires_grad


def test_forward_matches_base_with_z_player_as_override():
    model = build_model()
    wrapper = PersonalizedChessformer(model, nominal_elo=2000.0, identity="test-player")

    B = 3
    x, player_elo, opponent_elo = random_batch(B)

    policy_wrapper, value_wrapper = wrapper(x, player_elo, opponent_elo)

    override = wrapper.z_player.unsqueeze(0).expand(B, -1)
    policy_direct, value_direct = model(
        x, player_elo, opponent_elo, player_emb_override=override
    )

    assert torch.equal(policy_wrapper, policy_direct)
    assert torch.equal(value_wrapper, value_direct)


def test_z_player_receives_policy_gradient_and_base_stays_frozen():
    wrapper, x, player_elo, opponent_elo, policy_target, value_target, legal_mask = (
        build_wrapper_and_training_batch()
    )

    policy_logits, value_logits = wrapper(x, player_elo, opponent_elo)
    total_loss, _, _ = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask,
        value_coefficient=0.0,
    )
    total_loss.backward()

    assert wrapper.z_player.grad is not None
    assert torch.any(wrapper.z_player.grad != 0)
    assert all(p.grad is None for p in wrapper.base.parameters())


def test_value_coefficient_zero_eliminates_value_gradient_contribution():
    wrapper, x, player_elo, opponent_elo, policy_target, value_target, legal_mask = (
        build_wrapper_and_training_batch()
    )

    policy_a, value_a = wrapper(x, player_elo, opponent_elo)
    total_a, _, _ = chessformer_loss(
        policy_a, value_a, policy_target, value_target, legal_mask,
        value_coefficient=0.0,
    )
    total_a.backward()
    grad_with_zero_coefficient = wrapper.z_player.grad.clone()
    wrapper.z_player.grad = None

    policy_b, value_b = wrapper(x, player_elo, opponent_elo)
    _, policy_loss_b, _ = chessformer_loss(
        policy_b, value_b, policy_target, value_target, legal_mask,
        value_coefficient=0.0,
    )
    policy_loss_b.backward()
    grad_policy_only = wrapper.z_player.grad.clone()

    assert torch.allclose(grad_with_zero_coefficient, grad_policy_only)


def test_nonzero_value_coefficient_changes_the_gradient():
    wrapper, x, player_elo, opponent_elo, policy_target, value_target, legal_mask = (
        build_wrapper_and_training_batch()
    )

    policy_a, value_a = wrapper(x, player_elo, opponent_elo)
    total_zero, _, _ = chessformer_loss(
        policy_a, value_a, policy_target, value_target, legal_mask,
        value_coefficient=0.0,
    )
    total_zero.backward()
    grad_zero = wrapper.z_player.grad.clone()
    wrapper.z_player.grad = None

    policy_b, value_b = wrapper(x, player_elo, opponent_elo)
    total_one, _, _ = chessformer_loss(
        policy_b, value_b, policy_target, value_target, legal_mask,
        value_coefficient=1.0,
    )
    total_one.backward()
    grad_one = wrapper.z_player.grad.clone()

    # Sanity check that value_coefficient is actually wired up: a nonzero
    # coefficient must change the gradient relative to the zero-coefficient case.
    assert not torch.allclose(grad_zero, grad_one)


# --- device handling -------------------------------------------------------


def test_z_player_init_follows_base_device_not_cpu_default():
    """Regression test for a real bug found during Kelvin2 GPU smoke testing:
    PersonalizedChessformer.__init__ created the nominal-Elo tensor with
    torch.tensor(...) (implicit CPU default) and fed it to base.interpolate_elo,
    which crashed with a device-mismatch RuntimeError whenever `base` was
    already on CUDA. `meta` is used here as a real, distinct device that
    requires no GPU: it reproduces the same "expected all tensors on the
    same device" failure as CPU-vs-CUDA did, so this test fails on the old
    code and passes on the fix, without needing actual CUDA hardware.
    """
    model = build_model().to("meta")

    wrapper = PersonalizedChessformer(model, nominal_elo=2000.0, identity="test-player")

    assert wrapper.z_player.device.type == "meta"
