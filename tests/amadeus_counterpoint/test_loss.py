import torch

from amadeus_counterpoint.training.loss import chessformer_loss


def _inputs(B=4, policy_size=4352):
    torch.manual_seed(0)
    policy_logits = torch.randn(B, policy_size, requires_grad=True)
    value_logits = torch.randn(B, 3, requires_grad=True)
    policy_target = torch.randint(0, policy_size, (B,))
    value_target = torch.randint(0, 3, (B,))

    # Legal mask always includes the target, plus a handful of other moves.
    legal_mask = torch.zeros(B, policy_size, dtype=torch.bool)
    for b in range(B):
        legal_mask[b, policy_target[b]] = True
        legal_mask[b, (policy_target[b] + 1) % policy_size] = True
        legal_mask[b, (policy_target[b] + 2) % policy_size] = True

    return policy_logits, value_logits, policy_target, value_target, legal_mask


def test_loss_components_are_finite():
    policy_logits, value_logits, policy_target, value_target, legal_mask = _inputs()

    total, policy, value = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask
    )

    assert torch.isfinite(total)
    assert torch.isfinite(policy)
    assert torch.isfinite(value)


def test_value_coefficient_is_applied():
    policy_logits, value_logits, policy_target, value_target, legal_mask = _inputs()

    total_a, policy_a, value_a = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask,
        value_coefficient=0.1,
    )
    total_b, policy_b, value_b = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask,
        value_coefficient=0.5,
    )

    assert torch.allclose(policy_a, policy_b)
    assert torch.allclose(value_a, value_b)
    assert torch.allclose(total_a, policy_a + 0.1 * value_a)
    assert torch.allclose(total_b, policy_b + 0.5 * value_b)
    assert not torch.allclose(total_a, total_b)


def test_backward_produces_gradients():
    policy_logits, value_logits, policy_target, value_target, legal_mask = _inputs()

    total, _, _ = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask
    )
    total.backward()

    assert policy_logits.grad is not None
    assert torch.isfinite(policy_logits.grad).all()
    assert value_logits.grad is not None
    assert torch.isfinite(value_logits.grad).all()


def test_illegal_high_logit_cannot_steal_probability_mass():
    policy_size = 4352
    policy_logits = torch.zeros(1, policy_size, requires_grad=True)
    value_logits = torch.zeros(1, 3, requires_grad=True)
    policy_target = torch.tensor([0])
    value_target = torch.tensor([1])

    legal_mask = torch.zeros(1, policy_size, dtype=torch.bool)
    legal_mask[0, 0] = True  # only the target move is legal

    with torch.no_grad():
        # An illegal move gets an enormous logit that would otherwise
        # dominate the softmax.
        policy_logits[0, 1] = 1e6

    _, policy_loss, _ = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask
    )

    # The only legal move is the target, so cross-entropy loss must be ~0
    # regardless of how large the illegal move's logit is.
    assert policy_loss.item() < 1e-4


def test_raising_target_logit_lowers_policy_loss():
    policy_logits, value_logits, policy_target, value_target, legal_mask = _inputs()

    _, loss_before, _ = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask
    )

    raised_logits = policy_logits.clone().detach()
    for b in range(raised_logits.shape[0]):
        raised_logits[b, policy_target[b]] += 10.0
    raised_logits.requires_grad_(True)

    _, loss_after, _ = chessformer_loss(
        raised_logits, value_logits, policy_target, value_target, legal_mask
    )

    assert loss_after.item() < loss_before.item()


def test_dtype_and_device_are_preserved():
    policy_logits, value_logits, policy_target, value_target, legal_mask = _inputs()
    policy_logits = policy_logits.double()
    value_logits = value_logits.double()

    total, policy, value = chessformer_loss(
        policy_logits, value_logits, policy_target, value_target, legal_mask
    )

    assert total.dtype == torch.float64
    assert policy.dtype == torch.float64
    assert value.dtype == torch.float64
