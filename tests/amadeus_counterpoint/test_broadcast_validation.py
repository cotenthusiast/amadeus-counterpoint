import pytest
import torch

from amadeus_counterpoint.evaluation.broadcast_validation import evaluate_checkpoint, select_best_checkpoint


class _FakeModel:
    """Deterministic stand-in for Chessformer: fixed logits, so accuracy and
    loss can be computed by hand."""

    def __init__(self, policy_logits, value_logits):
        self.policy_logits = policy_logits
        self.value_logits = value_logits

    def eval(self):
        return self

    def __call__(self, x, player_elo, opponent_elo):
        return self.policy_logits, self.value_logits


def _batch(policy_target, value_target, legal_mask):
    n = len(policy_target)
    return {
        "x": torch.zeros(n, 64, 96),
        "player_elo": torch.zeros(n),
        "opponent_elo": torch.zeros(n),
        "policy_target": torch.tensor(policy_target),
        "value_target": torch.tensor(value_target),
        "legal_mask": torch.tensor(legal_mask, dtype=torch.bool),
    }


def test_evaluate_checkpoint_computes_expected_accuracies():
    # Example 0: legal everywhere, highest logit at index 0 -- matches target.
    # Example 1: index 0 has the highest raw logit but is ILLEGAL; the
    # highest LEGAL logit is index 1, but the target is 2 -- a miss.
    policy_logits = torch.tensor([
        [5.0, 1.0, 1.0, 1.0],
        [10.0, 3.0, 1.0, 1.0],
    ])
    legal_mask = [
        [True, True, True, True],
        [False, True, True, True],
    ]
    policy_target = [0, 2]

    # Example 0: value argmax = 0, matches target 0 -- correct.
    # Example 1: value argmax = 2, target is 1 -- a miss.
    value_logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    value_target = [0, 1]

    model = _FakeModel(policy_logits, value_logits)
    dataloader = [_batch(policy_target, value_target, legal_mask)]

    metrics = evaluate_checkpoint(model, dataloader, device=torch.device("cpu"))

    assert metrics["num_examples"] == 2
    assert metrics["move_match_accuracy"] == 0.5
    assert metrics["value_accuracy"] == 0.5
    assert isinstance(metrics["mean_policy_loss"], float)
    assert isinstance(metrics["mean_value_loss"], float)
    assert metrics["mean_policy_loss"] > 0.0


def test_evaluate_checkpoint_weights_batches_by_example_count():
    # Two batches of different sizes -- the mean must be example-weighted,
    # not batch-weighted (a naive average-of-batch-means would be wrong).
    policy_logits_a = torch.tensor([[5.0, 1.0]] * 3)  # 3 correct examples
    policy_logits_b = torch.tensor([[1.0, 5.0]])  # 1 incorrect example (target 0)
    legal_mask_a = [[True, True]] * 3
    legal_mask_b = [[True, True]]
    value_logits = torch.zeros(1, 3)

    model_a_batch = _batch([0, 0, 0], [0, 0, 0], legal_mask_a)
    model_a_batch["x"] = torch.zeros(3, 64, 96)
    model_b_batch = _batch([0], [0], legal_mask_b)

    class _SwitchingModel:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def __call__(self, x, player_elo, opponent_elo):
            self.calls += 1
            if self.calls == 1:
                return policy_logits_a, torch.zeros(3, 3)
            return policy_logits_b, torch.zeros(1, 3)

    metrics = evaluate_checkpoint(_SwitchingModel(), [model_a_batch, model_b_batch], torch.device("cpu"))

    assert metrics["num_examples"] == 4
    assert metrics["move_match_accuracy"] == 0.75  # 3 correct out of 4, not (1.0 + 0.0) / 2


def test_select_best_checkpoint_picks_lowest_policy_loss_ignoring_diagnostics():
    results = {
        "step_1": {"mean_policy_loss": 2.0, "mean_value_loss": 0.1, "value_accuracy": 0.9, "move_match_accuracy": 0.5},
        "step_2": {"mean_policy_loss": 1.5, "mean_value_loss": 5.0, "value_accuracy": 0.1, "move_match_accuracy": 0.1},
        "step_3": {"mean_policy_loss": 1.8, "mean_value_loss": 0.05, "value_accuracy": 0.99, "move_match_accuracy": 0.99},
    }

    assert select_best_checkpoint(results) == "step_2"


def test_select_best_checkpoint_rejects_empty_results():
    with pytest.raises(ValueError):
        select_best_checkpoint({})
