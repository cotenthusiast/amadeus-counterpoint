"""Held-out validation for Broadcast-adaptation checkpoints.

Selects among already-trained Broadcast-adaptation checkpoints using
whatever validation dataloader the caller supplies. This module never
touches target-player or sealed-dyad data itself -- that guarantee comes
entirely from the caller pointing `dataloader` at shards that already
exclude the 8 targets (the same exclusion already used to build the
Broadcast-adaptation training corpus), not from any check performed here.
"""

import torch

from amadeus_counterpoint.training.loss import chessformer_loss


def evaluate_checkpoint(model, dataloader, device) -> dict:
    """Run one full pass over `dataloader`, no gradient, and report:

    - mean_policy_loss: PRIMARY checkpoint-selection criterion.
    - mean_value_loss: auxiliary diagnostic, never used for selection.
    - value_accuracy: fraction of examples where argmax(value_logits) ==
      value_target (W/D/L accuracy) -- diagnostic only.
    - move_match_accuracy: fraction of examples where the single
      highest-scoring LEGAL move exactly matches the human move
      (`policy_target`) -- diagnostic only.
    - num_examples: total examples seen.
    """
    model.eval()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct_value = 0
    correct_move = 0
    num_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            player_elo = batch["player_elo"].to(device)
            opponent_elo = batch["opponent_elo"].to(device)
            policy_target = batch["policy_target"].to(device)
            value_target = batch["value_target"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            policy_logits, value_logits = model(x, player_elo, opponent_elo)

            _, policy_loss, value_loss = chessformer_loss(
                policy_logits, value_logits, policy_target, value_target, legal_mask,
            )

            batch_size = policy_target.shape[0]
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size

            masked_logits = policy_logits.masked_fill(
                ~legal_mask, torch.finfo(policy_logits.dtype).min
            )
            predicted_move = masked_logits.argmax(dim=-1)
            correct_move += (predicted_move == policy_target).sum().item()

            predicted_value = value_logits.argmax(dim=-1)
            correct_value += (predicted_value == value_target).sum().item()

            num_examples += batch_size

    return {
        "mean_policy_loss": total_policy_loss / num_examples,
        "mean_value_loss": total_value_loss / num_examples,
        "value_accuracy": correct_value / num_examples,
        "move_match_accuracy": correct_move / num_examples,
        "num_examples": num_examples,
    }


def select_best_checkpoint(results: dict) -> str:
    """Return the key (checkpoint filename/step) with the lowest
    `mean_policy_loss` in `results` -- the PRIMARY selection rule.
    Accuracy/value metrics are diagnostics only and never override this.
    """
    if not results:
        raise ValueError("results is empty -- nothing to select from")
    return min(results, key=lambda key: results[key]["mean_policy_loss"])
