"""Method-1 (player-embedding) personalization training mechanics.

One `PersonalizedChessformer` wrapper trains exactly one player's
`z_player`; this module never mixes player identities -- callers are
responsible for supplying train/val loaders already filtered to a single
player (see `data.style_dataset.build_style_datasets` on records
pre-filtered by `player_id`).

Deliberately separate from `training.style_trainer.StyleTrainer` (Method
2's different forward/loss shape) and from `training.trainer` (population
pretraining). Loss is direct legal-policy cross entropy -- no candidate
restriction, no value target, matching Method 1's frozen design.

Batch contract (same shape `data.style_dataset.StyleDataset` yields):
    x               [B, 64, 96]
    player_id       [B] long   -- unused here (constant: one player per run)
    player_elo      [B] float
    opponent_elo    [B] float
    policy_target   [B] long
    legal_mask      [B, 4352] bool
"""

import torch
import torch.nn.functional as F

from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer

# Implementation defaults, not a paper fact -- matches training.style_trainer's
# own defaults for consistency across the two personalization methods.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


class Method1Trainer:
    """Owns Method-1's frozen-base/trainable-z_player training mechanics.

    `wrapper.base` is frozen (already the case per `PersonalizedChessformer.
    __init__`); the ONLY trainable parameter is `wrapper.z_player`.
    """

    def __init__(
        self,
        wrapper: PersonalizedChessformer,
        lr: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ):
        self.wrapper = wrapper
        self.wrapper.base.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [self.wrapper.z_player], lr=lr, weight_decay=weight_decay,
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _policy_loss(self, batch) -> torch.Tensor:
        policy_logits, _ = self.wrapper(batch["x"], batch["player_elo"], batch["opponent_elo"])
        masked_logits = policy_logits.masked_fill(
            ~batch["legal_mask"], torch.finfo(policy_logits.dtype).min
        )
        return F.cross_entropy(masked_logits, batch["policy_target"])

    def train_epoch(self, train_loader) -> float:
        """Run one training pass. Returns the mean training loss over batches."""
        self.wrapper.base.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss = self._policy_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.current_epoch += 1
        return total_loss / num_batches

    def validate(self, val_loader) -> float:
        """Run one no-grad validation pass and update early-stopping state.

        Returns the mean validation loss over batches.
        """
        self.wrapper.base.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = self._policy_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        val_loss = total_loss / num_batches

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return val_loss
