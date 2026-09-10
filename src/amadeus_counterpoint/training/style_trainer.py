"""Method-2 style-residual training mechanics.

Deliberately separate from `training.trainer` (the population-pretraining
loop): the forward/loss shape here is structurally different (frozen base,
candidate-restricted scoring, per-player validation reporting), not a
variation the generic trainer can absorb.

Consumes already-formed batches from a caller-supplied `train_loader` /
`val_loader` (any iterable of batch dicts with the frozen contract below) --
where those batches come from (Broadcast loading, player-identity recovery,
phase-balanced sampling, train/validation game splitting) is explicitly out
of scope here and implemented separately.

Batch contract (each batch is a mapping with these keys):
    x               [B, 64, 96]
    player_id       [B] long
    player_elo      [B] float -- the mover's own Elo
    opponent_elo    [B] float
    policy_target   [B] long  -- human move, global 0..4351 policy index
    legal_mask      [B, 4352] bool
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from amadeus_counterpoint.models.candidates import topk_target_coverage
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.models.style_scoring import score_candidates

# Implementation defaults, not source-paper facts -- v1 does not specify an
# optimizer learning rate or weight decay for the style-embedding stage.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


@dataclass
class PlayerMetrics:
    player_id: int
    mean_ce: float
    coverage: float
    num_examples: int


@dataclass
class ValidationReport:
    mean_ce: float
    coverage: float
    num_examples: int
    per_player: dict[int, PlayerMetrics] = field(default_factory=dict)


class StyleTrainer:
    """Owns Method-2's frozen-base/trainable-style training mechanics.

    Freezes `base` on construction. `cnn`, `table`, and `residual` are the
    only trainable components (`residual.s` included); the base's value
    output is never used (see `score_candidates`).
    """

    def __init__(
        self,
        base: Chessformer,
        cnn: MoveStyleCNN,
        table: PlayerStyleTable,
        residual: StyleResidual,
        k: int,
        lr: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ):
        self.base = base
        self.base.requires_grad_(False)

        self.cnn = cnn
        self.table = table
        self.residual = residual
        self.k = k

        self.optimizer = torch.optim.AdamW(
            list(cnn.parameters()) + list(table.parameters()) + list(residual.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _score(self, batch: dict, target_index: torch.Tensor | None):
        return score_candidates(
            self.base,
            self.cnn,
            self.table,
            self.residual,
            batch["x"],
            batch["player_elo"],
            batch["opponent_elo"],
            batch["player_id"],
            batch["legal_mask"],
            self.k,
            target_index=target_index,
        )

    def train_epoch(self, train_loader) -> float:
        """Run one training pass. Returns the mean training loss over batches."""
        self.base.eval()
        self.cnn.train()
        self.table.train()
        self.residual.train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            output = self._score(batch, target_index=batch["policy_target"])
            loss = F.cross_entropy(output.candidate_scores, output.local_target_index)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.current_epoch += 1
        return total_loss / num_batches

    def validate(self, val_loader) -> ValidationReport:
        """Run one no-grad validation pass and update early-stopping state.

        Validation loss uses the same target-append candidate mechanics as
        training (so cross-entropy stays defined even when the human move
        falls outside the raw top-k); raw top-k coverage is measured
        separately, before any append, as a diagnostic that is never used
        as the optimization or early-stopping signal.
        """
        self.base.eval()
        self.cnn.eval()
        self.table.eval()
        self.residual.eval()

        overall_ce_sum = 0.0
        overall_coverage_sum = 0.0
        overall_count = 0
        per_player: dict[int, dict[str, float]] = {}

        with torch.no_grad():
            for batch in val_loader:
                target = batch["policy_target"]
                output = self._score(batch, target_index=target)

                per_example_ce = F.cross_entropy(
                    output.candidate_scores, output.local_target_index, reduction="none"
                )
                coverage = topk_target_coverage(output.masked_logits, self.k, target)

                for i in range(per_example_ce.shape[0]):
                    player_id = int(batch["player_id"][i].item())
                    ce = per_example_ce[i].item()
                    covered = float(coverage[i].item())

                    overall_ce_sum += ce
                    overall_coverage_sum += covered
                    overall_count += 1

                    stats = per_player.setdefault(
                        player_id, {"ce_sum": 0.0, "coverage_sum": 0.0, "count": 0.0}
                    )
                    stats["ce_sum"] += ce
                    stats["coverage_sum"] += covered
                    stats["count"] += 1.0

        per_player_metrics = {}
        for player_id, stats in per_player.items():
            per_player_metrics[player_id] = PlayerMetrics(
                player_id=player_id,
                mean_ce=stats["ce_sum"] / stats["count"],
                coverage=stats["coverage_sum"] / stats["count"],
                num_examples=int(stats["count"]),
            )

        report = ValidationReport(
            mean_ce=overall_ce_sum / overall_count,
            coverage=overall_coverage_sum / overall_count,
            num_examples=overall_count,
            per_player=per_player_metrics,
        )

        if report.mean_ce < self.best_val_loss:
            self.best_val_loss = report.mean_ce
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return report
