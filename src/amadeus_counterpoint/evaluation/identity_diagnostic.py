"""Post-hoc cross-player identity diagnostic for Method 1 / Method 2.

Purpose: on the SAME non-sealed held-out validation positions each method's
own training already used for checkpoint selection, check whether the
learned representations are actually player-specific -- i.e. whether player
P's own representation predicts P's held-out moves better than a
no-personalization baseline and better than another player's representation.

This is a diagnostic only. It reuses every existing piece of the
personalization stack unmodified (data.style_dataset's split/balance logic,
models.personalized_chessformer's embedding override, models.candidates'
top-K selection, models.style_scoring's candidate scoring, evaluation.
generation.checkpoint_loading's checkpoint wiring) -- nothing here retrains,
alters a checkpoint, or introduces a new identity/exclusion mechanism.
Sealed target-vs-target games cannot appear here: the StyleGameRecord input
this module consumes is only ever produced by data.broadcast_ingest.
iter_target_game_records, which never yields a target-vs-target game in the
first place.

Validation-split reconstruction is method-specific and NOT interchangeable
(verified empirically, not assumed): data.style_dataset.split_games_by_player
shuffles per player using ONE shared random.Random(seed), consumed in sorted
player_id order, so a solo (single-player) split and a joint (all-players)
split diverge for the same player even at the same seed. Method 1 training
(scripts/train_method1_player.py) filters to one player BEFORE splitting;
Method 2 training (scripts/train_method2.py) splits all 8 players jointly,
never filtering first. `build_method1_val_dataset`/`build_method2_val_datasets`
below mirror each script's own call sequence exactly, so each method's rows
here are the literal validation positions that method's own checkpoint
selection was scored against.

Method 2 metrics -- three genuinely different things, not to be confused:
  1. Full-action-space generic baseline (evaluate_generic): the frozen base
     over all 4352 legal actions. Descriptive only -- NEVER subtracted
     against a Method-2 candidate-conditioned NLL, since a K(+1)-restricted,
     renormalized distribution is not on the same scale as the full action
     space.
  2. Candidate-matched comparison (evaluate_method2_row): base, correct-style,
     and every wrong-style NLL/accuracy all computed over the IDENTICAL
     oracle-expanded K(+1) candidate set (raw top-K, target appended if
     absent -- exactly select_candidates' existing training/validation
     mechanics). These three ARE directly comparable, and are what the
     ranking matrix and correct-vs-generic / correct-vs-wrong deltas use.
  3. Deployable top-1 accuracy (also in evaluate_method2_row): real
     inference-style accuracy using ONLY the raw top-K (no target append) --
     automatically wrong if the true move isn't in the raw top-K, otherwise
     whether the style-reranked argmax among those K picks it.
`topk_target_coverage` stays a separate, representation-independent
diagnostic (the paper's own metric), never conflated with the above.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.style_dataset import (
    StyleDataset,
    build_style_datasets,
    filter_records_by_player,
)
from amadeus_counterpoint.models.candidates import select_candidates, topk_target_coverage
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.style_scoring import score_candidates

GENERIC = "generic"


# ---------------------------------------------------------------------------
# Validation-set reconstruction (method-specific -- see module docstring)
# ---------------------------------------------------------------------------


def build_method1_val_dataset(
    records: list[dict],
    player_id: int,
    train_fraction: float = 0.8,
    split_seed: int = 0,
    balance_seed: int = 0,
) -> StyleDataset:
    """Player P's held-out validation set, exactly as
    scripts/train_method1_player.py constructed it: filter to one player
    FIRST, then split/balance that player's games alone."""
    player_records = filter_records_by_player(records, player_id)
    _, val_dataset, _ = build_style_datasets(
        player_records, train_fraction=train_fraction,
        split_seed=split_seed, balance_seed=balance_seed,
    )
    return val_dataset


def build_method2_val_datasets(
    records: list[dict],
    player_ids: list[int],
    train_fraction: float = 0.8,
    split_seed: int = 0,
    balance_seed: int = 0,
) -> dict[int, StyleDataset]:
    """Per-player held-out validation sets, exactly as
    scripts/train_method2.py constructed its single joint validation set:
    split/balance ALL players' games together in one call, then partition
    the result by player_id. Returns {player_id: StyleDataset}."""
    _, joint_val_dataset, _ = build_style_datasets(
        records, train_fraction=train_fraction,
        split_seed=split_seed, balance_seed=balance_seed,
    )

    per_player: dict[int, StyleDataset] = {}
    for player_id in player_ids:
        decisions = [
            (game_index, ply)
            for game_index, ply in joint_val_dataset.decisions
            if joint_val_dataset.records[game_index]["player_id"] == player_id
        ]
        per_player[player_id] = StyleDataset(joint_val_dataset.records, decisions)
    return per_player


# ---------------------------------------------------------------------------
# Per-batch evaluation
# ---------------------------------------------------------------------------


@dataclass
class RunningStats:
    nll_sum: float = 0.0
    correct: int = 0
    n: int = 0

    def update(self, nll_sum: float, correct: int, n: int) -> None:
        self.nll_sum += nll_sum
        self.correct += correct
        self.n += n

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.n

    @property
    def accuracy(self) -> float:
        return self.correct / self.n


@dataclass
class AccuracyOnly:
    """Deployable accuracy has no well-defined NLL (an uncovered target has
    no probability at all in the raw-K set, not just a low one) -- track
    correct/n only, not RunningStats' NLL fields."""
    correct: int = 0
    n: int = 0

    def update(self, correct: int, n: int) -> None:
        self.correct += correct
        self.n += n

    @property
    def accuracy(self) -> float:
        return self.correct / self.n


def _full_action_metrics(policy_logits: torch.Tensor, batch: dict) -> tuple[float, int, int]:
    """Legal-masked cross-entropy and top-1 accuracy over the full 4352-action
    space -- the metric used for the generic baseline and every Method-1
    column. Returns (nll_sum, correct_count, n)."""
    legal_mask = batch["legal_mask"]
    target = batch["policy_target"]
    masked_logits = policy_logits.masked_fill(~legal_mask, torch.finfo(policy_logits.dtype).min)
    per_example_nll = F.cross_entropy(masked_logits, target, reduction="sum")
    correct = (masked_logits.argmax(dim=-1) == target).sum().item()
    return per_example_nll.item(), correct, target.shape[0]


def evaluate_generic(base: Chessformer, dataloader: DataLoader, device: torch.device) -> RunningStats:
    """The genuine frozen Broadcast base, no personalization, full action
    space: base(x, player_elo, opponent_elo) with each position's own real
    recorded Elos, no override. Descriptive baseline only -- see module
    docstring for why this is never subtracted against a Method-2
    candidate-conditioned NLL."""
    stats = RunningStats()
    base.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            policy_logits, _ = base(batch["x"], batch["player_elo"], batch["opponent_elo"])
            stats.update(*_full_action_metrics(policy_logits, batch))
    return stats


def evaluate_method1_representation(wrapper, dataloader: DataLoader, device: torch.device) -> RunningStats:
    """One column: true player P's positions scored under `wrapper`'s
    z_player (P's own wrapper for the "correct" column, another player's
    wrapper for a "wrong" column) -- wrapper.forward already implements the
    override; this just drives it. Method 1 has no candidate restriction, so
    this is directly comparable to evaluate_generic's full action space."""
    stats = RunningStats()
    wrapper.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            policy_logits, _ = wrapper(batch["x"], batch["player_elo"], batch["opponent_elo"])
            stats.update(*_full_action_metrics(policy_logits, batch))
    return stats


@dataclass
class Method2RowStats:
    coverage: float = 0.0  # raw top-K coverage -- representation-independent
    candidate_matched_generic: RunningStats = field(default_factory=RunningStats)
    per_representation: dict = field(default_factory=dict)  # {player_id: RunningStats}, oracle-expanded
    deployable_per_representation: dict = field(default_factory=dict)  # {player_id: AccuracyOnly}


def evaluate_method2_row(
    base: Chessformer, cnn, table, residual, k: int,
    dataloader: DataLoader, representation_ids: list[int], device: torch.device,
) -> Method2RowStats:
    """One true player's row across every representation column, on three
    distinct metrics -- see module docstring:
      - candidate_matched_generic: the frozen base's OWN candidate-restricted
        logits (no style residual at all), on the identical oracle-expanded
        K(+1) set every style representation is scored on. Directly
        comparable to per_representation's NLLs.
      - per_representation: oracle-expanded (target always present)
        candidate-conditioned NLL/accuracy per style representation --
        mirrors training/validation's own loss exactly.
      - deployable_per_representation: real inference-style top-1 accuracy
        using ONLY the raw top-K (no target append) per style
        representation -- automatically wrong if the true move isn't in the
        raw top-K.
    Plus raw top-K coverage, computed once (representation-independent).
    """
    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()

    row = Method2RowStats(
        per_representation={pid: RunningStats() for pid in representation_ids},
        deployable_per_representation={pid: AccuracyOnly() for pid in representation_ids},
    )
    coverage_sum = 0.0
    coverage_n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k_: v.to(device) for k_, v in batch.items()}
            target = batch["policy_target"]
            n = target.shape[0]

            # Everything in this block is representation-independent: it
            # only depends on the frozen base's own policy for this
            # position, computed once per batch.
            base_logits, _ = base(batch["x"], batch["player_elo"], batch["opponent_elo"])
            masked_logits = base_logits.masked_fill(~batch["legal_mask"], float("-inf"))

            coverage_sum += topk_target_coverage(masked_logits, k, target).sum().item()
            coverage_n += n

            _, candidate_base_logits, candidate_valid, local_target_index = select_candidates(
                masked_logits, k=k, target_index=target
            )
            masked_candidate_base_logits = candidate_base_logits.masked_fill(
                ~candidate_valid, float("-inf")
            )
            base_nll_sum = F.cross_entropy(
                masked_candidate_base_logits, local_target_index, reduction="sum"
            ).item()
            base_correct = (
                masked_candidate_base_logits.argmax(dim=-1) == local_target_index
            ).sum().item()
            row.candidate_matched_generic.update(base_nll_sum, base_correct, n)

            for pid in representation_ids:
                player_id_override = torch.full((n,), pid, dtype=torch.long, device=device)

                # Oracle-expanded: mirrors training/validation's own loss.
                oracle = score_candidates(
                    base, cnn, table, residual,
                    batch["x"], batch["player_elo"], batch["opponent_elo"],
                    player_id_override, batch["legal_mask"], k, target_index=target,
                )
                nll_sum = F.cross_entropy(
                    oracle.candidate_scores, oracle.local_target_index, reduction="sum"
                ).item()
                correct = (
                    oracle.candidate_scores.argmax(dim=-1) == oracle.local_target_index
                ).sum().item()
                row.per_representation[pid].update(nll_sum, correct, n)

                # Deployable: raw top-K only, no target append -- an
                # uncovered target can never be the argmax-selected
                # candidate's global index, so comparing global indices
                # directly gives the right "auto-wrong if uncovered"
                # semantics with no separate coverage check needed.
                deployed = score_candidates(
                    base, cnn, table, residual,
                    batch["x"], batch["player_elo"], batch["opponent_elo"],
                    player_id_override, batch["legal_mask"], k, target_index=None,
                )
                chosen_local = deployed.candidate_scores.argmax(dim=-1, keepdim=True)
                chosen_global = deployed.candidate_indices.gather(1, chosen_local).squeeze(1)
                deployable_correct = (chosen_global == target).sum().item()
                row.deployable_per_representation[pid].update(deployable_correct, n)

    row.coverage = coverage_sum / coverage_n
    return row


# ---------------------------------------------------------------------------
# Matrix assembly: rank/delta arithmetic shared by both methods
# ---------------------------------------------------------------------------


def summarize_row(true_player: int, generic: RunningStats, per_representation: dict[int, RunningStats]) -> dict:
    """One true player's full diagnostic row: generic/correct/mean-wrong
    loss+accuracy, rank of the correct representation among the 8 real
    identities by NLL (1 = lowest NLL = best), and both deltas. Positive
    deltas mean personalization helped (lower NLL than the comparison).

    `generic` is whichever comparator is fair for the caller's metric space
    -- the full-action-space baseline for Method 1, or the
    candidate-matched baseline for Method 2 (see module docstring). This
    function does no method-specific branching; it just needs `generic` and
    `per_representation` to already be on the same scale.
    """
    correct = per_representation[true_player]
    wrong = [stats for pid, stats in per_representation.items() if pid != true_player]

    ranked = sorted(per_representation.items(), key=lambda item: item[1].mean_nll)
    rank = next(i for i, (pid, _) in enumerate(ranked, start=1) if pid == true_player)

    mean_wrong_nll = sum(s.mean_nll for s in wrong) / len(wrong)
    mean_wrong_acc = sum(s.accuracy for s in wrong) / len(wrong)

    return {
        "true_player": true_player,
        "n_positions": correct.n,
        "generic_nll": generic.mean_nll,
        "generic_accuracy": generic.accuracy,
        "correct_nll": correct.mean_nll,
        "correct_accuracy": correct.accuracy,
        "mean_wrong_nll": mean_wrong_nll,
        "mean_wrong_accuracy": mean_wrong_acc,
        "correct_rank_among_8": rank,
        "generic_minus_correct_nll": generic.mean_nll - correct.mean_nll,
        "mean_wrong_minus_correct_nll": mean_wrong_nll - correct.mean_nll,
        "nll_by_representation": {pid: s.mean_nll for pid, s in per_representation.items()},
        "accuracy_by_representation": {pid: s.accuracy for pid, s in per_representation.items()},
    }


def summarize_overall(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "num_players": n,
        "correct_rank_1_count": sum(1 for r in rows if r["correct_rank_among_8"] == 1),
        "mean_correct_rank": sum(r["correct_rank_among_8"] for r in rows) / n,
        "mean_generic_minus_correct_nll": sum(r["generic_minus_correct_nll"] for r in rows) / n,
        "mean_wrong_minus_correct_nll_avg": sum(r["mean_wrong_minus_correct_nll"] for r in rows) / n,
    }
