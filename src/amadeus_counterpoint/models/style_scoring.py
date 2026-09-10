"""Method-2 candidate scoring: wires the Stage-2 primitives (candidate
selection, canonical index decoding, plane construction, MoveStyleCNN,
PlayerStyleTable, StyleResidual) into one reusable forward path over a
frozen Chessformer base.

Deliberately separate from both training and generation: `training.style_trainer`
(loss/optimization) and `evaluation.generation.method2_personalized` (inference)
both consume this same scoring function, so the base_logits -> candidates ->
planes -> residual pipeline is defined exactly once.
"""

from dataclasses import dataclass

import torch

from amadeus_counterpoint.encoding import indices_to_canonical_components
from amadeus_counterpoint.models.candidates import select_candidates
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import (
    CANDIDATE_PLANE_CHANNELS,
    MoveStyleCNN,
    board_planes_from_history,
    build_candidate_planes,
)


@dataclass
class StyleScoringOutput:
    """Structured result of one style-residual scoring pass.

    `candidate_scores`' invalid slots (see `candidate_valid`) are already
    masked to -inf, so it is directly usable as a cross-entropy input against
    `local_target_index` with no further masking by the caller.
    """

    candidate_indices: torch.Tensor  # [B, K'] long -- global 0..4351 policy indices
    candidate_scores: torch.Tensor  # [B, K'] float -- base + residual, invalid slots -inf
    candidate_valid: torch.Tensor  # [B, K'] bool
    local_target_index: torch.Tensor | None  # [B] long, or None if target_index was None
    # [B, 4352] float, legal-masked base logits -- kept so callers can also
    # compute topk_target_coverage without recomputing the base forward pass.
    masked_logits: torch.Tensor


def score_candidates(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    x: torch.Tensor,
    player_elo: torch.Tensor,
    opponent_elo: torch.Tensor,
    player_id: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int,
    target_index: torch.Tensor | None = None,
) -> StyleScoringOutput:
    """Score up to `k` (or `k+1`, with a target) legal candidate moves.

    `base` supplies the frozen policy; `cnn`/`table`/`residual` are the
    trainable Method-2 components. `base`'s value-head output is discarded.

    Args:
        x: [B, 64, 96]. player_elo, opponent_elo, player_id: [B]. legal_mask:
            [B, 4352] bool. k: candidate count. target_index: [B] long, the
            true human move -- pass at training/validation time so it is
            guaranteed present in the returned candidate set; omit at
            inference.
    """
    base_logits, _ = base(x, player_elo, opponent_elo)
    masked_logits = base_logits.masked_fill(~legal_mask, float("-inf"))

    candidate_indices, candidate_base_logits, candidate_valid, local_target_index = (
        select_candidates(masked_logits, k=k, target_index=target_index)
    )

    from_square, to_square, promotion_type = indices_to_canonical_components(candidate_indices)

    board_planes = board_planes_from_history(x)
    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    B, K = candidate_indices.shape
    planes = planes.reshape(B * K, CANDIDATE_PLANE_CHANNELS, 8, 8)
    move_features = cnn(planes).reshape(B, K, -1)

    z_u = table(player_id)
    residual_scores = residual(move_features, z_u)

    candidate_scores = candidate_base_logits + residual_scores
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, float("-inf"))

    return StyleScoringOutput(
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        candidate_valid=candidate_valid,
        local_target_index=local_target_index,
        masked_logits=masked_logits,
    )
