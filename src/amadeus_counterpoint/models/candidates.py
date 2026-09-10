"""Tensor-only top-K candidate selection for Method-2 style-residual scoring.

Chess-agnostic by design: this module knows nothing about boards, legal-move
generation, squares, or promotion semantics -- callers are responsible for
masking illegal actions to -inf before calling `select_candidates`, and for
mapping returned indices back to chess moves (see `encoding.py`).
"""

import torch


def select_candidates(
    masked_logits: torch.Tensor,
    k: int,
    target_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select up to `k` (or `k+1`, with a target) candidates per example.

    Args:
        masked_logits: [B, A] float. Illegal actions must already be -inf.
        k: number of top candidates to select.
        target_index: [B] long, the true target action per example. When
            given (training/validation), the candidate set is widened by one
            slot so the target is always present, without duplicating it if
            it was already in the top-k.

    Returns:
        candidate_indices: [B, k] (target_index=None) or [B, k+1] long --
            global action indices into `masked_logits`'s last dimension.
        candidate_base_logits: [B, k] or [B, k+1] float -- gathered from
            `masked_logits` at `candidate_indices`.
        candidate_valid: [B, k] or [B, k+1] bool. False for (a) top-k slots
            beyond the number of finite (legal) entries in that row, and
            (b) the appended k+1'th slot when the target was already inside
            the top-k.
        local_target_index: [B] long -- which column of the returned
            candidate set holds the target -- or None if `target_index` is
            None.
    """
    topk_values, topk_indices = masked_logits.topk(k, dim=-1)
    topk_valid = torch.isfinite(topk_values)

    if target_index is None:
        return topk_indices, topk_values, topk_valid, None

    match = topk_indices == target_index.unsqueeze(-1)  # [B, k]
    already_present = match.any(dim=-1)  # [B]

    # Padding slot: the real target index when it needs appending; otherwise
    # a harmless already-legal placeholder (the top-1 candidate), kept
    # invalid so it never contributes. Using a real index (never a
    # sentinel/out-of-range value) keeps every downstream gather/plane
    # construction well-defined regardless of validity.
    padding_index = torch.where(
        already_present, topk_indices[:, 0], target_index
    ).unsqueeze(-1)
    padding_valid = (~already_present).unsqueeze(-1)
    padding_logits = masked_logits.gather(-1, padding_index)

    candidate_indices = torch.cat([topk_indices, padding_index], dim=-1)
    candidate_base_logits = torch.cat([topk_values, padding_logits], dim=-1)
    candidate_valid = torch.cat([topk_valid, padding_valid], dim=-1)

    # top-k indices are unique, so each row has at most one True; argmax
    # returns that match position (0 when absent, handled below).
    existing_local = match.float().argmax(dim=-1)
    local_target_index = torch.where(
        already_present, existing_local, torch.full_like(existing_local, k)
    )

    return candidate_indices, candidate_base_logits, candidate_valid, local_target_index


def topk_target_coverage(
    masked_logits: torch.Tensor,
    k: int,
    target_index: torch.Tensor,
) -> torch.Tensor:
    """[B] bool: was `target_index` already inside the raw top-k of
    `masked_logits`, before any target-append padding?

    A pure diagnostic, independent of `select_candidates`' K+1 append/pad
    mechanics -- this reports what inference's real, unpadded K-only
    candidate set would have contained.
    """
    _, topk_indices = masked_logits.topk(k, dim=-1)
    return (topk_indices == target_index.unsqueeze(-1)).any(dim=-1)
