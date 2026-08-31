"""Loss functions for Chessformer training."""

import torch
import torch.nn.functional as F


def chessformer_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
    value_coefficient: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined Chessformer policy and value loss.

    Args:
        policy_logits: Raw policy logits of shape [B, 4352].
        value_logits: Raw W/D/L logits of shape [B, 3].
        policy_target: Human move indices of shape [B].
        value_target: Loss/draw/win targets of shape [B].
        legal_mask: Boolean legal-action mask of shape [B, 4352].
        value_coefficient: Weight applied to the auxiliary value loss.

    Returns:
        Total loss, policy loss, and value loss.
    """
    policy_logits = policy_logits.masked_fill(
        ~legal_mask,
        torch.finfo(policy_logits.dtype).min,
    )

    policy_loss = F.cross_entropy(
        policy_logits,
        policy_target,
    )

    value_loss = F.cross_entropy(
        value_logits,
        value_target,
    )

    total_loss = policy_loss + value_coefficient * value_loss

    return total_loss, policy_loss, value_loss