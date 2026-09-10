"""Method-2 per-player style vectors and the frozen-base style residual.

No generic/unknown-player row and no embedding dropout, by design -- for
this fixed-player-set experiment, an unpersonalized player is simply served
by the plain (non-wrapped) Chessformer, not by a fallback row here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerStyleTable(nn.Module):
    """One learned [style_dim] row per player. `num_players` is caller-supplied,
    not hard-coded -- the eventual 8-player experiment is just one instantiation."""

    def __init__(self, num_players: int, style_dim: int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(num_players, style_dim)

    def forward(self, player_id: torch.Tensor) -> torch.Tensor:
        """[B] long -> [B, style_dim]."""
        return self.embeddings(player_id)


class StyleResidual(nn.Module):
    """score(u, p, c) - base(p, c) term: s * sqrt(d) * (phi_hat . z_hat), v1 Eq. (2)."""

    def __init__(self, style_dim: int = 32, init_s: float = 0.17):
        super().__init__()
        self.style_dim = style_dim
        self.s = nn.Parameter(torch.tensor(float(init_s)))

    def forward(self, move_features: torch.Tensor, z_u: torch.Tensor) -> torch.Tensor:
        """move_features [B,K,style_dim], z_u [B,style_dim] -> residual [B,K]."""
        phi_hat = F.normalize(move_features, dim=-1)
        z_hat = F.normalize(z_u, dim=-1)
        compatibility = (phi_hat * z_hat.unsqueeze(1)).sum(dim=-1)
        return self.s * math.sqrt(self.style_dim) * compatibility
