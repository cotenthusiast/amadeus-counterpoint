import torch
import torch.nn as nn

from amadeus_counterpoint.models.chessformer import Chessformer


class PersonalizedChessformer(nn.Module):
    """Method-1 player-specific embedding personalization.

    Wraps a frozen `Chessformer`; the mover's Elo-interpolated embedding is
    replaced by a single trainable per-player vector (`z_player`),
    initialized from that player's own nominal-Elo interpolation. Opponent
    conditioning always goes through the base model's ordinary Elo
    interpolation, unchanged -- personalization is single-sided.
    """

    def __init__(self, base: Chessformer, nominal_elo: float, identity: str):
        super().__init__()

        self.base = base
        self.base.requires_grad_(False)

        # base may already be on a non-CPU device (e.g. moved there by the
        # caller before wrapping) -- the nominal-Elo tensor must be created
        # on that same device, not default to CPU, or interpolate_elo's
        # arithmetic against base's (possibly CUDA) embedding weights raises
        # a device-mismatch error.
        device = next(base.parameters()).device
        with torch.no_grad():
            init = base.interpolate_elo(
                torch.tensor([nominal_elo], dtype=torch.float32, device=device)
            ).squeeze(0).clone()
        self.z_player = nn.Parameter(init)

        self.identity = identity
        self.nominal_elo = nominal_elo

    def forward(self, x, player_elo, opponent_elo):
        # player_elo is accepted for call-signature compatibility with
        # Chessformer.forward but is unused here: z_player already stands
        # in for the mover's Elo-interpolated embedding.
        override = self.z_player.unsqueeze(0).expand(x.shape[0], -1)
        return self.base(x, player_elo, opponent_elo, player_emb_override=override)
