"""Method-2 style-residual move encoder (phi) and its CNN input construction.

v1 specifies only "a small CNN over a side-to-move-canonicalized board
tensor" for phi -- the exact architecture below (channel counts, kernel
sizes, no pooling/norm/dropout/residual blocks) is this repo's own
implementation choice, not a paper fact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from amadeus_counterpoint.encoding import BOARD_CHANNELS

# 12 board planes + 1 from-square plane + 1 to-square plane + 4 promotion
# planes (Q, R, B, N, matching encoding.py's _PROMOTION_PIECE_TO_INDEX order).
CANDIDATE_PLANE_CHANNELS = BOARD_CHANNELS + 1 + 1 + 4
_NUM_PROMOTION_TYPES = 4


def board_planes_from_history(x: torch.Tensor) -> torch.Tensor:
    """Extract CNN-ready board planes from a Chessformer history tensor.

    `x` is `[B, 64, 96]`; its trailing 12 channels are exactly the current
    (most recent) position's `encode_board()` output, already canonicalized
    to the side-to-move-as-White frame -- no new board encoding is needed.

    `x[..., -12:]` is a non-contiguous slice of a larger tensor, so `.view()`
    would raise; `.reshape()` (which copies when necessary) is used instead.
    """
    B = x.shape[0]
    board = x[..., -BOARD_CHANNELS:]              # [B, 64, 12], non-contiguous
    board = board.reshape(B, 8, 8, BOARD_CHANNELS)  # square index = rank*8+file
    return board.permute(0, 3, 1, 2)               # [B, 12, 8, 8]


def build_candidate_planes(
    board_planes: torch.Tensor,
    from_square: torch.Tensor,
    to_square: torch.Tensor,
    promotion_type: torch.Tensor,
) -> torch.Tensor:
    """Assemble phi's per-candidate CNN input.

    Args:
        board_planes: [B, 12, 8, 8].
        from_square, to_square, promotion_type: [B, K'] each, as returned by
            `encoding.indices_to_canonical_components` (promotion_type in
            {0,1,2,3} for Q/R/B/N, or -1 for non-promotion moves).

    Returns:
        [B, K', 18, 8, 8]. For a promotion candidate, exactly one of the
        four promotion channels has a single 1 at the destination square;
        for a non-promotion candidate, all four are entirely zero.
    """
    B, K = from_square.shape
    dtype = board_planes.dtype

    board_expanded = board_planes.unsqueeze(1).expand(-1, K, -1, -1, -1)  # [B,K,12,8,8]

    from_plane = F.one_hot(from_square, 64).to(dtype).view(B, K, 1, 8, 8)
    to_plane = F.one_hot(to_square, 64).to(dtype).view(B, K, 1, 8, 8)

    is_promo = (promotion_type >= 0).to(dtype).unsqueeze(-1)               # [B,K,1]
    type_onehot = F.one_hot(promotion_type.clamp(min=0), _NUM_PROMOTION_TYPES).to(dtype)
    type_onehot = type_onehot * is_promo                                   # zero for non-promotion

    dest_onehot = F.one_hot(to_square, 64).to(dtype)                       # [B,K,64]
    # [B,K,4,1] * [B,K,1,64] -> [B,K,4,64]
    promo_planes = type_onehot.unsqueeze(-1) * dest_onehot.unsqueeze(-2)
    promo_planes = promo_planes.view(B, K, _NUM_PROMOTION_TYPES, 8, 8)

    return torch.cat([board_expanded, from_plane, to_plane, promo_planes], dim=2)


class MoveStyleCNN(nn.Module):
    """phi(position, candidate_move) -> R^32.

    Fixed architecture (this repo's own choice, see module docstring):
    three same-padding 3x3 convs (no downsampling), GELU after each, then a
    single linear projection to `style_dim`. No pooling, normalization,
    dropout, residual connections, or trailing activation.
    """

    def __init__(self, in_channels: int = CANDIDATE_PLANE_CHANNELS, style_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, style_dim),
        )

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        """[N, 18, 8, 8] -> [N, style_dim]."""
        return self.net(planes)
