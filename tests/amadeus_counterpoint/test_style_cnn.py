import chess
import torch

from amadeus_counterpoint.encoding import encode_board
from amadeus_counterpoint.models.style_cnn import (
    CANDIDATE_PLANE_CHANNELS,
    MoveStyleCNN,
    board_planes_from_history,
    build_candidate_planes,
)

# Channel layout produced by build_candidate_planes: 0-11 board, 12 from,
# 13 to, 14-17 promotion (Q,R,B,N).
FROM_CHANNEL = 12
TO_CHANNEL = 13
PROMOTION_CHANNELS = slice(14, 18)


# --- board_planes_from_history ---------------------------------------------


def test_board_planes_from_history_matches_encode_board():
    board = chess.Board()
    single = encode_board(board)  # [64, 12]
    history = torch.cat([single] * 8, dim=-1).unsqueeze(0)  # [1, 64, 96]

    planes = board_planes_from_history(history)

    assert planes.shape == (1, 12, 8, 8)
    for square in chess.SQUARES:
        rank, file = chess.square_rank(square), chess.square_file(square)
        for channel in range(12):
            assert planes[0, channel, rank, file].item() == single[square, channel].item()


def test_board_planes_from_history_handles_noncontiguous_slice():
    x = torch.randn(2, 64, 96)  # x[..., -12:] is a non-contiguous view

    planes = board_planes_from_history(x)

    assert planes.shape == (2, 12, 8, 8)
    assert torch.equal(planes, x[..., -12:].reshape(2, 8, 8, 12).permute(0, 3, 1, 2))


# --- build_candidate_planes --------------------------------------------------


def test_build_candidate_planes_shape():
    board_planes = torch.zeros(2, 12, 8, 8)
    from_square = torch.zeros(2, 3, dtype=torch.long)
    to_square = torch.zeros(2, 3, dtype=torch.long)
    promotion_type = torch.full((2, 3), -1, dtype=torch.long)

    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    assert planes.shape == (2, 3, CANDIDATE_PLANE_CHANNELS, 8, 8)
    assert CANDIDATE_PLANE_CHANNELS == 18


def test_build_candidate_planes_repeats_board_across_candidates():
    board_planes = torch.randn(1, 12, 8, 8)
    from_square = torch.zeros(1, 3, dtype=torch.long)
    to_square = torch.zeros(1, 3, dtype=torch.long)
    promotion_type = torch.full((1, 3), -1, dtype=torch.long)

    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    for k in range(3):
        assert torch.equal(planes[0, k, :12], board_planes[0])


def test_build_candidate_planes_from_and_to_one_hot():
    board_planes = torch.zeros(1, 12, 8, 8)
    from_square = torch.tensor([[chess.A1]])
    to_square = torch.tensor([[chess.H8]])
    promotion_type = torch.tensor([[-1]])

    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    from_plane = planes[0, 0, FROM_CHANNEL]
    to_plane = planes[0, 0, TO_CHANNEL]

    assert from_plane.sum().item() == 1.0
    assert from_plane[0, 0].item() == 1.0  # a1 -> rank 0, file 0
    assert to_plane.sum().item() == 1.0
    assert to_plane[7, 7].item() == 1.0  # h8 -> rank 7, file 7


def test_build_candidate_planes_non_promotion_channels_all_zero():
    board_planes = torch.zeros(1, 12, 8, 8)
    from_square = torch.tensor([[chess.E7]])
    to_square = torch.tensor([[chess.E8]])
    promotion_type = torch.tensor([[-1]])

    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    assert planes[0, 0, PROMOTION_CHANNELS].sum().item() == 0.0


def test_build_candidate_planes_promotion_activates_exactly_one_correct_channel():
    board_planes = torch.zeros(1, 12, 8, 8)
    from_square = torch.tensor([[chess.E7, chess.E7, chess.E7, chess.E7]])
    to_square = torch.tensor([[chess.E8, chess.E8, chess.E8, chess.E8]])
    promotion_type = torch.tensor([[0, 1, 2, 3]])  # Q, R, B, N

    planes = build_candidate_planes(board_planes, from_square, to_square, promotion_type)

    dest_rank, dest_file = chess.square_rank(chess.E8), chess.square_file(chess.E8)

    for k, promo_channel in enumerate([0, 1, 2, 3]):
        promo_planes = planes[0, k, PROMOTION_CHANNELS]
        assert promo_planes.sum().item() == 1.0
        assert promo_planes[promo_channel, dest_rank, dest_file].item() == 1.0


# --- MoveStyleCNN ------------------------------------------------------------


def test_move_style_cnn_shape():
    cnn = MoveStyleCNN()
    planes = torch.randn(6, 18, 8, 8)

    out = cnn(planes)

    assert out.shape == (6, 32)


def test_move_style_cnn_finite_output():
    cnn = MoveStyleCNN()
    planes = torch.randn(6, 18, 8, 8)

    out = cnn(planes)

    assert torch.isfinite(out).all()


def test_move_style_cnn_dtype_and_device_sanity():
    cnn = MoveStyleCNN()
    planes = torch.randn(2, 18, 8, 8)

    out = cnn(planes)

    assert out.dtype == planes.dtype
    assert out.device == planes.device


def test_move_style_cnn_configurable_channels_and_style_dim():
    cnn = MoveStyleCNN(in_channels=18, style_dim=16)
    planes = torch.randn(3, 18, 8, 8)

    out = cnn(planes)

    assert out.shape == (3, 16)
