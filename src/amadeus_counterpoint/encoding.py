"""Deterministic conversions between python-chess state and Chessformer tensors.

Canonicalization follows the public Maia-3 convention: whichever side is to
move is always presented to the model as White. `chess.Board.mirror()`
handles the vertical flip and color swap; moves are canonicalized by
mirroring their squares the same way.
"""

from typing import Sequence

import chess
import torch

BOARD_CHANNELS = 12
HISTORY_LENGTH = 8
BASE_POLICY_SIZE = 4096
PROMOTION_POLICY_SIZE = 256
POLICY_SIZE = BASE_POLICY_SIZE + PROMOTION_POLICY_SIZE

_PROMOTION_PIECE_TO_INDEX = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}
_INDEX_TO_PROMOTION_PIECE = {v: k for k, v in _PROMOTION_PIECE_TO_INDEX.items()}


def _piece_channel(piece: chess.Piece) -> int:
    """Map a piece to its 0..11 channel (white pawn..king, then black)."""
    offset = 0 if piece.color == chess.WHITE else 6
    return (piece.piece_type - 1) + offset


def encode_board(board: chess.Board) -> torch.Tensor:
    """Encode a board into a [64, 12] one-hot piece tensor.

    The board is canonicalized so the side to move is always seen as White,
    matching Maia-3: `board.mirror()` is used when Black is to move. The
    input board is not mutated.
    """
    canonical = board if board.turn == chess.WHITE else board.mirror()

    tensor = torch.zeros(64, BOARD_CHANNELS, dtype=torch.float32)
    for square, piece in canonical.piece_map().items():
        tensor[square, _piece_channel(piece)] = 1.0

    return tensor


def encode_history(
    boards: Sequence[chess.Board],
    history_length: int = HISTORY_LENGTH,
) -> torch.Tensor:
    """Encode ordered (oldest -> newest) board history into [64, 12 * history_length].

    Each board is canonicalized independently via `encode_board`, using its
    own `board.turn`. Fewer than `history_length` boards are padded by
    prepending copies of the earliest available encoded position, matching
    Maia-3's inference padding convention. More than `history_length` boards
    are truncated to the most recent ones.
    """
    if not boards:
        raise ValueError("encode_history requires at least one board")

    encoded = [encode_board(b) for b in boards[-history_length:]]

    if len(encoded) < history_length:
        pad = [encoded[0]] * (history_length - len(encoded))
        encoded = pad + encoded

    return torch.cat(encoded, dim=-1)


def _mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move's squares vertically, preserving promotion/drop."""
    return chess.Move(
        from_square=chess.square_mirror(move.from_square),
        to_square=chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def _promotion_piece_index(promotion: int) -> int:
    try:
        return _PROMOTION_PIECE_TO_INDEX[promotion]
    except KeyError:
        raise ValueError(f"unsupported promotion piece: {promotion!r}") from None


def move_to_policy_index(move: chess.Move, board: chess.Board) -> int:
    """Map a move, from `board`'s perspective, to its 0..4351 policy index.

    `board` supplies whose turn it is, so the move can be canonicalized the
    same way as `encode_board`. `board` is not mutated.
    """
    canonical = move if board.turn == chess.WHITE else _mirror_move(move)

    if canonical.promotion is not None:
        from_file = chess.square_file(canonical.from_square)
        to_file = chess.square_file(canonical.to_square)
        piece_index = _promotion_piece_index(canonical.promotion)
        return BASE_POLICY_SIZE + from_file * 32 + to_file * 4 + piece_index

    return canonical.from_square * 64 + canonical.to_square


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of `move_to_policy_index`: decode an index into `board`'s perspective.

    The decoded move is not guaranteed to be legal; legality is the caller's
    responsibility (see `legal_move_mask`). `board` is not mutated.
    """
    if not (0 <= index < POLICY_SIZE):
        raise ValueError(f"policy index {index} out of range [0, {POLICY_SIZE})")

    if index < BASE_POLICY_SIZE:
        canonical = chess.Move(from_square=index // 64, to_square=index % 64)
    else:
        promo_index = index - BASE_POLICY_SIZE
        from_file, remainder = divmod(promo_index, 32)
        to_file, piece_index = divmod(remainder, 4)
        canonical = chess.Move(
            from_square=chess.square(from_file, 6),
            to_square=chess.square(to_file, 7),
            promotion=_INDEX_TO_PROMOTION_PIECE[piece_index],
        )

    return canonical if board.turn == chess.WHITE else _mirror_move(canonical)


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a [4352] bool mask marking every legal move's policy index True."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_policy_index(move, board)] = True
    return mask
