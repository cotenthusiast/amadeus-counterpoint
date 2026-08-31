import chess
import pytest
import torch

from amadeus_counterpoint.encoding import (
    BASE_POLICY_SIZE,
    BOARD_CHANNELS,
    HISTORY_LENGTH,
    POLICY_SIZE,
    PROMOTION_POLICY_SIZE,
    encode_board,
    encode_history,
    legal_move_mask,
    move_to_policy_index,
    policy_index_to_move,
)

# A board with a lone White pawn on e7, ready to promote, no checks involved.
PROMOTION_FEN = "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"

# A board where the e7 pawn promotes by capturing on b8 (distinct from/to files).
CAPTURE_PROMOTION_FEN = "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"

# A board where the analogous Black pawn promotes on rank 1.
BLACK_PROMOTION_FEN = "4k3/8/8/8/8/8/4p3/4K3 b - - 0 1"

CASTLING_FEN_WHITE = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
CASTLING_FEN_BLACK = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"


def _board_after(moves):
    """Build a board by pushing a sequence of UCI moves from the start."""
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def _game_boards(count):
    """Return `count` sequential board snapshots, oldest first."""
    board = chess.Board()
    boards = [board.copy()]

    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
    for move in moves[: count - 1]:
        board.push_uci(move)
        boards.append(board.copy())

    return boards


# --- constants -----------------------------------------------------------


def test_constants_match_expected_values():
    assert BOARD_CHANNELS == 12
    assert HISTORY_LENGTH == 8
    assert BASE_POLICY_SIZE == 4096
    assert PROMOTION_POLICY_SIZE == 256
    assert POLICY_SIZE == 4352


# --- A. board encoding -----------------------------------------------------


def test_encode_board_shape_and_dtype():
    encoded = encode_board(chess.Board())

    assert encoded.shape == (64, 12)
    assert encoded.dtype == torch.float32


def test_exactly_32_occupied_squares_in_starting_position():
    encoded = encode_board(chess.Board())

    assert encoded.sum().item() == 32.0


def test_e2_white_pawn_is_channel_0():
    encoded = encode_board(chess.Board())

    assert encoded[chess.E2, 0] == 1.0


def test_e7_black_pawn_is_channel_6():
    encoded = encode_board(chess.Board())

    assert encoded[chess.E7, 6] == 1.0


def test_king_channels_in_starting_position():
    encoded = encode_board(chess.Board())

    assert encoded[chess.E1, 5] == 1.0
    assert encoded[chess.E8, 11] == 1.0


def test_empty_square_is_all_zero():
    encoded = encode_board(chess.Board())

    assert torch.all(encoded[chess.E4] == 0.0)


def test_encode_board_does_not_mutate_white_to_move_board():
    board = chess.Board()
    fen_before = board.fen()

    encode_board(board)

    assert board.fen() == fen_before


def test_encode_board_does_not_mutate_black_to_move_board():
    board = _board_after(["e2e4"])
    fen_before = board.fen()

    encode_board(board)

    assert board.fen() == fen_before


# --- B. Black canonicalization ---------------------------------------------


def test_black_to_move_encoding_matches_manual_mirror():
    board = _board_after(["e2e4"])
    assert board.turn == chess.BLACK

    mirrored = board.mirror()
    assert mirrored.turn == chess.WHITE

    # White's pawn on e4 becomes, from the canonical always-White perspective,
    # a Black pawn on e5.
    assert mirrored.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)

    encoded = encode_board(board)
    assert encoded[chess.E5, 6] == 1.0
    assert torch.equal(encoded, encode_board(mirrored))


def test_black_to_move_king_color_and_square_transform():
    board = _board_after(["e2e4"])

    mirrored = board.mirror()
    # White's untouched king on e1 becomes a Black king on e8.
    assert mirrored.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK)

    encoded = encode_board(board)
    assert encoded[chess.E8, 11] == 1.0


# --- C. history --------------------------------------------------------


def test_single_board_history_pads_to_eight_copies():
    board = chess.Board()

    history = encode_history([board])
    single = encode_board(board)
    expected = torch.cat([single] * 8, dim=-1)

    assert history.shape == (64, 96)
    assert history.dtype == torch.float32
    assert torch.equal(history, expected)


def test_two_board_history_prepends_earliest_copies():
    board_a = chess.Board()
    board_b = _board_after(["e2e4"])

    history = encode_history([board_a, board_b])

    enc_a = encode_board(board_a)
    enc_b = encode_board(board_b)
    expected = torch.cat([enc_a] * 7 + [enc_b], dim=-1)

    assert torch.equal(history, expected)


def test_eight_board_history_has_no_padding():
    boards = _game_boards(8)

    history = encode_history(boards)
    expected = torch.cat([encode_board(b) for b in boards], dim=-1)

    assert torch.equal(history, expected)


def test_more_than_eight_boards_keeps_newest_eight():
    boards = _game_boards(10)

    history = encode_history(boards)
    expected = torch.cat([encode_board(b) for b in boards[-8:]], dim=-1)

    assert torch.equal(history, expected)


def test_empty_history_raises_value_error():
    with pytest.raises(ValueError):
        encode_history([])


# --- D. ordinary policy index -----------------------------------------------


def test_ordinary_index_pawn_push_matches_formula():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    index = move_to_policy_index(move, board)

    assert index == chess.E2 * 64 + chess.E4


def test_ordinary_index_knight_move_matches_formula():
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")

    index = move_to_policy_index(move, board)

    assert index == chess.G1 * 64 + chess.F3


# --- E. promotion policy index -----------------------------------------------


def test_promotion_index_queen():
    board = chess.Board(PROMOTION_FEN)
    move = chess.Move.from_uci("e7e8q")

    index = move_to_policy_index(move, board)

    assert index == 4096 + 4 * 32 + 4 * 4 + 0
    assert index == 4240


def test_promotion_index_rook():
    board = chess.Board(PROMOTION_FEN)
    index = move_to_policy_index(chess.Move.from_uci("e7e8r"), board)

    assert index == 4096 + 4 * 32 + 4 * 4 + 1


def test_promotion_index_bishop():
    board = chess.Board(PROMOTION_FEN)
    index = move_to_policy_index(chess.Move.from_uci("e7e8b"), board)

    assert index == 4096 + 4 * 32 + 4 * 4 + 2


def test_promotion_index_knight():
    board = chess.Board(PROMOTION_FEN)
    index = move_to_policy_index(chess.Move.from_uci("e7e8n"), board)

    assert index == 4096 + 4 * 32 + 4 * 4 + 3


def test_promotion_index_distinct_from_to_files():
    board = chess.Board(CAPTURE_PROMOTION_FEN)
    move = chess.Move.from_uci("a7b8n")

    index = move_to_policy_index(move, board)

    assert index == 4096 + 0 * 32 + 1 * 4 + 3


# --- F. Black move canonicalization ---------------------------------------


def test_black_ordinary_move_canonicalizes_like_white_equivalent():
    board = _board_after(["e2e4"])
    black_move = chess.Move.from_uci("e7e5")

    index = move_to_policy_index(black_move, board)

    white_board = chess.Board()
    expected = move_to_policy_index(chess.Move.from_uci("e2e4"), white_board)

    assert index == expected == chess.E2 * 64 + chess.E4


def test_black_castling_canonicalizes_like_white_equivalent():
    black_board = chess.Board(CASTLING_FEN_BLACK)
    index = move_to_policy_index(chess.Move.from_uci("e8g8"), black_board)

    white_board = chess.Board(CASTLING_FEN_WHITE)
    expected = move_to_policy_index(chess.Move.from_uci("e1g1"), white_board)

    assert index == expected == chess.E1 * 64 + chess.G1


def test_black_promotion_canonicalizes_to_same_index_as_white_equivalent():
    board = chess.Board(BLACK_PROMOTION_FEN)
    move = chess.Move.from_uci("e2e1q")

    index = move_to_policy_index(move, board)

    assert index == 4240  # same canonical e7e8q index as test_promotion_index_queen


# --- G. round trip -----------------------------------------------------


ROUND_TRIP_BOARDS = [
    chess.Board(),
    _board_after(["e2e4"]),
    chess.Board(CASTLING_FEN_WHITE),
    chess.Board(CASTLING_FEN_BLACK),
    chess.Board(PROMOTION_FEN),
    chess.Board(CAPTURE_PROMOTION_FEN),
    chess.Board(BLACK_PROMOTION_FEN),
]


@pytest.mark.parametrize("board", ROUND_TRIP_BOARDS, ids=lambda b: b.fen())
def test_round_trip_every_legal_move(board):
    for move in board.legal_moves:
        index = move_to_policy_index(move, board)
        decoded = policy_index_to_move(index, board)

        assert decoded == move


# --- H. legal move mask --------------------------------------------------


def test_start_position_mask_has_twenty_legal_moves():
    board = chess.Board()

    mask = legal_move_mask(board)

    assert mask.shape == (4352,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 20


def test_mask_marks_every_legal_move_true_with_no_duplicates():
    board = chess.Board()

    mask = legal_move_mask(board)
    indices = [move_to_policy_index(move, board) for move in board.legal_moves]

    assert len(indices) == len(set(indices))
    assert all(mask[i] for i in indices)
    assert mask.sum().item() == len(indices)


def test_mask_black_to_move_position():
    board = _board_after(["e2e4"])

    mask = legal_move_mask(board)
    indices = [move_to_policy_index(move, board) for move in board.legal_moves]

    assert len(indices) == len(set(indices))
    assert mask.sum().item() == len(indices)


def test_mask_promotion_position():
    board = chess.Board(PROMOTION_FEN)

    mask = legal_move_mask(board)
    legal_count = sum(1 for _ in board.legal_moves)

    assert mask.sum().item() == legal_count


# --- I. input validation --------------------------------------------------


def test_policy_index_to_move_rejects_negative_index():
    with pytest.raises(ValueError):
        policy_index_to_move(-1, chess.Board())


def test_policy_index_to_move_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        policy_index_to_move(4352, chess.Board())


def test_move_to_policy_index_rejects_unsupported_promotion_piece():
    board = chess.Board(PROMOTION_FEN)
    bad_move = chess.Move(chess.E7, chess.E8, promotion=chess.KING)

    with pytest.raises(ValueError):
        move_to_policy_index(bad_move, board)
