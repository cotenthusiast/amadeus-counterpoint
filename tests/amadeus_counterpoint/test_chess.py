import chess
import pytest

from amadeus_counterpoint.chess import (
    check_end,
    check_legal,
    create_board,
    make_move,
)

CHECKMATE_FEN = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"
STALEMATE_FEN = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"
INSUFFICIENT_MATERIAL_FEN = "8/8/8/8/8/8/5k2/7K w - - 0 1"
SEVENTYFIVE_MOVE_FEN = "8/8/8/8/8/8/5k2/R6K w - - 150 76"
FIFTY_MOVE_FEN = "8/8/8/8/8/8/5k2/R6K w - - 100 51"

def create_fivefold_repetition_board() -> chess.Board:
    """Create a board whose current position has occurred five times."""
    board = chess.Board()

    cycle = (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    )

    # The initial position is the first occurrence. Four complete cycles
    # return to it four more times, producing fivefold repetition.
    for _ in range(4):
        for move in cycle:
            board.push_uci(move)

    return board


def create_threefold_repetition_board() -> chess.Board:
    """Create a board where a draw can be claimed by threefold repetition."""
    board = chess.Board()

    cycle = (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    )

    # The initial position plus two returns gives three occurrences.
    for _ in range(2):
        for move in cycle:
            board.push_uci(move)

    return board


def test_create_board():
    """Test that a new board uses the standard chess starting position."""
    assert create_board().fen() == chess.STARTING_FEN


def test_check_legal_move():
    """Test that a legal move is recognized from the starting position."""
    board = create_board()
    move = chess.Move.from_uci("e2e4")

    assert check_legal(board, move)


def test_check_illegal_move():
    """Test that an illegal move is rejected from the starting position."""
    board = create_board()
    move = chess.Move.from_uci("e2e5")

    assert not check_legal(board, move)


def test_make_move():
    """Test that making a legal move updates the board."""
    board = create_board()

    make_move(board, "e2e4")

    assert board.piece_at(chess.E2) is None
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert board.turn == chess.BLACK


def test_make_illegal_move():
    """Test that attempting an illegal move raises an appropriate error."""
    board = create_board()

    with pytest.raises(chess.IllegalMoveError):
        make_move(board, "e2e5")


def test_ongoing_game():
    """Test that an ongoing game has no outcome."""
    assert check_end(create_board()) is None


def test_checkmate():
    """Test detection of checkmate and the winning color."""
    result = check_end(chess.Board(CHECKMATE_FEN))

    assert result is not None
    assert result.termination == chess.Termination.CHECKMATE
    assert result.winner == chess.WHITE


def test_stalemate():
    """Test detection of a stalemate draw."""
    result = check_end(chess.Board(STALEMATE_FEN))

    assert result is not None
    assert result.termination == chess.Termination.STALEMATE
    assert result.winner is None


def test_insufficient_material():
    """Test detection of a draw caused by insufficient mating material."""
    result = check_end(chess.Board(INSUFFICIENT_MATERIAL_FEN))

    assert result is not None
    assert result.termination == chess.Termination.INSUFFICIENT_MATERIAL
    assert result.winner is None


def test_seventyfive_move_rule():
    """Test detection of the automatic seventy-five-move draw."""
    result = check_end(chess.Board(SEVENTYFIVE_MOVE_FEN))

    assert result is not None
    assert result.termination == chess.Termination.SEVENTYFIVE_MOVES
    assert result.winner is None


def test_fifty_move_rule():
    """Test detection of a claimable fifty-move draw."""
    result = check_end(chess.Board(FIFTY_MOVE_FEN))

    assert result is not None
    assert result.termination == chess.Termination.FIFTY_MOVES
    assert result.winner is None


def test_fivefold_repetition():
    """Test detection of an automatic fivefold-repetition draw."""
    result = check_end(create_fivefold_repetition_board())

    assert result is not None
    assert result.termination == chess.Termination.FIVEFOLD_REPETITION
    assert result.winner is None


def test_threefold_repetition():
    """Test detection of a claimable threefold-repetition draw."""
    result = check_end(create_threefold_repetition_board())

    assert result is not None
    assert result.termination == chess.Termination.THREEFOLD_REPETITION
    assert result.winner is None