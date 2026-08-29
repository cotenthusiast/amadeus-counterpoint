import chess


def create_board() -> chess.Board:
    """Create and return a chess board in the standard starting position."""
    return chess.Board()


def check_legal(board: chess.Board, move: chess.Move) -> bool:
    """Return whether the given move is legal in the current position."""
    return move in board.legal_moves


def make_move(board: chess.Board, move: str) -> None:
    """Parse and apply a legal UCI move to the board.

    Args:
        board: The board to modify.
        move: A move in UCI notation, such as "e2e4".

    Raises:
        chess.IllegalMoveError: If the move is not legal in the current position.
    """
    parsed_move = chess.Move.from_uci(move)

    if not check_legal(board, parsed_move):
        raise chess.IllegalMoveError(
            f"Move {move!r} is not legal in the current position."
        )

    board.push(parsed_move)


def check_end(board: chess.Board) -> chess.Outcome | None:
    """Return the game outcome if the game has ended, otherwise None.

    Claimable draws, such as threefold repetition and the fifty-move rule,
    are treated as game-ending conditions.
    """
    return board.outcome(claim_draw=True)