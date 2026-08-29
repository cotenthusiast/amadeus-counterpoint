from amadeus_counterpoint.chess import (
    check_end,
    create_board,
    make_move,
)
import random
import chess

def main() -> None:
    board = create_board()

    while True:
        mover = board.turn

        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)

        make_move(board, move.uci())

        print(f"{mover} has moved {move}")

        termination = check_end(board)

        if termination is not None:
            break

    if termination.winner is None:
        print(f"The game results in a draw by {termination.termination}.")
    else:
        print(
            f"{termination.winner} has won by "
            f"{termination.termination}!"
        )

if __name__ == "__main__":
    main()