import chess

from amadeus_counterpoint.data.phase import divide_game, phase_at_ply

# Empirically verified (not hand-derived) against the actual _majors_and_minors /
# _backrank_sparse / _mixedness functions, so each isolates exactly one trigger.

# majorsAndMinors <= 10, isolated: exactly 10, both knights removed each
# side, backrank not sparse (6 pieces left per back rank), mixedness=64.
MAJORS_MINORS_FEN = "r1bqkb1r/8/8/8/8/8/8/R1BQKB1R w KQkq - 0 1"

# majorsAndMinors <= 6, and low enough to also be <=10: almost no pieces left.
LOW_MATERIAL_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"  # majorsAndMinors=0

# backrankSparse only: majorsAndMinors=13 (>10), mixedness=97 (<=150),
# White has only 3 pieces left on rank 1.
BACKRANK_SPARSE_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPPQPPP/R3K2R w KQkq - 0 1"

# mixedness only: majorsAndMinors=14 (>10), backrank not sparse (4 on each back
# rank), mixedness=203 (>150) from the dense symmetric pawn/minor wall.
MIXEDNESS_FEN = "r2qk2r/8/2nbbn2/pppppppp/PPPPPPPP/2NBBN2/8/R2QK2R w KQkq - 0 1"

# endgame: majorsAndMinors=0 (<=6).
ENDGAME_FEN = LOW_MATERIAL_FEN


def _board(fen):
    return chess.Board(fen)


def test_opening_only_game_has_no_middle_boundary():
    # Replay a short quiet opening and confirm every ply stays "opening".
    board = chess.Board()
    sequence = [board.copy(stack=False)]
    for move_uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push_uci(move_uci)
        sequence.append(board.copy(stack=False))
    boards_before_each_ply = sequence[:-1]

    division = divide_game(boards_before_each_ply)

    assert division.middle_ply is None
    assert division.end_ply is None
    for ply in range(len(boards_before_each_ply)):
        assert phase_at_ply(division, ply) == "opening"


def test_majors_and_minors_triggers_middlegame():
    division = divide_game([_board(MAJORS_MINORS_FEN)])
    assert division.middle_ply == 0
    assert division.end_ply is None


def test_backrank_sparse_triggers_middlegame():
    division = divide_game([_board(BACKRANK_SPARSE_FEN)])
    assert division.middle_ply == 0


def test_mixedness_triggers_middlegame():
    division = divide_game([_board(MIXEDNESS_FEN)])
    assert division.middle_ply == 0


def test_majors_and_minors_triggers_endgame_after_middlegame():
    boards = [_board(BACKRANK_SPARSE_FEN), _board(ENDGAME_FEN)]
    division = divide_game(boards)

    assert division.middle_ply == 0
    assert division.end_ply == 1
    assert phase_at_ply(division, 0) == "middlegame"
    assert phase_at_ply(division, 1) == "endgame"


def test_sequential_behavior_finds_first_satisfying_ply_not_any():
    # ply 0: starting position, no trigger. ply 1: backrank-sparse trigger.
    boards = [chess.Board(), _board(BACKRANK_SPARSE_FEN)]
    division = divide_game(boards)

    assert division.middle_ply == 1
    assert phase_at_ply(division, 0) == "opening"
    assert phase_at_ply(division, 1) == "middlegame"


def test_degenerate_middle_equals_end_keeps_only_endgame():
    # A single board satisfying both majorsAndMinors<=10 and <=6 at once.
    boards = [_board(LOW_MATERIAL_FEN)]
    division = divide_game(boards)

    assert division.middle_ply is None
    assert division.end_ply == 0
    assert phase_at_ply(division, 0) == "endgame"


def test_no_middle_boundary_found_keeps_whole_game_opening():
    boards = [chess.Board(), chess.Board()]
    division = divide_game(boards)

    assert division.middle_ply is None
    assert division.end_ply is None
    assert phase_at_ply(division, 0) == "opening"
    assert phase_at_ply(division, 1) == "opening"


def test_num_plies_matches_input_length():
    boards = [chess.Board(), chess.Board(), chess.Board()]
    division = divide_game(boards)
    assert division.num_plies == 3
