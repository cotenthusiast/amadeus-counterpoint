"""Exact port of lichess-org/scalachess's Divider (core/src/main/scala/Divider.scala,
verified against the live master source earlier in this session).

This is a whole-game, sequential divider, not a per-position classifier:
`divide_game` scans a game's boards in order and finds at most two global
ply boundaries (middle_ply, end_ply); `phase_at_ply` then labels any single
ply by which of the resulting three ranges it falls in.
"""

from dataclasses import dataclass

import chess

_SMALL_SQUARE = 0x0303  # a 2x2 block: bits 0,1,8,9 (a1,b1,a2,b2)

# 49 overlapping 2x2 windows tiling the board. Built with y as the OUTER
# loop and x as the INNER loop, matching scalachess's own
# `for y <- 0 to 6; x <- 0 to 6 yield ...` -- this ordering matters because
# `_mixedness` below recovers y from the flat index via `i // 7`.
_MIXEDNESS_REGIONS = [_SMALL_SQUARE << (x + 8 * y) for y in range(7) for x in range(7)]


def _majors_and_minors(board: chess.Board) -> int:
    """Count of every occupied queen/rook/bishop/knight, both colors combined."""
    non_pawn_non_king = board.occupied & ~(board.kings | board.pawns) & chess.BB_ALL
    return chess.popcount(non_pawn_non_king)


def _backrank_sparse(board: chess.Board) -> bool:
    """True once White has developed most of rank 1, or Black most of rank 8."""
    white_on_rank_1 = chess.popcount(chess.BB_RANK_1 & board.occupied_co[chess.WHITE])
    black_on_rank_8 = chess.popcount(chess.BB_RANK_8 & board.occupied_co[chess.BLACK])
    return white_on_rank_1 < 4 or black_on_rank_8 < 4


def _mixedness_score(y: int, white_count: int, black_count: int) -> int:
    """Exact port of scalachess's `score(y, white, black)` switch table."""
    if white_count == 0:
        if black_count == 1:
            return 1 + y
        if black_count == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black_count == 3:
            return 3 + (7 - y) if y < 7 else 0
        if black_count == 4:
            return 3 + (7 - y) if y < 7 else 0
        return 0

    if white_count == 1:
        if black_count == 0:
            return 1 + (8 - y)
        if black_count == 1:
            return 5 + abs(4 - y)
        if black_count == 2:
            return 4 + (7 - y)
        if black_count == 3:
            return 5 + (7 - y)
        return 0

    if white_count == 2:
        if black_count == 0:
            return 2 + (y - 2) if y > 2 else 0
        if black_count == 1:
            return 4 + (y - 1)
        if black_count == 2:
            return 7
        return 0

    if white_count == 3:
        if black_count == 0:
            return 3 + (y - 1) if y > 1 else 0
        if black_count == 1:
            return 5 + (y - 1)
        return 0

    if white_count == 4:
        if black_count == 0:
            return 3 + (y - 1) if y > 1 else 0
        return 0

    return 0


def _mixedness(board: chess.Board) -> int:
    total = 0
    for index, region in enumerate(_MIXEDNESS_REGIONS):
        y = index // 7 + 1
        white_count = chess.popcount(board.occupied_co[chess.WHITE] & region)
        black_count = chess.popcount(board.occupied_co[chess.BLACK] & region)
        total += _mixedness_score(y, white_count, black_count)
    return total


@dataclass
class PhaseDivision:
    middle_ply: int | None
    end_ply: int | None
    num_plies: int


def divide_game(boards: list[chess.Board]) -> PhaseDivision:
    """Find the middlegame/endgame ply boundaries for one game.

    `boards[i]` must be the real (uncanonicalized) board position before
    ply `i` is played, for every ply of the game in order -- exactly what
    `_majors_and_minors`/`_backrank_sparse`/`_mixedness` need, since they
    are White/Black-absolute, not side-to-move-relative.

    middle_ply is the first ply satisfying majorsAndMinors<=10, or
    backrankSparse, or mixedness>150. end_ply, only searched if middle_ply
    was found, is the first ply satisfying majorsAndMinors<=6. If both are
    found and they land on the same ply (or end_ply is not after
    middle_ply), the degenerate middlegame is discarded and only end_ply is
    kept. If middle_ply is never found, the whole game stays "opening".
    """
    middle_ply = None
    for ply, board in enumerate(boards):
        if _majors_and_minors(board) <= 10 or _backrank_sparse(board) or _mixedness(board) > 150:
            middle_ply = ply
            break

    end_ply = None
    if middle_ply is not None:
        for ply, board in enumerate(boards):
            if _majors_and_minors(board) <= 6:
                end_ply = ply
                break

    if middle_ply is not None and end_ply is not None and middle_ply >= end_ply:
        middle_ply = None

    return PhaseDivision(middle_ply=middle_ply, end_ply=end_ply, num_plies=len(boards))


def phase_at_ply(division: PhaseDivision, ply: int) -> str:
    """Label a single ply as "opening", "middlegame", or "endgame" using an
    already-computed `PhaseDivision` -- range membership only, not a fresh
    per-position evaluation of the three trigger conditions.

    Checks end_ply before middle_ply deliberately: in the degenerate case
    (middle_ply discarded because it coincided with end_ply, so middle_ply
    is None here but end_ply is not), this correctly labels plies before
    end_ply as "opening" and everything from end_ply onward as "endgame",
    with no middlegame region at all -- without needing to separately
    remember the discarded boundary value.
    """
    if division.end_ply is not None and ply >= division.end_ply:
        return "endgame"
    if division.middle_ply is not None and ply >= division.middle_ply:
        return "middlegame"
    return "opening"
