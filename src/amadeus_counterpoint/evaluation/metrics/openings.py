"""Pinned Lichess opening-family classification and distance calculations."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import chess

from amadeus_counterpoint.evaluation.metrics.wdl import ORIENTATIONS, total_variation

UNKNOWN_OPENING_FAMILY = "Unknown"
_TSV_COLUMNS = ("eco", "name", "pgn", "uci", "epd")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def load_opening_index(path: str | Path, commit: str) -> dict[str, object]:
    """Load an explicitly supplied pinned ``dist/all.tsv`` without I/O elsewhere.

    ``commit`` is caller-supplied provenance for the checked-out
    ``lichess-org/chess-openings`` source.  The loader never fetches, caches,
    or substitutes a moving ``latest`` dataset.  The returned tiny mapping
    keeps that commit beside an exact-EPD-to-full-name index.
    """
    if not isinstance(commit, str) or (
        len(commit) != 40 or not all(character in _HEX_DIGITS for character in commit)
    ):
        raise ValueError("opening database commit must be a 40-character hexadecimal SHA")
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(_TSV_COLUMNS):
            raise ValueError("opening TSV must have eco, name, pgn, uci, epd columns")
        names_by_epd = {}
        for row in reader:
            epd = row["epd"]
            if epd in names_by_epd:
                raise ValueError(f"duplicate opening EPD: {epd!r}")
            names_by_epd[epd] = row["name"]
    return {"commit": commit, "epd_to_name": names_by_epd}


def opening_family(name: str) -> str:
    """Collapse an official full name at its one documented colon boundary."""
    return name.partition(":")[0]


def classify_opening_family(
    moves: Sequence[str], opening_index: Mapping[str, object]
) -> str:
    """Replay standard-start UCI moves and return the latest matching family.

    Matching uses ``Board.epd()``, including python-chess's legal
    en-passant normalization.  Each position is considered in replay order;
    this selects the deepest recognized position and naturally handles
    transpositions.  Games with no named position return ``Unknown``.
    """
    try:
        names_by_epd = opening_index["epd_to_name"]
    except KeyError as error:
        raise ValueError("opening index missing 'epd_to_name'") from error
    if not isinstance(names_by_epd, Mapping):
        raise TypeError("opening index has invalid 'epd_to_name'")

    board = chess.Board()
    latest_name = names_by_epd.get(board.epd())
    for move in moves:
        board.push_uci(move)
        name = names_by_epd.get(board.epd())
        if name is not None:
            latest_name = name
    return opening_family(latest_name) if latest_name is not None else UNKNOWN_OPENING_FAMILY


def opening_distribution(
    games: Sequence[Mapping[str, object]], opening_index: Mapping[str, object]
) -> dict[str, float]:
    """Return the opening-family distribution for game records' UCI moves."""
    counts = {}
    for game in games:
        try:
            moves = game["moves"]
        except KeyError as error:
            raise ValueError("game record missing 'moves'") from error
        family = classify_opening_family(moves, opening_index)
        counts[family] = counts.get(family, 0) + 1
    count = len(games)
    return {family: value / count for family, value in counts.items()} if count else {}


def opening_orientation_distance(
    generated_by_orientation: Mapping[str, Sequence[Mapping[str, object]]],
    real_by_orientation: Mapping[str, Sequence[Mapping[str, object]]],
    opening_index: Mapping[str, object],
) -> dict[str, float]:
    """Return opening TV per orientation and their exact unpooled 50/50 mean.

    Both samples must contain at least one game in each orientation because
    Total Variation is undefined for an empty opening distribution.
    """
    distances = {}
    for orientation in ORIENTATIONS:
        try:
            generated = generated_by_orientation[orientation]
            real = real_by_orientation[orientation]
        except KeyError as error:
            raise ValueError(f"missing orientation: {error.args[0]!r}") from error
        if not generated:
            raise ValueError(f"{orientation} generated sample has no games")
        if not real:
            raise ValueError(f"{orientation} real sample has no games")
        distances[orientation] = total_variation(
            opening_distribution(generated, opening_index),
            opening_distribution(real, opening_index),
        )
    return {**distances, "mean": (distances["A_WHITE"] + distances["B_WHITE"]) / 2}
