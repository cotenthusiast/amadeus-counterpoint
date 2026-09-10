"""Stream raw Broadcast PGN files into compact per-game style-training
records, one per eligible (exactly-one-target) game.

Reuses `iter_pgn_games`/`game_to_record` from `preprocess.py` unmodified --
the same validity rule population training already uses (Standard variant,
no custom FEN/SetUp, both Elo fields parse, a terminal result, at least one
move). Target-vs-target games are excluded completely: they are sealed for
later evaluation and must never contribute even one ply here.
"""

import json
from pathlib import Path
from typing import TypedDict

from amadeus_counterpoint.data.broadcast_targets import match_target
from amadeus_counterpoint.data.preprocess import game_to_record, iter_pgn_games


class StyleGameRecord(TypedDict):
    """One eligible (exactly-one-target) game, compact enough to keep many
    of them in memory or in a small JSON file. Move-by-move encoding is
    done lazily later, in `style_dataset.StyleDataset`."""

    player_id: int
    mover_color: str  # "white" or "black" -- which side the target played
    white_elo: int
    black_elo: int
    result: str
    moves: list[str]


def classify_game(
    white_name: str,
    black_name: str,
    white_fide_id: str,
    black_fide_id: str,
    targets: list[dict],
) -> tuple[str, int | None, str | None]:
    """Classify one game by how many of the 8 targets played in it.

    Returns (classification, player_id, mover_color):
      "zero_target"       -- (zero_target, None, None)
      "one_target"        -- (one_target, that player's player_id, "white" or "black")
      "target_vs_target"  -- (target_vs_target, None, None), always excluded
    """
    white_player_id = match_target(white_name, white_fide_id, targets)
    black_player_id = match_target(black_name, black_fide_id, targets)

    if white_player_id is not None and black_player_id is not None:
        return "target_vs_target", None, None
    if white_player_id is not None:
        return "one_target", white_player_id, "white"
    if black_player_id is not None:
        return "one_target", black_player_id, "black"
    return "zero_target", None, None


def iter_target_game_records(
    pgn_paths,
    targets: list[dict],
    stats: dict | None = None,
    quarantined_game_urls: frozenset[str] | None = None,
):
    """Stream every eligible one-target game across `pgn_paths` as a
    `StyleGameRecord`.

    Args:
        pgn_paths: Paths to raw Broadcast PGN files, scanned in order.
        targets: The loaded `configs/broadcast_targets.json` targets list.
        stats: If given, incremented in place with counts for
            "games_scanned", "zero_target_games", "one_target_games",
            "target_vs_target_games", "quarantined_games", and
            "invalid_games" (a one-target game that failed
            `game_to_record`'s validity rule).
        quarantined_game_urls: GameURLs to exclude even when one side
            resolves to a target (see
            `broadcast_targets.load_quarantined_game_urls` and
            `configs/quarantined_broadcast_games.json`) -- these are games
            where the OTHER side's identity is not verified to the standard
            `configs/broadcast_targets.json` itself requires, so treating
            them as ordinary one-target data would risk silently including
            an actual target-vs-target game. Checked before classification;
            defaults to excluding none, for backward compatibility.
    """
    if quarantined_game_urls is None:
        quarantined_game_urls = frozenset()

    if stats is not None:
        stats.setdefault("games_scanned", 0)
        stats.setdefault("zero_target_games", 0)
        stats.setdefault("one_target_games", 0)
        stats.setdefault("target_vs_target_games", 0)
        stats.setdefault("quarantined_games", 0)
        stats.setdefault("invalid_games", 0)

    for path in pgn_paths:
        for game in iter_pgn_games(path):
            if stats is not None:
                stats["games_scanned"] += 1

            headers = game.headers

            if headers.get("GameURL") in quarantined_game_urls:
                if stats is not None:
                    stats["quarantined_games"] += 1
                continue

            white_name = headers.get("White", "")
            black_name = headers.get("Black", "")
            white_fide_id = headers.get("WhiteFideId", "")
            black_fide_id = headers.get("BlackFideId", "")

            classification, player_id, mover_color = classify_game(
                white_name, black_name, white_fide_id, black_fide_id, targets
            )

            if classification == "zero_target":
                if stats is not None:
                    stats["zero_target_games"] += 1
                continue

            if classification == "target_vs_target":
                if stats is not None:
                    stats["target_vs_target_games"] += 1
                continue

            if stats is not None:
                stats["one_target_games"] += 1

            record = game_to_record(game)
            if record is None:
                if stats is not None:
                    stats["invalid_games"] += 1
                continue

            yield {
                "player_id": player_id,
                "mover_color": mover_color,
                "white_elo": record["white_elo"],
                "black_elo": record["black_elo"],
                "result": record["result"],
                "moves": record["moves"],
            }


def save_game_records(records: list[StyleGameRecord], path: str | Path) -> None:
    """Save style game records to a plain, human-readable JSON file.

    This is the "expensive scan once, cheap repeatable runs" artifact:
    a full corpus scan can be slow, but the saved records are small
    (roughly a kilobyte per game) and re-loading them is instant.
    """
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_game_records(path: str | Path) -> list[StyleGameRecord]:
    """Load style game records previously written by `save_game_records`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
