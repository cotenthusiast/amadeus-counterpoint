"""Extraction of sealed target-vs-target games for the 28 unordered dyads
from the raw Lichess Broadcast PGN corpus.

These are the held-out real-world comparison games for later evaluation
(EVALUATION_PROTOCOL.local.md, Sealed Data Rule). They must never be used
for population training, Broadcast adaptation, or Method-1/Method-2
personalization -- `data/broadcast_ingest.py` already excludes them
completely from personalization ingestion by discarding its
"target_vs_target" classification. This module does the mirror-image scan:
it keeps exactly the games that function discards.

Reuses the same frozen identity rule (`data/broadcast_targets.match_target`)
and the same validity rule (`data/preprocess.game_to_record`) as Stage 6 --
no second identity matcher and no second validity rule are introduced here.
"""

from typing import TypedDict

from amadeus_counterpoint.data.broadcast_targets import match_target
from amadeus_counterpoint.data.preprocess import game_to_record, iter_pgn_games

A_WHITE = "A_WHITE"
B_WHITE = "B_WHITE"


class SealedDyadGameRecord(TypedDict):
    """One valid target-vs-target game, kept only as real comparison data.

    `player_id_a`/`player_id_b` are the dyad's two target identities, always
    ordered `player_id_a < player_id_b` -- this is the canonical dyad
    ordering. Which target actually played White is recorded separately in
    `orientation` (`"A_WHITE"` means the lower-id target had White,
    `"B_WHITE"` means the higher-id target did) and in `white_player_id`/
    `black_player_id`, so callers never have to re-derive it.

    `date` and `game_url` are passed through verbatim from the PGN headers
    (`None` if absent) for structural duplicate/provenance inspection --
    neither is a fabricated identifier.
    """

    player_id_a: int
    player_id_b: int
    orientation: str
    white_player_id: int
    black_player_id: int
    white_elo: int
    black_elo: int
    result: str
    moves: list[str]
    date: str | None
    game_url: str | None


def dyad_key(player_id_a: int, player_id_b: int) -> str:
    """Canonical unordered-dyad key: the two player_ids, lower one first."""
    low = min(player_id_a, player_id_b)
    high = max(player_id_a, player_id_b)
    return f"{low}__{high}"


def iter_sealed_dyad_games(pgn_paths, targets: list[dict], stats: dict | None = None):
    """Stream every valid target-vs-target game across `pgn_paths`.

    Args:
        pgn_paths: Paths to raw Broadcast PGN files, scanned in order.
        targets: The loaded `configs/broadcast_targets.json` targets list.
        stats: If given, incremented in place with counts for
            "games_scanned", "two_target_games", "self_pair_games" (both
            sides resolved to the *same* target -- a data anomaly, not a
            real dyad game), and "invalid_sealed_games" (a genuine two
            distinct-target game that failed `game_to_record`'s validity
            rule).
    """
    if stats is not None:
        stats.setdefault("games_scanned", 0)
        stats.setdefault("two_target_games", 0)
        stats.setdefault("self_pair_games", 0)
        stats.setdefault("invalid_sealed_games", 0)

    for path in pgn_paths:
        for game in iter_pgn_games(path):
            if stats is not None:
                stats["games_scanned"] += 1

            headers = game.headers
            white_name = headers.get("White", "")
            black_name = headers.get("Black", "")
            white_fide_id = headers.get("WhiteFideId", "")
            black_fide_id = headers.get("BlackFideId", "")

            white_player_id = match_target(white_name, white_fide_id, targets)
            black_player_id = match_target(black_name, black_fide_id, targets)

            if white_player_id is None or black_player_id is None:
                # Zero- or one-target game: not sealed. broadcast_ingest.py
                # handles those cases; this module only wants both-target.
                continue

            if white_player_id == black_player_id:
                # Both header fields resolved to the same target -- a data
                # anomaly (e.g. a duplicated/mistyped FIDE ID), not a real
                # dyad game. Never turn this into a self-pair.
                if stats is not None:
                    stats["self_pair_games"] += 1
                continue

            if stats is not None:
                stats["two_target_games"] += 1

            record = game_to_record(game)
            if record is None:
                if stats is not None:
                    stats["invalid_sealed_games"] += 1
                continue

            orientation = A_WHITE if white_player_id < black_player_id else B_WHITE

            yield {
                "player_id_a": min(white_player_id, black_player_id),
                "player_id_b": max(white_player_id, black_player_id),
                "orientation": orientation,
                "white_player_id": white_player_id,
                "black_player_id": black_player_id,
                "white_elo": record["white_elo"],
                "black_elo": record["black_elo"],
                "result": record["result"],
                "moves": record["moves"],
                "date": headers.get("Date"),
                "game_url": headers.get("GameURL"),
            }


def group_sealed_games_by_dyad(
    games,
) -> dict[str, dict[str, list[SealedDyadGameRecord]]]:
    """Group sealed game records by dyad, split into A_WHITE/B_WHITE lists.

    Returns `{dyad_key(...): {"A_WHITE": [...], "B_WHITE": [...]}}`, matching
    the orientation-keyed shape `evaluation.metrics`/`evaluation.bootstrap`
    already expect for a dyad's real games. Every input game belongs to
    exactly one dyad and exactly one orientation bucket within it.
    """
    grouped: dict[str, dict[str, list[SealedDyadGameRecord]]] = {}

    for game in games:
        key = dyad_key(game["player_id_a"], game["player_id_b"])
        if key not in grouped:
            grouped[key] = {A_WHITE: [], B_WHITE: []}
        grouped[key][game["orientation"]].append(game)

    return grouped


def find_repeated_game_urls(games) -> dict[str, int]:
    """Structural duplicate check: `game_url` values shared by >1 game.

    Returns `{game_url: count}` for every non-null `game_url` that appears
    more than once among `games`. Reports only -- never removes or merges
    anything; deduplication is a policy decision this module does not make.
    """
    counts: dict[str, int] = {}
    for game in games:
        game_url = game["game_url"]
        if game_url is None:
            continue
        counts[game_url] = counts.get(game_url, 0) + 1

    return {game_url: count for game_url, count in counts.items() if count > 1}


def _content_signature(game) -> tuple:
    """The exact-game identity signature: who played which color, the date
    the game was actually played, the result, and the full move sequence.

    Deliberately excludes broadcast-publishing metadata (`game_url`, and any
    StudyName/ChapterName/BroadcastName/Round/Board fields this module never
    even carries) -- those describe how Lichess chose to republish a game,
    not the game itself. `date` is kept: two DIFFERENT real games between the
    same two players, with the same result and the exact same full move
    list, would be an astronomically unlikely coincidence for anything past
    a handful of plies -- but dropping `date` entirely was checked against
    the real corpus and does produce a handful of false merges (short,
    drawish move sequences the same two players independently repeated in
    unrelated events years apart). Keeping it avoids that, at the small,
    accepted cost of not catching the rarer case where the same real game
    carries a corrupted/placeholder date in one of its two broadcast copies.
    """
    return (
        game["white_player_id"],
        game["black_player_id"],
        game["date"],
        game["result"],
        tuple(game["moves"]),
    )


def find_repeated_content(games) -> dict[tuple, int]:
    """Structural duplicate check: identical (players, date, result, moves).

    Catches the same real game re-published under a different `game_url`
    (e.g. re-broadcast in a different study), which `find_repeated_game_urls`
    cannot see. Returns `{content_key: count}` for every content signature
    that appears more than once. Reports only -- see `find_repeated_game_urls`.
    """
    counts: dict[tuple, int] = {}
    for game in games:
        content_key = _content_signature(game)
        counts[content_key] = counts.get(content_key, 0) + 1

    return {content_key: count for content_key, count in counts.items() if count > 1}


def deduplicate_sealed_games(
    games: list[SealedDyadGameRecord],
) -> tuple[list[SealedDyadGameRecord], dict[str, int]]:
    """Keep exactly one record per exact-content duplicate group.

    Uses the same `_content_signature` as `find_repeated_content`: these
    fields fully identify the underlying real chess game, independent of
    which broadcast republished it -- the goal is that the deduplicated
    sealed sample's unit is a unique real game, not a broadcast row.

    Within each duplicate group, the first-occurring record in `games`'s own
    order is kept as canonical; later duplicates are dropped entirely, never
    merged or altered. This is deterministic given a fixed input order (the
    same order `iter_sealed_dyad_games` always produces for the same PGN
    corpus), and mirrors the audit workspace's own "keep the first
    occurrence" precedent for its `games_dedup` view.

    Only ever call this on sealed target-vs-target records -- it has nothing
    to do with, and must never be applied to, personalization/Broadcast
    training data.

    Returns `(deduplicated_games, report)`, where `report` has
    `total_before`, `total_after`, `duplicate_groups`, and `extra_removed`.
    """
    seen_signatures = set()
    deduplicated = []

    for game in games:
        signature = _content_signature(game)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduplicated.append(game)

    duplicate_groups = find_repeated_content(games)
    report = {
        "total_before": len(games),
        "total_after": len(deduplicated),
        "duplicate_groups": len(duplicate_groups),
        "extra_removed": len(games) - len(deduplicated),
    }
    return deduplicated, report
