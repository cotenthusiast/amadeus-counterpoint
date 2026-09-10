"""Derive each target's frozen representative Elo-conditioning scalar.

Frozen rule (EVALUATION_PROTOCOL.local.md Sec. 6): one fixed value per
target, the MEDIAN of that target's own recorded rating across exactly
their accepted non-cohort (cohort-excluded) Broadcast games -- median, not
mean, because the corpus mixes rating provenances of unequal/unknown scale.

This module only computes a median over caller-supplied per-game records;
it does not decide which games are "allowed". That guarantee comes from
`data.broadcast_ingest.iter_target_game_records`, which already excludes
target-vs-target games entirely and yields only one-target (cohort-excluded)
records -- passing sealed or otherwise-disallowed records in is a caller
error this module has no way to detect.
"""

import statistics


def compute_representative_elos(records: list[dict]) -> dict[int, float]:
    """One median Elo per player_id, over one rating observation per game.

    `records` must be game-level (one entry per game, e.g. StyleGameRecord
    from `broadcast_ingest.iter_target_game_records`/`load_game_records`),
    never per-ply/per-move -- a per-move dataset would silently weight each
    game's rating once per move played instead of once per game.

    Each record's own rating is `white_elo` if `mover_color == "white"`,
    else `black_elo` -- the target's own recorded rating for that game, not
    the opponent's.

    Returns `{player_id: median_elo}`.
    """
    ratings_by_player: dict[int, list[float]] = {}

    for record in records:
        player_id = record["player_id"]
        own_elo = record["white_elo"] if record["mover_color"] == "white" else record["black_elo"]
        ratings_by_player.setdefault(player_id, []).append(own_elo)

    return {
        player_id: statistics.median(ratings)
        for player_id, ratings in ratings_by_player.items()
    }
