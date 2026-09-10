"""Broadcast-corpus target-player identity resolution for Method 1/2
personalization ingestion.

Matching order, exact FIDE ID first, exact alias string second -- never
substring or fuzzy matching. See `configs/broadcast_targets.json` for the
verified identity data and the reasoning behind which aliases are included.
"""

import json
from pathlib import Path


def load_broadcast_targets(path: str | Path) -> list[dict]:
    """Load the Broadcast target-identity config's `targets` list as-is."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["targets"]


def load_quarantined_game_urls(path: str | Path) -> frozenset[str]:
    """Load the quarantined-game GameURLs from `configs/quarantined_broadcast_games.json`.

    These are games where one side resolves to a target via `match_target`
    but the other side's identity could not be verified to the standard
    required for `configs/broadcast_targets.json` itself -- see that file's
    provenance for why each game is here. Matched by exact `GameURL` string,
    never by name/alias.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    urls: set[str] = set()
    for group in data.get("groups", []):
        urls.update(group.get("game_urls", []))

    return frozenset(urls)


def match_target(name: str, fide_id: str, targets: list[dict]) -> int | None:
    """Return the matching target's player_id, or None.

    Checks every target's `fide_id` for an exact match first (only when
    `fide_id` is non-empty). If no FIDE ID match is found, falls back to an
    exact match against every target's `aliases` list. No substring,
    fuzzy, or normalized matching is performed anywhere.
    """
    if fide_id:
        for target in targets:
            if fide_id == target["fide_id"]:
                return target["player_id"]

    for target in targets:
        if name in target["aliases"]:
            return target["player_id"]

    return None
