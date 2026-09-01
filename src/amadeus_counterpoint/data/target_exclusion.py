"""Target-cohort exclusion for population training.

Population training must exclude every known Lichess account belonging to
the 8-player target cohort. The alias list is authoritative-source data --
never guessed -- so it lives in a dedicated config file
(`configs/target_aliases.json`) rather than being hard-coded here. Each
target's canonical real name is a provenance/documentation label only; it is
never itself matched as a Lichess username.
"""

import json
from pathlib import Path


class TargetExclusionRequiredError(RuntimeError):
    """Raised when target exclusion is required but no aliases are configured."""


def normalize_username(username: str) -> str:
    """Normalize a Lichess username for case-insensitive alias matching."""
    return username.strip().casefold()


_ALIAS_TIER_FIELDS = ("verified_usernames", "conservative_candidate_usernames")


def load_target_aliases(path: str | Path) -> frozenset[str]:
    """Load and normalize every configured target-cohort Lichess username.

    Both evidence tiers -- "verified" and "conservative_candidate" -- are
    pooled and excluded identically: the leakage risk is asymmetric (a
    missed target alias contaminates the corpus; an over-cautious
    candidate merely removes a negligible number of unrelated games), so
    exclusion never depends on which tier an alias is in. The tier is
    kept in the config purely for audit/documentation -- see
    `load_target_alias_report`.

    Args:
        path: Path to the target-alias JSON config
            (see `configs/target_aliases.json`).

    Returns:
        Normalized (case-folded) Lichess usernames pooled across all
        configured targets and both tiers. Empty if the config lists no
        usernames yet -- an empty config is valid but means exclusion
        cannot be applied.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    aliases: set[str] = set()
    for target in data.get("targets", []):
        for field in _ALIAS_TIER_FIELDS:
            for username in target.get(field, []):
                aliases.add(normalize_username(username))

    return frozenset(aliases)


def load_target_alias_report(path: str | Path) -> list[dict]:
    """Load the full per-target, per-tier alias breakdown for audit output.

    Returns:
        One dict per target: `canonical_name`, `verified_usernames`, and
        `conservative_candidate_usernames` (both as-configured, not
        normalized) -- for reporting which tier each excluded alias came
        from. Does not affect exclusion itself (see `load_target_aliases`,
        which pools and excludes both tiers identically).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return [
        {
            "canonical_name": target.get("canonical_name", ""),
            "verified_usernames": list(target.get("verified_usernames", [])),
            "conservative_candidate_usernames": list(
                target.get("conservative_candidate_usernames", [])
            ),
        }
        for target in data.get("targets", [])
    ]


def is_target_game(white: str, black: str, aliases: frozenset[str]) -> bool:
    """Return True if either player's username matches a configured target alias."""
    return normalize_username(white) in aliases or normalize_username(black) in aliases
