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


def load_target_aliases(path: str | Path) -> frozenset[str]:
    """Load and normalize the configured target-cohort Lichess usernames.

    Args:
        path: Path to the target-alias JSON config
            (see `configs/target_aliases.json`).

    Returns:
        Normalized (case-folded) Lichess usernames pooled across all
        configured targets. Empty if the config lists no usernames yet --
        an empty config is valid but means exclusion cannot be applied.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    aliases: set[str] = set()
    for target in data.get("targets", []):
        for username in target.get("lichess_usernames", []):
            aliases.add(normalize_username(username))

    return frozenset(aliases)


def is_target_game(white: str, black: str, aliases: frozenset[str]) -> bool:
    """Return True if either player's username matches a configured target alias."""
    return normalize_username(white) in aliases or normalize_username(black) in aliases
