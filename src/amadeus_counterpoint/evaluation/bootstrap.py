"""Paired whole-game bootstrap uncertainty over sealed real games."""

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_ORIENTATIONS = ("A_WHITE", "B_WHITE")
_CONDITIONS = ("GG", "AG", "GB", "AB")


def paired_bootstrap(
    real_games: Mapping[str, Sequence[Mapping[str, Any]]],
    generated_games: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    metric: Callable[
        [Mapping[str, Sequence[Mapping[str, Any]]], Mapping[str, Sequence[Mapping[str, Any]]]],
        Any,
    ],
    *,
    replicates: int = 10_000,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Bootstrap distances from fixed generated games to paired real samples.

    ``metric`` receives one condition's fixed generated games and the same
    whole-game real resample for every condition in a replicate.  Supply
    exactly one of ``seed`` or ``rng`` so resampling is reproducible.
    """
    for orientation in _ORIENTATIONS:
        if not real_games[orientation]:
            raise ValueError(f"Cannot bootstrap an empty real {orientation} orientation")

    if (seed is None) == (rng is None):
        raise ValueError("Provide exactly one of seed or rng")
    random_source = rng if rng is not None else random.Random(seed)

    conditions = tuple(generated_games)
    distances = {condition: [] for condition in conditions}
    contrast_pairs = (
        ("GG_minus_AB", "GG", "AB"),
        ("AG_minus_AB", "AG", "AB"),
        ("GB_minus_AB", "GB", "AB"),
    )
    available_contrasts = tuple(
        (name, left, right)
        for name, left, right in contrast_pairs
        if left in conditions and right in conditions
    )
    contrasts = {name: [] for name, _, _ in available_contrasts}

    for _ in range(replicates):
        resampled_real = {}
        for orientation in _ORIENTATIONS:
            games = real_games[orientation]
            resampled_real[orientation] = [
                games[random_source.randrange(len(games))]
                for _ in range(len(games))
            ]
        for condition in conditions:
            distances[condition].append(metric(generated_games[condition], resampled_real))

        for name, left, right in available_contrasts:
            contrasts[name].append(distances[left][-1] - distances[right][-1])

    return {"distances": distances, "contrasts": contrasts}
