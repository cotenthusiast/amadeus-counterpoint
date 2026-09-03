"""Deterministic per-game seed derivation, independent of batching, execution
order, resumption, or SLURM sharding.
"""

import hashlib


def derive_seed(
    root_seed: int,
    dyad: str,
    condition: str,
    orientation: str,
    game_index: int,
) -> int:
    """Derive a deterministic per-game seed from a game's experimental identity.

    The seed is a pure function of `(root_seed, dyad, condition, orientation,
    game_index)`: the same tuple always hashes to the same seed, regardless
    of process, machine, SLURM task, batch packing, or execution order. Pass
    the result to `torch.Generator().manual_seed(...)`, matching
    `generation.single.play_game` / `generation.batch.play_games`.

    `dyad` is expected to already be a canonical identifier (e.g.
    "playerA__playerB"); this function does not attempt player-name
    normalization or resolution.

    Fields are length-prefixed before hashing so that no combination of
    field values can collide by concatenating differently (e.g.
    `("ab", "c")` vs. `("a", "bc")`).
    """
    fields = (str(root_seed), dyad, condition, orientation, str(game_index))

    buf = bytearray()
    for field in fields:
        encoded = field.encode("utf-8")
        buf += len(encoded).to_bytes(4, "big")
        buf += encoded

    digest = hashlib.blake2b(bytes(buf), digest_size=8).digest()
    return int.from_bytes(digest, "big")
