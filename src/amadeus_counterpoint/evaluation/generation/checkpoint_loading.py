"""Checkpoint -> ready-to-generate object wiring for Method 1 and Method 2.

Deliberately separate from `training.checkpoints` (plain save/load into
already-constructed modules) and from `models.personalized_chessformer`/
`models.player_style`/`models.style_cnn` (the modules themselves). This file
adds exactly the missing step: constructing those modules with the right
shapes and loading a checkpoint into them, so the result can be passed
straight into `play_game_method1`/`play_game_method1_ab`/
`play_game_method2`/`play_game_method2_ab`.

The frozen Broadcast-adapted base model is always supplied by the caller,
already loaded -- one base is shared by every personalized player in an
experiment, and nothing here ever constructs, copies, or reloads it.
"""

from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import (
    load_method1_checkpoint,
    load_method2_checkpoint,
    verify_method1_checkpoint_identity,
)


def load_method1_wrapper_for_generation(
    base: Chessformer,
    checkpoint_path,
    nominal_elo: float,
    identity: str,
    map_location=None,
) -> PersonalizedChessformer:
    """Build a ready-to-generate `PersonalizedChessformer` from a Method-1 checkpoint.

    `base` is the already-loaded, frozen, Broadcast-adapted Chessformer --
    the same object should be reused across every personalized player in an
    experiment; this function never constructs or reloads it. `nominal_elo`
    and `identity` are the caller's own record of which player this
    checkpoint belongs to; they're needed to construct the wrapper before
    its `z_player` is overwritten by the checkpoint, and are checked against
    the checkpoint's own saved values so loading state for the wrong player
    raises instead of silently succeeding.

    Returns the wrapper in eval mode, on `base`'s device, ready to pass into
    `play_game_method1`/`play_game_method1_ab`.
    """
    device = next(base.parameters()).device

    wrapper = PersonalizedChessformer(base, nominal_elo=nominal_elo, identity=identity)
    wrapper.to(device)

    metadata = load_method1_checkpoint(
        checkpoint_path, wrapper, map_location=map_location or device
    )

    verify_method1_checkpoint_identity(metadata, identity, nominal_elo)

    wrapper.eval()
    return wrapper


def load_method2_style_stack_for_generation(
    base: Chessformer,
    checkpoint_path,
    num_players: int,
    style_dim: int = 32,
    map_location=None,
) -> tuple[MoveStyleCNN, PlayerStyleTable, StyleResidual, dict]:
    """Build ready-to-generate Method-2 style modules from a checkpoint.

    `base` is the already-loaded, frozen, Broadcast-adapted Chessformer,
    shared by every player -- this function only reads its device, never
    constructs or reloads it. `num_players` and `style_dim` must match how
    the checkpoint was trained; a `num_players` mismatch fails inside
    `PlayerStyleTable.load_state_dict` with a clear shape error, and a
    `style_dim` mismatch is checked explicitly below since the checkpoint
    records its own.

    Returns `(cnn, table, residual, metadata)`, each module in eval mode on
    `base`'s device and ready to pass into `play_game_method2`/
    `play_game_method2_ab` alongside `base`. `metadata` is exactly
    `training.checkpoints.load_method2_checkpoint`'s return value (`epoch`,
    `step`, `best_val_loss`, `epochs_without_improvement`, `k`, `style_dim`,
    `player_id_map`) -- read `k`/`player_id_map` from there rather than this
    function guessing or inventing a player-identity scheme.
    """
    device = next(base.parameters()).device

    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=num_players, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim)
    cnn.to(device)
    table.to(device)
    residual.to(device)

    metadata = load_method2_checkpoint(
        checkpoint_path, cnn, table, residual, map_location=map_location or device
    )

    if metadata["style_dim"] is not None and metadata["style_dim"] != style_dim:
        raise ValueError(
            f"checkpoint style_dim {metadata['style_dim']!r} does not match "
            f"requested style_dim {style_dim!r}"
        )

    cnn.eval()
    table.eval()
    residual.eval()
    return cnn, table, residual, metadata
