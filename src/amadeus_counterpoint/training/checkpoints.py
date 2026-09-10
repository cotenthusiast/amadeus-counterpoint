"""Minimal save/load helpers for personalization training runs.

Deliberately not a checkpoint framework: four plain functions, plain dicts,
plain torch.save/torch.load. The frozen Chessformer base is never saved --
only the personalization state (Method 1's z_player, or Method 2's
MoveStyleCNN/PlayerStyleTable/StyleResidual) plus whatever training-progress
metadata the caller passes in.

The load functions restore state into modules the caller has already
constructed; they do not build a Chessformer, a PersonalizedChessformer, or
a StyleTrainer.
"""

import torch


def save_method1_checkpoint(
    path,
    wrapper,
    optimizer=None,
    epoch=None,
    step=None,
    best_val_loss=None,
    epochs_without_improvement=None,
):
    """Save a Method-1 (player-embedding) training run.

    `wrapper` is a PersonalizedChessformer; only its trainable z_player and
    its own identity/nominal_elo attributes are saved -- `wrapper.base`
    (the frozen Chessformer) is never touched.
    """
    checkpoint = {
        "z_player": wrapper.z_player.detach().clone(),
        "identity": wrapper.identity,
        "nominal_elo": wrapper.nominal_elo,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    torch.save(checkpoint, path)


def load_method1_checkpoint(path, wrapper, optimizer=None, map_location=None):
    """Restore a Method-1 checkpoint into an already-constructed `wrapper`.

    Returns a dict with the saved metadata (`identity`, `nominal_elo`,
    `epoch`, `step`, `best_val_loss`, `epochs_without_improvement`) so the
    caller can resume its own training loop. `epochs_without_improvement`
    defaults to 0 for a legacy checkpoint saved before this field existed,
    matching a fresh trainer's own starting value.
    """
    checkpoint = torch.load(path, map_location=map_location)

    with torch.no_grad():
        wrapper.z_player.copy_(checkpoint["z_player"])

    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "identity": checkpoint["identity"],
        "nominal_elo": checkpoint["nominal_elo"],
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "epochs_without_improvement": checkpoint.get("epochs_without_improvement") or 0,
    }


def verify_method1_checkpoint_identity(metadata: dict, identity: str, nominal_elo: float) -> None:
    """Raise if a loaded Method-1 checkpoint's saved identity doesn't match
    the identity/nominal_elo the caller intended to load, so continuing
    training (or generation) from the wrong player's checkpoint fails
    clearly instead of silently overwriting `z_player` with the wrong
    player's state. Shared by `checkpoint_loading.
    load_method1_wrapper_for_generation` and `scripts.train_method1_player`
    so both use identical validation.
    """
    if metadata["identity"] != identity:
        raise ValueError(
            f"checkpoint identity {metadata['identity']!r} does not match "
            f"requested identity {identity!r}"
        )
    if metadata["nominal_elo"] != nominal_elo:
        raise ValueError(
            f"checkpoint nominal_elo {metadata['nominal_elo']!r} does not "
            f"match requested nominal_elo {nominal_elo!r}"
        )


def save_method2_checkpoint(
    path,
    cnn,
    table,
    residual,
    optimizer=None,
    epoch=None,
    step=None,
    best_val_loss=None,
    epochs_without_improvement=None,
    k=None,
    style_dim=None,
    player_id_map=None,
):
    """Save a Method-2 (style-residual) training run.

    Only `cnn`, `table`, and `residual`'s own state_dicts are saved -- the
    frozen Chessformer base is never touched. `player_id_map`, if given, is
    saved as opaque caller-supplied metadata; this function does not define
    or invent any player-identity scheme of its own.
    """
    checkpoint = {
        "cnn": cnn.state_dict(),
        "table": table.state_dict(),
        "residual": residual.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "k": k,
        "style_dim": style_dim,
        "player_id_map": player_id_map,
    }
    torch.save(checkpoint, path)


def load_method2_checkpoint(path, cnn, table, residual, optimizer=None, map_location=None):
    """Restore a Method-2 checkpoint into already-constructed `cnn`, `table`,
    and `residual`.

    Returns a dict with the saved metadata (`epoch`, `step`,
    `best_val_loss`, `epochs_without_improvement`, `k`, `style_dim`,
    `player_id_map`) so the caller can resume its own training loop.
    """
    checkpoint = torch.load(path, map_location=map_location)

    cnn.load_state_dict(checkpoint["cnn"])
    table.load_state_dict(checkpoint["table"])
    residual.load_state_dict(checkpoint["residual"])

    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "epochs_without_improvement": checkpoint["epochs_without_improvement"],
        "k": checkpoint["k"],
        "style_dim": checkpoint["style_dim"],
        "player_id_map": checkpoint["player_id_map"],
    }
