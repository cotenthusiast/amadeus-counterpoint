"""Cell-level filtering for the production generation scripts
(scripts/generate_method1_production.py, scripts/generate_method2_production.py).

Verifies the additive --only-dyad-a/--only-dyad-b/--only-orientation/
--only-condition filters: unfiltered behavior is unchanged, filtered runs
touch only the requested cells, and a cell generated via a filtered
invocation is byte-identical (same seed-derived moves/metadata) to that
same cell generated inside the full unfiltered grid -- proving
derive_seed/write_cell/cell_is_complete behave exactly the same either way.

Uses the tiny `_helpers.CONFIG` architecture throughout (monkeypatched onto
the script modules' own constants) so this runs fast; the scripts'
generation logic itself is exercised unmodified.
"""

import importlib
import json
import sys

import pytest
import torch

from _helpers import CONFIG
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import save_method1_checkpoint, save_method2_checkpoint

IDENTITIES = {"0": "Player Zero", "1": "Player One"}
ELOS = {"0": 1600.0, "1": 1900.0}


def _patch_tiny_architecture(monkeypatch, module):
    monkeypatch.setattr(module, "D_MODEL", CONFIG["d_model"])
    monkeypatch.setattr(module, "NUM_HEADS", CONFIG["num_heads"])
    monkeypatch.setattr(module, "NUM_LAYERS", CONFIG["num_layers"])
    monkeypatch.setattr(module, "D1", CONFIG["d1"])
    monkeypatch.setattr(module, "D2", CONFIG["d2"])
    monkeypatch.setattr(module, "D3", CONFIG["d3"])
    monkeypatch.setattr(module, "D_FF", CONFIG["d_ff"])
    monkeypatch.setattr(module, "ELO_DIM", CONFIG["elo_dim"])
    monkeypatch.setattr(module, "HEAD_HID_DIM", CONFIG["head_hid_dim"])
    monkeypatch.setattr(module, "RAW_INPUT_DIM", CONFIG["input_dim"])
    monkeypatch.setattr(module, "DROPOUT", CONFIG["dropout"])


@pytest.fixture
def tiny_base_checkpoint(tmp_path):
    base = Chessformer(**CONFIG)
    path = tmp_path / "base.pt"
    torch.save({"model": base.state_dict()}, path)
    return base, path


@pytest.fixture
def elos_path(tmp_path):
    path = tmp_path / "representative_elos.json"
    path.write_text(json.dumps(ELOS))
    return path


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    module.main()


def _shard_files(output_root, method):
    return sorted((output_root / method).glob("*/*.parquet"))


# --- Method 1 ----------------------------------------------------------------


@pytest.fixture
def method1_module(monkeypatch):
    module = importlib.import_module("scripts.generate_method1_production")
    _patch_tiny_architecture(monkeypatch, module)
    return module


@pytest.fixture
def method1_checkpoints(tmp_path, tiny_base_checkpoint):
    base, _ = tiny_base_checkpoint
    paths = {}
    for pid, name, elo in [("0", "Player Zero", 1600.0), ("1", "Player One", 1900.0)]:
        wrapper = PersonalizedChessformer(base, nominal_elo=elo, identity=name)
        path = tmp_path / f"player_{pid}.pt"
        save_method1_checkpoint(path, wrapper, epoch=1, best_val_loss=1.0, epochs_without_improvement=0)
        paths[pid] = str(path)
    return paths


def _method1_argv(output_root, checkpoints_json, base_path, elos_path, extra=()):
    return [
        "--base-checkpoint", str(base_path),
        "--player-checkpoints", checkpoints_json,
        "--identities", json.dumps(IDENTITIES),
        "--representative-elos", str(elos_path),
        "--output-root", str(output_root),
        "--root-seed", "12345",
        "--games-per-orientation", "2",
        "--checkpoint-identity", "test-method1",
        "--protocol-version", "test",
        "--code-commit", "test",
        *extra,
    ]


def test_method1_unfiltered_produces_the_full_grid(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    output_root = tmp_path / "out"

    _run(method1_module, _method1_argv(output_root, json.dumps(method1_checkpoints), base_path, elos_path), monkeypatch)

    # 1 dyad x 4 conditions x 2 orientations = 8 cells
    assert len(_shard_files(output_root, "method1")) == 8


def test_method1_only_condition_filters_to_matching_cells_only(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    output_root = tmp_path / "out"

    _run(
        method1_module,
        _method1_argv(
            output_root, json.dumps(method1_checkpoints), base_path, elos_path,
            extra=["--only-condition", "GG"],
        ),
        monkeypatch,
    )

    files = _shard_files(output_root, "method1")
    assert len(files) == 2  # GG x 2 orientations
    assert all("GG_" in f.name for f in files)


def test_method1_only_orientation_filters_to_matching_cells_only(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    output_root = tmp_path / "out"

    _run(
        method1_module,
        _method1_argv(
            output_root, json.dumps(method1_checkpoints), base_path, elos_path,
            extra=["--only-orientation", "A_WHITE"],
        ),
        monkeypatch,
    )

    files = _shard_files(output_root, "method1")
    assert len(files) == 4  # 4 conditions x 1 orientation
    assert all("A_WHITE" in f.name for f in files)


def test_method1_only_dyad_requires_both_args_together(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    output_root = tmp_path / "out"

    with pytest.raises(SystemExit):
        _run(
            method1_module,
            _method1_argv(
                output_root, json.dumps(method1_checkpoints), base_path, elos_path,
                extra=["--only-dyad-a", "0"],
            ),
            monkeypatch,
        )


def test_method1_fully_filtered_cell_matches_the_same_cell_from_the_unfiltered_grid(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    """The core correctness requirement: derive_seed/metadata for one cell
    must be identical whether that cell runs inside the full grid or alone
    via filters -- same seed in, same game out."""
    _, base_path = tiny_base_checkpoint
    checkpoints_json = json.dumps(method1_checkpoints)

    full_root = tmp_path / "full"
    _run(method1_module, _method1_argv(full_root, checkpoints_json, base_path, elos_path), monkeypatch)

    filtered_root = tmp_path / "filtered"
    _run(
        method1_module,
        _method1_argv(
            filtered_root, checkpoints_json, base_path, elos_path,
            extra=[
                "--only-dyad-a", "0", "--only-dyad-b", "1",
                "--only-condition", "AB", "--only-orientation", "A_WHITE",
            ],
        ),
        monkeypatch,
    )

    full_cell = next(f for f in _shard_files(full_root, "method1") if "AB_A_WHITE" in f.name)
    filtered_cell = next(f for f in _shard_files(filtered_root, "method1") if "AB_A_WHITE" in f.name)

    import pyarrow.parquet as pq
    full_games = pq.read_table(full_cell).to_pylist()
    filtered_games = pq.read_table(filtered_cell).to_pylist()

    assert full_games == filtered_games


def test_method1_cell_is_complete_resume_behavior_intact(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    checkpoints_json = json.dumps(method1_checkpoints)
    output_root = tmp_path / "out"

    _run(method1_module, _method1_argv(output_root, checkpoints_json, base_path, elos_path), monkeypatch)
    files_after_first_run = {f: f.read_bytes() for f in _shard_files(output_root, "method1")}

    # Re-running with the same args must not touch any already-complete cell.
    _run(method1_module, _method1_argv(output_root, checkpoints_json, base_path, elos_path), monkeypatch)
    files_after_second_run = {f: f.read_bytes() for f in _shard_files(output_root, "method1")}

    assert files_after_first_run == files_after_second_run


# --- Method 2 ----------------------------------------------------------------


@pytest.fixture
def method2_module(monkeypatch):
    module = importlib.import_module("scripts.generate_method2_production")
    _patch_tiny_architecture(monkeypatch, module)
    return module


@pytest.fixture
def method2_checkpoint(tmp_path):
    style_dim = 16
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=2, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    path = tmp_path / "method2_joint.pt"
    save_method2_checkpoint(
        path, cnn, table, residual, epoch=1, best_val_loss=1.0, epochs_without_improvement=0,
        k=3, style_dim=style_dim, player_id_map={"Player Zero": 0, "Player One": 1},
    )
    return path, style_dim


def _method2_argv(output_root, base_path, method2_ckpt_path, style_dim, elos_path, extra=()):
    return [
        "--base-checkpoint", str(base_path),
        "--method2-checkpoint", str(method2_ckpt_path),
        "--representative-elos", str(elos_path),
        "--player-ids", "0,1",
        "--num-players", "2",
        "--style-dim", str(style_dim),
        "--k", "3",
        "--output-root", str(output_root),
        "--root-seed", "12345",
        "--games-per-orientation", "2",
        "--checkpoint-identity", "test-method2",
        "--protocol-version", "test",
        "--code-commit", "test",
        *extra,
    ]


def test_method2_unfiltered_produces_the_full_grid(
    tmp_path, method2_module, tiny_base_checkpoint, method2_checkpoint, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    method2_ckpt_path, style_dim = method2_checkpoint
    output_root = tmp_path / "out"

    _run(method2_module, _method2_argv(output_root, base_path, method2_ckpt_path, style_dim, elos_path), monkeypatch)

    assert len(_shard_files(output_root, "method2")) == 8


def test_method2_only_condition_and_orientation_filter_to_one_cell(
    tmp_path, method2_module, tiny_base_checkpoint, method2_checkpoint, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    method2_ckpt_path, style_dim = method2_checkpoint
    output_root = tmp_path / "out"

    _run(
        method2_module,
        _method2_argv(
            output_root, base_path, method2_ckpt_path, style_dim, elos_path,
            extra=["--only-condition", "AB", "--only-orientation", "B_WHITE"],
        ),
        monkeypatch,
    )

    files = _shard_files(output_root, "method2")
    assert len(files) == 1
    assert "AB_B_WHITE" in files[0].name


def test_method2_fully_filtered_cell_matches_the_same_cell_from_the_unfiltered_grid(
    tmp_path, method2_module, tiny_base_checkpoint, method2_checkpoint, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    method2_ckpt_path, style_dim = method2_checkpoint

    full_root = tmp_path / "full"
    _run(method2_module, _method2_argv(full_root, base_path, method2_ckpt_path, style_dim, elos_path), monkeypatch)

    filtered_root = tmp_path / "filtered"
    _run(
        method2_module,
        _method2_argv(
            filtered_root, base_path, method2_ckpt_path, style_dim, elos_path,
            extra=[
                "--only-dyad-a", "0", "--only-dyad-b", "1",
                "--only-condition", "GB", "--only-orientation", "A_WHITE",
            ],
        ),
        monkeypatch,
    )

    full_cell = next(f for f in _shard_files(full_root, "method2") if "GB_A_WHITE" in f.name)
    filtered_cell = next(f for f in _shard_files(filtered_root, "method2") if "GB_A_WHITE" in f.name)

    import pyarrow.parquet as pq
    assert pq.read_table(full_cell).to_pylist() == pq.read_table(filtered_cell).to_pylist()
