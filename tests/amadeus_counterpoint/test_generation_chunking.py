"""Internal chunking of generate_method1_cell_batched/
generate_method2_cell_batched (and the production scripts'
--generation-batch-size CLI flag that drives it).

A production cell always contains exactly n_games (e.g. 5000) games --
chunking only changes HOW MANY of those games are active on the GPU at
once, never the seed a given global game_index gets, never the count of
games in the cell, and never the output schema. This file proves:
  1. game_index stays 0..n_games-1 regardless of chunk_size.
  2. Every seed (derive_seed output) is identical regardless of chunk_size.
  3. A chunk_size covering all n_games in one chunk is exactly equivalent
     to no chunking (trivial floor).
  4. A games count NOT divisible by chunk_size exercises a partial final
     chunk correctly (right count, right indices).
  5. Metadata fields match across chunk sizes.
  6. The production scripts' resume/cell_is_complete behavior and default
     (no --generation-batch-size) behavior are unaffected by chunking.

One thing this file does NOT claim: that sub-batch chunk_size values (e.g.
3 or 8 games per chunk out of 13) reproduce the unchunked call move-for-
move. Batched neural-net inference is not exactly reduction-order-
associative across different active-batch sizes -- the same floating-point
caveat already documented for single-vs-batched generation. This was tested
directly (all 4 conditions x several sub-batch chunk sizes x both methods,
on the tiny CPU test model): divergence turned out to be rare but real
(Method 2's AB condition, 2 of ~24 sub-batch cases produced different
moves for the same seed at a different chunk size). What IS guaranteed by
construction and IS asserted below for every chunk size: game count, every
game_index/seed/metadata field, and that every produced game is legal and
schema-valid. See the real-GPU sanity check (production checkpoints,
production chunk_size=128) for the validation that determines whether
production output is affected in practice.
"""

import importlib
import json
import sys

import chess
import pytest
import torch

from _helpers import CONFIG, replay_and_check_legal
from amadeus_counterpoint.data.sealed_dyads import A_WHITE
from amadeus_counterpoint.evaluation.generation.experiment import (
    generate_method1_cell_batched,
    generate_method2_cell_batched,
)
from amadeus_counterpoint.evaluation.seeding import derive_seed
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import save_method1_checkpoint, save_method2_checkpoint

DYAD = "0_1"
ROOT_SEED = 20260911
N_GAMES = 13  # deliberately not divisible by any chunk_size tested below
CHUNK_SIZES = [1, 3, 5, 8, N_GAMES, N_GAMES + 7, None]


@pytest.fixture
def base():
    return Chessformer(**CONFIG)


@pytest.fixture
def method1_wrappers(base):
    wrapper_a = PersonalizedChessformer(base, nominal_elo=1600.0, identity="Player Zero")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=1900.0, identity="Player One")
    return wrapper_a, wrapper_b


@pytest.fixture
def method2_stack(base):
    style_dim = 16
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=2, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    return cnn, table, residual


# --- direct function-level chunking correctness -------------------------------


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
def test_method1_chunking_preserves_game_index_and_seeds_regardless_of_chunk_size(
    method1_wrappers, base, condition
):
    wrapper_a, wrapper_b = method1_wrappers

    expected_seeds = [
        derive_seed(
            root_seed=ROOT_SEED, dyad=f"method1__{DYAD}", condition=condition,
            orientation=A_WHITE, game_index=i,
        )
        for i in range(N_GAMES)
    ]

    for chunk_size in CHUNK_SIZES:
        games = generate_method1_cell_batched(
            wrapper_a, wrapper_b, base, chess.WHITE, condition,
            1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
            chunk_size=chunk_size,
        )
        assert [g["game_index"] for g in games] == list(range(N_GAMES))
        assert [g["seed"] for g in games] == expected_seeds


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
def test_method2_chunking_preserves_game_index_and_seeds_regardless_of_chunk_size(
    method2_stack, base, condition
):
    cnn, table, residual = method2_stack

    expected_seeds = [
        derive_seed(
            root_seed=ROOT_SEED, dyad=f"method2__{DYAD}", condition=condition,
            orientation=A_WHITE, game_index=i,
        )
        for i in range(N_GAMES)
    ]

    for chunk_size in CHUNK_SIZES:
        games = generate_method2_cell_batched(
            base, cnn, table, residual, 0, 1, chess.WHITE, condition,
            1600.0, 1900.0, 3, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
            chunk_size=chunk_size,
        )
        assert [g["game_index"] for g in games] == list(range(N_GAMES))
        assert [g["seed"] for g in games] == expected_seeds


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
@pytest.mark.parametrize("chunk_size", [N_GAMES, N_GAMES + 7])
def test_method1_chunk_size_covering_all_games_matches_unchunked_exactly(
    method1_wrappers, base, condition, chunk_size
):
    """chunk_size >= n_games collapses to the same single-chunk call as
    chunk_size=None -- a trivial but real correctness floor."""
    wrapper_a, wrapper_b = method1_wrappers

    unchunked = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, condition,
        1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=None,
    )
    chunked = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, condition,
        1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=chunk_size,
    )

    assert chunked == unchunked


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
@pytest.mark.parametrize("chunk_size", [N_GAMES, N_GAMES + 7])
def test_method2_chunk_size_covering_all_games_matches_unchunked_exactly(
    method2_stack, base, condition, chunk_size
):
    cnn, table, residual = method2_stack

    unchunked = generate_method2_cell_batched(
        base, cnn, table, residual, 0, 1, chess.WHITE, condition,
        1600.0, 1900.0, 3, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=None,
    )
    chunked = generate_method2_cell_batched(
        base, cnn, table, residual, 0, 1, chess.WHITE, condition,
        1600.0, 1900.0, 3, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=chunk_size,
    )

    assert chunked == unchunked


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
@pytest.mark.parametrize("chunk_size", [1, 3, 5, 8])
def test_method1_sub_batch_chunk_sizes_preserve_count_legality_and_metadata(
    method1_wrappers, base, condition, chunk_size
):
    """Sub-batch chunk sizes (chunk_size < n_games) are NOT guaranteed to
    reproduce the unchunked call move-for-move: batched neural-net inference
    is not exactly reduction-order-associative across different active-batch
    sizes, the same floating-point caveat already documented for single-vs-
    batched generation (see evaluation.generation.batch's docstring, and
    Phase 4's real-GPU validation). Empirically (see this file's git history
    / the Phase 6 report) this divergence is rare but real even on a tiny
    CPU model: it showed up for Method 2's AB condition at some chunk sizes
    in this exact parametrization. What chunking DOES guarantee by
    construction -- game count, game_index/seed/metadata identity (asserted
    exhaustively in the tests above), and every produced game being a legal,
    schema-valid game -- is what this test checks.
    """
    wrapper_a, wrapper_b = method1_wrappers

    chunked = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, condition,
        1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=chunk_size,
    )

    assert len(chunked) == N_GAMES
    assert [g["game_index"] for g in chunked] == list(range(N_GAMES))
    for g in chunked:
        replay_and_check_legal(g["moves"])
        if g["censored"]:
            assert g["result"] is None


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
@pytest.mark.parametrize("chunk_size", [1, 3, 5, 8])
def test_method2_sub_batch_chunk_sizes_preserve_count_legality_and_metadata(
    method2_stack, base, condition, chunk_size
):
    """See test_method1_sub_batch_chunk_sizes_preserve_count_legality_and_metadata."""
    cnn, table, residual = method2_stack

    chunked = generate_method2_cell_batched(
        base, cnn, table, residual, 0, 1, chess.WHITE, condition,
        1600.0, 1900.0, 3, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
        chunk_size=chunk_size,
    )

    assert len(chunked) == N_GAMES
    assert [g["game_index"] for g in chunked] == list(range(N_GAMES))
    for g in chunked:
        replay_and_check_legal(g["moves"])
        if g["censored"]:
            assert g["result"] is None


def test_chunk_size_zero_or_negative_raises(method1_wrappers, base):
    wrapper_a, wrapper_b = method1_wrappers
    with pytest.raises(ValueError):
        generate_method1_cell_batched(
            wrapper_a, wrapper_b, base, chess.WHITE, "GG",
            1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
            chunk_size=0,
        )
    with pytest.raises(ValueError):
        generate_method1_cell_batched(
            wrapper_a, wrapper_b, base, chess.WHITE, "GG",
            1600.0, 1900.0, DYAD, N_GAMES, ROOT_SEED, "chunk-test",
            chunk_size=-1,
        )


def test_zero_games_returns_empty_list_regardless_of_chunk_size(method1_wrappers, base):
    wrapper_a, wrapper_b = method1_wrappers
    for chunk_size in [None, 1, 128]:
        games = generate_method1_cell_batched(
            wrapper_a, wrapper_b, base, chess.WHITE, "GG",
            1600.0, 1900.0, DYAD, 0, ROOT_SEED, "chunk-test",
            chunk_size=chunk_size,
        )
        assert games == []


# --- production-script CLI level (--generation-batch-size) --------------------


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


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    module.main()


def _shard_files(output_root, method):
    return sorted((output_root / method).glob("*/*.parquet"))


IDENTITIES = {"0": "Player Zero", "1": "Player One"}
ELOS = {"0": 1600.0, "1": 1900.0}


@pytest.fixture
def tiny_base_checkpoint(tmp_path):
    b = Chessformer(**CONFIG)
    path = tmp_path / "base.pt"
    torch.save({"model": b.state_dict()}, path)
    return b, path


@pytest.fixture
def elos_path(tmp_path):
    path = tmp_path / "representative_elos.json"
    path.write_text(json.dumps(ELOS))
    return path


@pytest.fixture
def method1_module(monkeypatch):
    module = importlib.import_module("scripts.generate_method1_production")
    _patch_tiny_architecture(monkeypatch, module)
    return module


@pytest.fixture
def method1_checkpoints(tmp_path, tiny_base_checkpoint):
    b, _ = tiny_base_checkpoint
    paths = {}
    for pid, name, elo in [("0", "Player Zero", 1600.0), ("1", "Player One", 1900.0)]:
        wrapper = PersonalizedChessformer(b, nominal_elo=elo, identity=name)
        path = tmp_path / f"player_{pid}.pt"
        save_method1_checkpoint(path, wrapper, epoch=1, best_val_loss=1.0, epochs_without_improvement=0)
        paths[pid] = str(path)
    return paths


def _method1_argv(output_root, checkpoints_json, base_path, elos_path, games_per_orientation, extra=()):
    return [
        "--base-checkpoint", str(base_path),
        "--player-checkpoints", checkpoints_json,
        "--identities", json.dumps(IDENTITIES),
        "--representative-elos", str(elos_path),
        "--output-root", str(output_root),
        "--root-seed", "12345",
        "--games-per-orientation", str(games_per_orientation),
        "--checkpoint-identity", "test-method1",
        "--protocol-version", "test",
        "--code-commit", "test",
        "--only-dyad-a", "0", "--only-dyad-b", "1",
        "--only-condition", "AB", "--only-orientation", "A_WHITE",
        *extra,
    ]


def test_method1_default_generation_batch_size_is_128(method1_module):
    assert method1_module.PRODUCTION_GENERATION_BATCH_SIZE == 128


def test_method1_script_output_identical_across_generation_batch_sizes(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    """A cell run with different --generation-batch-size values (including a
    size that forces multiple chunks for a not-divisible game count) must
    produce byte-identical output."""
    _, base_path = tiny_base_checkpoint
    checkpoints_json = json.dumps(method1_checkpoints)
    games_per_orientation = 7  # not divisible by 3

    import pyarrow.parquet as pq

    reference = None
    for batch_size in [3, 4, 7, 128]:
        output_root = tmp_path / f"out_{batch_size}"
        _run(
            method1_module,
            _method1_argv(
                output_root, checkpoints_json, base_path, elos_path, games_per_orientation,
                extra=["--generation-batch-size", str(batch_size)],
            ),
            monkeypatch,
        )
        cell = next(f for f in _shard_files(output_root, "method1") if "AB_A_WHITE" in f.name)
        records = pq.read_table(cell).to_pylist()
        assert len(records) == games_per_orientation
        assert [r["game_index"] for r in records] == list(range(games_per_orientation))
        if reference is None:
            reference = records
        else:
            assert records == reference


def test_method1_script_omitting_generation_batch_size_matches_explicit_default(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    checkpoints_json = json.dumps(method1_checkpoints)

    import pyarrow.parquet as pq

    default_root = tmp_path / "default"
    _run(method1_module, _method1_argv(default_root, checkpoints_json, base_path, elos_path, 5), monkeypatch)

    explicit_root = tmp_path / "explicit"
    _run(
        method1_module,
        _method1_argv(
            explicit_root, checkpoints_json, base_path, elos_path, 5,
            extra=["--generation-batch-size", "128"],
        ),
        monkeypatch,
    )

    default_cell = next(f for f in _shard_files(default_root, "method1") if "AB_A_WHITE" in f.name)
    explicit_cell = next(f for f in _shard_files(explicit_root, "method1") if "AB_A_WHITE" in f.name)
    assert pq.read_table(default_cell).to_pylist() == pq.read_table(explicit_cell).to_pylist()


def test_method1_resume_still_skips_completed_cell_with_small_generation_batch_size(
    tmp_path, method1_module, tiny_base_checkpoint, method1_checkpoints, elos_path, monkeypatch
):
    _, base_path = tiny_base_checkpoint
    checkpoints_json = json.dumps(method1_checkpoints)
    output_root = tmp_path / "out"
    argv = _method1_argv(
        output_root, checkpoints_json, base_path, elos_path, 7,
        extra=["--generation-batch-size", "3"],
    )

    _run(method1_module, argv, monkeypatch)
    files_after_first_run = {f: f.read_bytes() for f in _shard_files(output_root, "method1")}

    _run(method1_module, argv, monkeypatch)
    files_after_second_run = {f: f.read_bytes() for f in _shard_files(output_root, "method1")}

    assert files_after_first_run == files_after_second_run
