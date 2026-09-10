import chess
import pyarrow as pa
import pytest
import torch

from amadeus_counterpoint.evaluation import artifacts
from amadeus_counterpoint.evaluation.generation import batch, single
from amadeus_counterpoint.models import Chessformer

MODEL_CONFIG = {
    "d_model": 32,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "d1": 4,
    "d2": 8,
    "d3": 4,
    "d_ff": 64,
    "head_hid_dim": 16,
    "input_dim": 96,
    "elo_dim": 8,
}


class _CapturingChessformer(Chessformer):
    def forward(self, x, player_elo, opponent_elo):
        self.player_elos = player_elo.detach().clone()
        self.opponent_elos = opponent_elo.detach().clone()
        self.player_interpolation = self.interpolate_elo(player_elo).detach().clone()
        return torch.zeros((len(x), 4352)), torch.zeros((len(x), 3))


def _first_legal_move(index, board):
    return next(iter(board.legal_moves))


def _outcome(result):
    winner = {
        "1-0": chess.WHITE,
        "0-1": chess.BLACK,
        "1/2-1/2": None,
    }[result]
    return chess.Outcome(chess.Termination.CHECKMATE, winner)


def _artifact_game(**overrides):
    game = {
        "game_index": 0,
        "seed": 9,
        "dyad": "alpha__beta",
        "condition": "AG",
        "orientation": "A_WHITE",
        "result": "1-0",
        "censored": False,
        "moves": ["e2e4"],
        "white_elo": 1500.5,
        "black_elo": 1600.5,
        "checkpoint_identity": "checkpoint-42",
        "white_representation_identity": "alpha",
        "black_representation_identity": "beta",
    }
    game.update(overrides)
    return game


def test_play_game_marks_cap_as_censored_and_preserves_fractional_elo(monkeypatch):
    model = _CapturingChessformer(**MODEL_CONFIG)
    calls = 0

    def check_end_after_one_ply(board):
        nonlocal calls
        calls += 1
        return None if calls == 1 else _outcome("1-0")

    monkeypatch.setattr(single, "check_end", check_end_after_one_ply)
    monkeypatch.setattr(single, "policy_index_to_move", _first_legal_move)

    normal = single.play_game(model, 1500.5, 1600.5, seed=7)

    assert normal["result"] == "1-0"
    assert normal["censored"] is False
    assert model.player_elos.dtype == torch.float32
    assert model.player_elos.tolist() == [1500.5]
    assert model.opponent_elos.tolist() == [1600.5]
    assert torch.allclose(
        model.player_interpolation,
        model.interpolate_elo(torch.tensor([1500.5])),
    )

    monkeypatch.setattr(single, "MAX_PLIES", 0)
    monkeypatch.setattr(single, "check_end", lambda board: None)
    capped = single.play_game(model, 1500.5, 1600.5, seed=8)

    assert capped["result"] is None
    assert capped["censored"] is True


def test_single_and_batch_use_the_frozen_500_ply_cap():
    assert single.MAX_PLIES == 500
    assert batch.MAX_PLIES == 500


def test_play_games_preserves_fractional_elo_and_rejects_misaligned_inputs(monkeypatch):
    model = _CapturingChessformer(**MODEL_CONFIG)
    calls = 0

    def check_end_after_one_batch_ply(board):
        nonlocal calls
        calls += 1
        return None if calls <= 2 else _outcome("1/2-1/2")

    monkeypatch.setattr(batch, "check_end", check_end_after_one_batch_ply)
    monkeypatch.setattr(batch, "policy_index_to_move", _first_legal_move)

    games = batch.play_games(
        model,
        white_elos=[1500.5, 1700.5],
        black_elos=[1600.5, 1800.5],
        seeds=[1, 2],
    )

    assert [game["result"] for game in games] == ["1/2-1/2", "1/2-1/2"]
    assert [game["censored"] for game in games] == [False, False]
    assert model.player_elos.dtype == torch.float32
    assert sorted(model.player_elos.tolist()) == [1500.5, 1700.5]
    assert sorted(model.opponent_elos.tolist()) == [1600.5, 1800.5]
    assert torch.allclose(
        model.player_interpolation,
        model.interpolate_elo(model.player_elos),
    )

    with pytest.raises(ValueError, match="equal lengths"):
        batch.play_games(model, [1500.5], [1600.5, 1700.5], [1])


def test_play_games_marks_ply_cap_as_censored(monkeypatch):
    monkeypatch.setattr(batch, "MAX_PLIES", 0)

    games = batch.play_games(
        _CapturingChessformer(**MODEL_CONFIG),
        white_elos=[1500.5],
        black_elos=[1600.5],
        seeds=[1],
    )

    assert games == [{
        "white_elo": 1500.5,
        "black_elo": 1600.5,
        "result": None,
        "censored": True,
        "moves": [],
    }]


def test_games_artifact_round_trip_and_refuses_overwrite(tmp_path):
    path = tmp_path / "games.parquet"
    games = [
        {
            "game_index": 0,
            "seed": 2**64 - 1,
            "dyad": "alpha__beta",
            "condition": "AG",
            "orientation": "A_WHITE",
            "result": "1-0",
            "censored": False,
            "moves": ["e2e4", "e7e5"],
            "white_elo": 1500.5,
            "black_elo": 1600.5,
            "checkpoint_identity": "checkpoint-42",
            "white_representation_identity": "alpha",
            "black_representation_identity": "beta",
        },
        {
            "game_index": 1,
            "seed": 2**64 - 2,
            "dyad": "alpha__beta",
            "condition": "GB",
            "orientation": "B_WHITE",
            "result": None,
            "censored": True,
            "moves": [],
            "white_elo": 1750.5,
            "black_elo": 1650.5,
            "checkpoint_identity": "checkpoint-42",
            "white_representation_identity": "beta",
            "black_representation_identity": "global",
        },
    ]
    metadata = {
        "protocol_version": "1",
        "code_commit": "abc123",
        "root_seed": "11",
        "temperature": "1.0",
    }

    artifacts.write_games(path, games, metadata)
    read_back = artifacts.read_games(path)

    assert read_back.schema == artifacts.GAME_SCHEMA.with_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
    )
    assert read_back.to_pylist() == games
    assert read_back.schema.field("seed").type == pa.uint64()
    assert read_back.schema.field("white_elo").type == pa.float64()

    with pytest.raises(FileExistsError):
        artifacts.write_games(path, games, metadata)


@pytest.mark.parametrize(
    ("games", "message"),
    [
        ([_artifact_game(), _artifact_game()], "game_index values must be unique"),
        ([_artifact_game(censored=True)], "censored games must have result=None"),
        ([_artifact_game(result=None)], "normal games must have a result"),
        ([_artifact_game(condition="invalid")], "unknown condition"),
        ([_artifact_game(orientation="invalid")], "unknown orientation"),
    ],
)
def test_write_games_validates_scientific_game_identity(tmp_path, games, message):
    with pytest.raises(ValueError, match=message):
        artifacts.write_games(tmp_path / "invalid.parquet", games, {})


def test_write_games_requires_minimal_provenance_metadata(tmp_path):
    metadata = {
        "protocol_version": "1",
        "code_commit": "abc123",
        "root_seed": "11",
    }

    with pytest.raises(ValueError, match="temperature"):
        artifacts.write_games(
            tmp_path / "missing-provenance.parquet",
            [_artifact_game()],
            metadata,
        )

    metadata.update(temperature="1.0", extra_provenance="allowed")
    path = tmp_path / "complete-provenance.parquet"
    artifacts.write_games(path, [_artifact_game()], metadata)

    assert artifacts.read_games(path).schema.metadata[b"extra_provenance"] == b"allowed"
