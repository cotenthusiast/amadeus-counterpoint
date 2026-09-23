import importlib
import importlib.util
import json
from pathlib import Path

import chess
import torch
from _helpers import CONFIG, replay_and_check_legal

from amadeus_counterpoint.chess import create_board
from amadeus_counterpoint.encoding import (
    encode_history,
    legal_move_mask,
    move_to_policy_index,
)
from amadeus_counterpoint.evaluation.generation import (
    batch,
    guarded_generation,
    method2_personalized,
)
from amadeus_counterpoint.evaluation.generation.experiment import (
    generate_method3_cell_batched,
)
from amadeus_counterpoint.evaluation.generation.strength_guardrail import (
    StrengthGuardrailConfig,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.models.style_scoring import score_candidates
from amadeus_counterpoint.training.checkpoints import (
    save_method1_checkpoint,
    save_method2_checkpoint,
)

STYLE_DIM = 16


class _FixedLogitsChessformer(Chessformer):
    def __init__(self, *args, fixed_logits, override_logits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_logits = fixed_logits
        self.override_logits = override_logits if override_logits is not None else fixed_logits
        self.calls = []

    def forward(self, x, player_elo, opponent_elo, player_emb_override=None):
        self.calls.append(None if player_emb_override is None else player_emb_override.detach().clone())
        logits = self.override_logits if player_emb_override is not None else self.fixed_logits
        return logits.expand(x.shape[0], -1).clone(), torch.zeros(x.shape[0], 3)


class _FakeEnginePool:
    def evaluate_many(self, tasks, depth):
        return [[0.0] * len(ucis) for _fen, ucis in tasks]


def _ranked_logits(moves):
    board = create_board()
    logits = torch.full((1, 4352), float("-inf"))
    for rank, uci in enumerate(moves, start=1):
        logits[0, move_to_policy_index(chess.Move.from_uci(uci), board)] = rank
    return logits


def _split_wrappers():
    generic = _ranked_logits(["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"])
    personalized = _ranked_logits(["b2b3", "c2c4", "g1f3", "d2d4", "e2e4"])
    base = _FixedLogitsChessformer(**CONFIG, fixed_logits=generic, override_logits=personalized)
    wrapper_a = PersonalizedChessformer(base, 1600.0, "p0")
    wrapper_b = PersonalizedChessformer(base, 1800.0, "p1")
    wrapper_a.z_player.data.fill_(1.0)
    wrapper_b.z_player.data.fill_(2.0)
    return base, wrapper_a, wrapper_b


def _style_stack():
    return (
        MoveStyleCNN(style_dim=STYLE_DIM),
        PlayerStyleTable(num_players=2, style_dim=STYLE_DIM),
        StyleResidual(style_dim=STYLE_DIM),
    )


def _m3_script_module():
    path = Path(__file__).parents[2] / "scripts" / "generate_method3_production.py"
    spec = importlib.util.spec_from_file_location("generate_method3_production", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m3_personalized_base_is_used_before_m2_candidate_selection(monkeypatch):
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 1)
    base, wrapper_a, _wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()

    games = method2_personalized.play_games_method2(
        base, cnn, table, residual, 0, chess.WHITE, 1600.0, [1800.0], 1, [7],
        personalized_base=wrapper_a,
    )

    assert games[0]["moves"] == ["e2e4"]
    assert base.calls[0] is not None


def test_m3_personalized_logits_and_candidates_match_direct_m1_forward():
    _base, wrapper_a, _wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()
    board = create_board()
    x = encode_history([board.copy(stack=False)]).unsqueeze(0)
    player_elo = torch.tensor([1600.0])
    opponent_elo = torch.tensor([1800.0])
    mask = legal_move_mask(board).unsqueeze(0)
    player_id = torch.tensor([0])

    direct_logits, _ = wrapper_a(x, player_elo, opponent_elo)
    output = score_candidates(
        wrapper_a, cnn, table, residual, x, player_elo, opponent_elo, player_id, mask, 1,
        target_index=None,
    )

    assert torch.equal(output.masked_logits, direct_logits)
    assert output.candidate_indices[0, 0].item() == move_to_policy_index(chess.Move.from_uci("e2e4"), board)


def test_m3_generic_gg_uses_the_existing_generic_m2_path(monkeypatch):
    base, wrapper_a, wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()
    captured = {}

    def fake_play_games(received_base, white_elos, black_elos, seeds):
        captured["base"] = received_base
        return [{
            "white_elo": white_elos[0], "black_elo": black_elos[0],
            "result": None, "censored": True, "moves": [],
        }]

    monkeypatch.setattr(batch, "play_games", fake_play_games)
    games = generate_method3_cell_batched(
        wrapper_a, wrapper_b, base, cnn, table, residual, 0, 1,
        chess.WHITE, "GG", 1600.0, 1800.0, 5, "0__1", 1, 123, "m3-test",
    )

    assert captured["base"] is base
    assert games[0]["white_representation_identity"] == "global"
    assert games[0]["black_representation_identity"] == "global"


def test_m3_changes_top_k_before_style_reranking():
    base, wrapper_a, _wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()
    board = create_board()
    x = encode_history([board.copy(stack=False)]).unsqueeze(0)
    elos = torch.tensor([1600.0])
    mask = legal_move_mask(board).unsqueeze(0)
    player_id = torch.tensor([0])

    generic = score_candidates(base, cnn, table, residual, x, elos, elos, player_id, mask, 1, None)
    personalized = score_candidates(wrapper_a, cnn, table, residual, x, elos, elos, player_id, mask, 1, None)

    assert generic.candidate_indices.tolist() != personalized.candidate_indices.tolist()


def test_m3_ab_routes_each_mover_to_that_players_m1_wrapper(monkeypatch):
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 2)
    base, wrapper_a, wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()

    games = method2_personalized.play_games_method2_ab(
        base, cnn, table, residual, 0, 1, chess.WHITE, 1600.0, 1800.0, 1, [9],
        personalized_base_a=wrapper_a, personalized_base_b=wrapper_b,
    )

    replay_and_check_legal(games[0]["moves"])
    assert torch.equal(base.calls[0].squeeze(0), wrapper_a.z_player.detach())
    assert torch.equal(base.calls[1].squeeze(0), wrapper_b.z_player.detach())


def test_m3_guarded_single_sided_routes_personalized_policy(monkeypatch):
    monkeypatch.setattr(guarded_generation, "MAX_PLIES", 1)
    base, wrapper_a, _wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()
    config = StrengthGuardrailConfig(lam=0.0, k=1, cheap_depth=1)

    games = guarded_generation.play_games_method2_guarded(
        base, cnn, table, residual, 0, chess.WHITE, 1600.0, [1800.0], 1, [7],
        config, _FakeEnginePool(), personalized_base=wrapper_a,
    )

    assert games[0]["moves"] == ["e2e4"]
    assert base.calls[0] is not None


def test_m3_guarded_ab_routes_each_mover_to_that_players_m1_wrapper(monkeypatch):
    monkeypatch.setattr(guarded_generation, "MAX_PLIES", 2)
    base, wrapper_a, wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()
    config = StrengthGuardrailConfig(lam=0.0, k=1, cheap_depth=1)

    games = guarded_generation.play_games_method2_ab_guarded(
        base, cnn, table, residual, 0, 1, chess.WHITE, 1600.0, 1800.0, 1, [9],
        config, _FakeEnginePool(), personalized_base_a=wrapper_a,
        personalized_base_b=wrapper_b,
    )

    replay_and_check_legal(games[0]["moves"])
    assert torch.equal(base.calls[0].squeeze(0), wrapper_a.z_player.detach())
    assert torch.equal(base.calls[1].squeeze(0), wrapper_b.z_player.detach())


def test_m3_cell_uses_distinct_seed_namespace_and_condition_identities(monkeypatch):
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 0)
    monkeypatch.setattr("amadeus_counterpoint.evaluation.generation.batch.MAX_PLIES", 0)
    base, wrapper_a, wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()

    games = generate_method3_cell_batched(
        wrapper_a, wrapper_b, base, cnn, table, residual,
        0, 1,
        chess.WHITE, "AG", 1600.0, 1800.0, 5, "0__1", 2, 123, "m3-test",
        chunk_size=1,
    )

    assert len(games) == 2
    assert all(game["censored"] and game["moves"] == [] for game in games)
    assert all(game["white_representation_identity"] == "p0" for game in games)
    assert all(game["black_representation_identity"] == "global" for game in games)
    assert games[0]["seed"] != games[1]["seed"]
    assert games[0]["dyad"] == "0__1"


def test_m3_default_arguments_preserve_m2_output(monkeypatch):
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 1)
    base, _wrapper_a, _wrapper_b = _split_wrappers()
    cnn, table, residual = _style_stack()

    without_new_argument = method2_personalized.play_games_method2(
        base, cnn, table, residual, 0, chess.WHITE, 1600.0, [1800.0], 1, [11],
    )
    with_explicit_default = method2_personalized.play_games_method2(
        base, cnn, table, residual, 0, chess.WHITE, 1600.0, [1800.0], 1, [11],
        personalized_base=None,
    )

    assert without_new_argument == with_explicit_default


def test_m3_production_runner_is_generation_only():
    source = _m3_script_module().__file__
    text = Path(source).read_text(encoding="utf-8")
    assert "Optimizer" not in text
    assert "train_method" not in text
    assert "save_method" not in text


def test_m3_production_runner_writes_distinct_method_metadata(
    tmp_path, monkeypatch,
):
    module = _m3_script_module()
    for name, value in {
        "D_MODEL": CONFIG["d_model"], "NUM_HEADS": CONFIG["num_heads"],
        "NUM_LAYERS": CONFIG["num_layers"], "D1": CONFIG["d1"], "D2": CONFIG["d2"],
        "D3": CONFIG["d3"], "D_FF": CONFIG["d_ff"], "ELO_DIM": CONFIG["elo_dim"],
        "HEAD_HID_DIM": CONFIG["head_hid_dim"], "RAW_INPUT_DIM": CONFIG["input_dim"],
        "DROPOUT": CONFIG["dropout"],
    }.items():
        monkeypatch.setattr(module, name, value)

    base = Chessformer(**CONFIG)
    base_path = tmp_path / "base.pt"
    torch.save({"model": base.state_dict()}, base_path)
    identities = {"0": "Player Zero", "1": "Player One"}
    checkpoints = {}
    for pid, elo in [("0", 1600.0), ("1", 1900.0)]:
        wrapper = PersonalizedChessformer(base, elo, identities[pid])
        path = tmp_path / f"player_{pid}.pt"
        save_method1_checkpoint(path, wrapper)
        checkpoints[pid] = str(path)
    cnn, table, residual = _style_stack()
    m2_path = tmp_path / "m2.pt"
    save_method2_checkpoint(m2_path, cnn, table, residual, k=5, style_dim=STYLE_DIM)
    elos_path = tmp_path / "elos.json"
    elos_path.write_text(json.dumps({"0": 1600.0, "1": 1900.0}))

    monkeypatch.setattr(module, "generate_method3_cell_batched", lambda *args, **kwargs: [{
        "game_index": 0, "seed": 1, "dyad": "0__1", "condition": "GG",
        "orientation": "A_WHITE", "result": None, "censored": True, "moves": [],
        "white_elo": 1600.0, "black_elo": 1900.0, "checkpoint_identity": "m3",
        "white_representation_identity": "global", "black_representation_identity": "global",
    }])
    monkeypatch.setattr(
        "sys.argv", ["prog", "--base-checkpoint", str(base_path),
        "--player-checkpoints", json.dumps(checkpoints), "--identities", json.dumps(identities),
        "--method2-checkpoint", str(m2_path), "--representative-elos", str(elos_path),
        "--player-ids", "0,1", "--num-players", "2", "--style-dim", str(STYLE_DIM), "--k", "5",
        "--output-root", str(tmp_path / "out"), "--root-seed", "1",
        "--games-per-orientation", "1", "--checkpoint-identity", "m3",
        "--protocol-version", "test", "--code-commit", "test",
        "--only-dyad-a", "0", "--only-dyad-b", "1", "--only-condition", "GG",
        "--only-orientation", "A_WHITE"],
    )

    module.main()
    import pyarrow.parquet as pq
    path = tmp_path / "out" / "method3_hybrid" / "0__1" / "GG_A_WHITE.parquet"
    metadata = pq.read_table(path).schema.metadata
    assert metadata[b"method_identifier"] == b"method3_hybrid"
    assert b"m2_checkpoint" in metadata
