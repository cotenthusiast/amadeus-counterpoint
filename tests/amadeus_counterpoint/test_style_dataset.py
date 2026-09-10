import chess
import torch
from torch.utils.data import DataLoader

from _helpers import CONFIG
from amadeus_counterpoint.data.style_dataset import (
    StyleDataset,
    balance_by_phase,
    build_style_datasets,
    filter_records_by_player,
    list_decisions,
    split_games_by_player,
)
from amadeus_counterpoint.encoding import encode_history, legal_move_mask, move_to_policy_index
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.style_trainer import StyleTrainer

# Both records replay the same six plies (1.e4 e5 2.Nf3 Nc6 3.Bb5 a6) so the
# only difference between them is which color is the target mover.
WHITE_MOVER_RECORD = {
    "player_id": 0,
    "mover_color": "white",
    "white_elo": 2800,
    "black_elo": 2600,
    "result": "1-0",
    "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
}

BLACK_MOVER_RECORD = {
    "player_id": 1,
    "mover_color": "black",
    "white_elo": 2600,
    "black_elo": 2900,
    "result": "0-1",
    "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
}


def _many_games_for_player(player_id, n):
    records = []
    for _ in range(n):
        records.append(
            {
                "player_id": player_id,
                "mover_color": "white",
                "white_elo": 2700,
                "black_elo": 2600,
                "result": "1-0",
                "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            }
        )
    return records


# --- B. mover filtering ---------------------------------------------------


def test_list_decisions_only_includes_white_target_plies():
    decisions = list_decisions([WHITE_MOVER_RECORD])
    assert decisions == [(0, 0), (0, 2), (0, 4)]


def test_list_decisions_only_includes_black_target_plies():
    decisions = list_decisions([BLACK_MOVER_RECORD])
    assert decisions == [(0, 1), (0, 3), (0, 5)]


# --- C. Elo semantics -------------------------------------------------------


def test_elo_semantics_white_mover():
    dataset = StyleDataset([WHITE_MOVER_RECORD], [(0, 0)])
    item = dataset[0]
    assert item["player_elo"] == 2800.0
    assert item["opponent_elo"] == 2600.0


def test_elo_semantics_black_mover():
    dataset = StyleDataset([BLACK_MOVER_RECORD], [(0, 1)])
    item = dataset[0]
    assert item["player_elo"] == 2900.0
    assert item["opponent_elo"] == 2600.0


# --- D. Black canonicalization ----------------------------------------------


def test_black_canonicalization_matches_existing_encoding_utilities():
    dataset = StyleDataset([BLACK_MOVER_RECORD], [(0, 1)])
    item = dataset[0]

    # Replay to ply 1 (one White move played) using the same primitives
    # single.py/dataset.py already use, and confirm the Dataset's output for
    # a Black decision is identical -- still the existing mover-as-White
    # canonical representation, not a second orientation scheme.
    board = chess.Board()
    board.push_uci("e2e4")
    history = [chess.Board(), board.copy(stack=False)]
    expected_x = encode_history(history)
    expected_target = move_to_policy_index(chess.Move.from_uci("e7e5"), board)

    assert torch.equal(item["x"], expected_x)
    assert item["policy_target"] == expected_target


# --- E. legal mask -----------------------------------------------------------


def test_legal_mask_matches_board_legal_moves():
    dataset = StyleDataset([WHITE_MOVER_RECORD], [(0, 2)])  # White to play move 3 (Nf3)
    item = dataset[0]

    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    expected_mask = legal_move_mask(board)

    assert torch.equal(item["legal_mask"], expected_mask)
    assert item["legal_mask"][item["policy_target"]]


# --- F. history --------------------------------------------------------------


def test_history_shape():
    dataset = StyleDataset([WHITE_MOVER_RECORD], [(0, 0)])
    item = dataset[0]
    assert item["x"].shape == (64, 96)


# --- G. game-level split ------------------------------------------------------


def test_split_is_deterministic_under_seed():
    records = _many_games_for_player(0, 10)
    train_a, val_a, _ = split_games_by_player(records, seed=42)
    train_b, val_b, _ = split_games_by_player(records, seed=42)
    assert train_a == train_b
    assert val_a == val_b


def test_split_is_game_level_with_no_leakage():
    records = _many_games_for_player(0, 10)
    train, val, _ = split_games_by_player(records, seed=1)

    train_ids = {id(record) for record in train}
    val_ids = {id(record) for record in val}

    assert train_ids.isdisjoint(val_ids)
    assert len(train) + len(val) == len(records)


def test_split_report_matches_actual_counts():
    records = _many_games_for_player(0, 10)
    train, val, report = split_games_by_player(records, seed=1)

    assert report[0]["total_eligible_games"] == 10
    assert report[0]["train_games"] == len(train)
    assert report[0]["val_games"] == len(val)


def test_split_happens_before_balancing_no_game_crosses_splits():
    records = _many_games_for_player(0, 10) + _many_games_for_player(1, 10)
    train_dataset, val_dataset, _ = build_style_datasets(records, split_seed=3, balance_seed=3)

    train_game_ids = {id(record) for record in train_dataset.records}
    val_game_ids = {id(record) for record in val_dataset.records}
    assert train_game_ids.isdisjoint(val_game_ids)


# --- I. phase balancing -------------------------------------------------------


def test_balance_by_phase_downsamples_to_smallest_phase_count():
    fake_records = [{"player_id": 0}]
    decisions = [(0, i) for i in range(6)]
    phases = ["opening", "opening", "opening", "middlegame", "middlegame", "endgame"]

    balanced, report = balance_by_phase(decisions, phases, fake_records, seed=0)

    assert report[0]["available_counts"] == {"opening": 3, "middlegame": 2, "endgame": 1}
    assert report[0]["balanced_count_per_phase"] == 1
    assert len(balanced) == 3
    assert len(balanced) == len(set(balanced))  # no duplicates


def test_balance_by_phase_is_deterministic_under_seed():
    fake_records = [{"player_id": 0}]
    decisions = [(0, i) for i in range(9)]
    phases = ["opening"] * 3 + ["middlegame"] * 3 + ["endgame"] * 3

    balanced_a, _ = balance_by_phase(decisions, phases, fake_records, seed=7)
    balanced_b, _ = balance_by_phase(decisions, phases, fake_records, seed=7)

    assert balanced_a == balanced_b


def test_balance_by_phase_reports_zero_when_a_phase_is_empty():
    fake_records = [{"player_id": 0}]
    decisions = [(0, 0), (0, 1)]
    phases = ["opening", "middlegame"]  # no endgame examples at all

    balanced, report = balance_by_phase(decisions, phases, fake_records, seed=0)

    assert report[0]["available_counts"]["endgame"] == 0
    assert report[0]["balanced_count_per_phase"] == 0
    assert balanced == []


def test_balance_by_phase_respects_max_decisions_per_player():
    fake_records = [{"player_id": 0}]
    decisions = [(0, i) for i in range(30)]
    phases = ["opening"] * 10 + ["middlegame"] * 10 + ["endgame"] * 10

    balanced, report = balance_by_phase(
        decisions, phases, fake_records, seed=0, max_decisions_per_player=6
    )

    assert report[0]["balanced_count_per_phase"] == 2  # 6 // 3
    assert len(balanced) == 6


def test_train_and_validation_are_balanced_independently():
    records = _many_games_for_player(0, 10)
    _train_dataset, _val_dataset, report = build_style_datasets(
        records, train_fraction=0.8, split_seed=5, balance_seed=5
    )
    # Independent computation, not shared state -- different dicts.
    assert report["train_balance"] is not report["val_balance"]


# --- J. StyleTrainer integration ---------------------------------------------


def test_dataloader_batch_feeds_directly_into_style_trainer():
    records = [WHITE_MOVER_RECORD, BLACK_MOVER_RECORD]
    decisions = list_decisions(records)
    dataset = StyleDataset(records, decisions)
    loader = DataLoader(dataset, batch_size=len(decisions), shuffle=False)
    batch = next(iter(loader))

    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=16)
    table = PlayerStyleTable(num_players=2, style_dim=16)
    residual = StyleResidual(style_dim=16, init_s=0.17)
    trainer = StyleTrainer(base, cnn, table, residual, k=5)

    # No reshaping/renaming/reformatting: the batch is passed to
    # train_epoch exactly as the DataLoader produced it.
    mean_loss = trainer.train_epoch([batch])

    assert torch.isfinite(torch.tensor(mean_loss))


def test_filter_records_by_player_never_mixes_identities():
    records = [WHITE_MOVER_RECORD, BLACK_MOVER_RECORD]

    filtered = filter_records_by_player(records, player_id=0)

    assert filtered == [WHITE_MOVER_RECORD]
    assert all(record["player_id"] == 0 for record in filtered)


def test_filter_records_by_player_returns_empty_for_an_absent_player():
    records = [WHITE_MOVER_RECORD, BLACK_MOVER_RECORD]

    assert filter_records_by_player(records, player_id=99) == []
