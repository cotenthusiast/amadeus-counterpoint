import random
from types import SimpleNamespace

import chess
import pytest
import torch

from amadeus_counterpoint.data.dataset import (
    ChessDataset,
    iter_game_examples,
    iter_shard_records,
    result_to_value_target,
    shuffle_buffer,
)
from amadeus_counterpoint.data.preprocess import write_shard
from amadeus_counterpoint.encoding import (
    HISTORY_LENGTH,
    encode_history,
    legal_move_mask,
    move_to_policy_index,
)


def _legal_record(
    *ucis: str, white_elo=1500, black_elo=1900, result="1-0", eligible_ply_count=None,
) -> dict:
    """Build a GameRecord, validating every move is legal by replaying it."""
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)  # raises chess.IllegalMoveError if not legal
    if eligible_ply_count is None:
        eligible_ply_count = len(ucis)
    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "moves": list(ucis),
        "eligible_ply_count": eligible_ply_count,
    }


# --- result_to_value_target --------------------------------------------------


def test_value_target_perspective_flips_with_side_to_move():
    assert result_to_value_target("1-0", chess.WHITE) == 2
    assert result_to_value_target("1-0", chess.BLACK) == 0
    assert result_to_value_target("0-1", chess.WHITE) == 0
    assert result_to_value_target("0-1", chess.BLACK) == 2
    assert result_to_value_target("1/2-1/2", chess.WHITE) == 1
    assert result_to_value_target("1/2-1/2", chess.BLACK) == 1


# --- iter_game_examples: full cross-check against independent recomputation -


def test_examples_match_independent_recomputation_including_castling():
    moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7",
    ]
    record = _legal_record(*moves, white_elo=1500, black_elo=1900, result="1-0")

    # history_mask_prob=0: this test cross-checks encode_history(history)
    # against the unmasked recomputation below, so augmentation must be off.
    examples = list(iter_game_examples(record, history_mask_prob=0.0))
    assert len(examples) == len(moves)

    board = chess.Board()
    history = [board.copy(stack=False)]

    for i, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)

        if board.turn == chess.WHITE:
            expected_player_elo, expected_opponent_elo = 1500, 1900
        else:
            expected_player_elo, expected_opponent_elo = 1900, 1500

        expected_x = encode_history(history)
        expected_policy = move_to_policy_index(move, board)
        expected_value = result_to_value_target("1-0", board.turn)
        expected_mask = legal_move_mask(board)

        ex = examples[i]
        assert torch.equal(ex["x"], expected_x)
        assert ex["player_elo"] == expected_player_elo
        assert ex["opponent_elo"] == expected_opponent_elo
        assert ex["policy_target"] == expected_policy
        assert ex["value_target"] == expected_value
        assert torch.equal(ex["legal_mask"], expected_mask)
        assert bool(ex["legal_mask"][ex["policy_target"]])

        board.push(move)
        history.append(board.copy(stack=False))
        if len(history) > HISTORY_LENGTH:
            history.pop(0)

    # the 9th ply (index 8) is White's kingside castle
    assert moves[8] == "e1g1"


def test_en_passant_target_is_legal_and_correctly_indexed():
    moves = ["e2e4", "h7h6", "e4e5", "d7d5", "e5d6"]  # e5d6 is an en passant capture
    record = _legal_record(*moves)

    examples = list(iter_game_examples(record))
    last = examples[-1]

    board = chess.Board()
    for uci in moves[:-1]:
        board.push_uci(uci)
    expected_policy = move_to_policy_index(chess.Move.from_uci("e5d6"), board)

    assert last["policy_target"] == expected_policy
    assert bool(last["legal_mask"][last["policy_target"]])


def test_promotion_target_is_legal_and_correctly_indexed():
    # White marches the b-pawn down to capture-promote on a8.
    moves = [
        "a2a4", "h7h5",
        "a4a5", "h5h4",
        "a5a6", "h4h3",
        "a6b7", "h3g2",
        "b7a8q",
    ]
    record = _legal_record(*moves)

    examples = list(iter_game_examples(record))
    last = examples[-1]

    board = chess.Board()
    for uci in moves[:-1]:
        board.push_uci(uci)
    expected_policy = move_to_policy_index(chess.Move.from_uci("b7a8q"), board)

    assert last["policy_target"] == expected_policy
    assert last["policy_target"] >= 4096  # promotion range
    assert bool(last["legal_mask"][last["policy_target"]])


def test_every_produced_policy_target_is_legal():
    games = [
        _legal_record("e2e4", "e7e5", "g1f3", "b8c6"),
        _legal_record("d2d4", "d7d5", "c2c4", "e7e6"),
        _legal_record("e2e4", "h7h6", "e4e5", "d7d5", "e5d6"),
    ]

    for record in games:
        for example in iter_game_examples(record):
            assert bool(example["legal_mask"][example["policy_target"]])


def test_short_game_uses_every_position():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    record = _legal_record(*moves)

    examples = list(iter_game_examples(record, max_positions_per_game=32))

    assert len(examples) == len(moves)


def test_game_at_exactly_the_cap_uses_every_position():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 8  # 32 plies, no early game-ending condition
    record = _legal_record(*moves)

    examples = list(iter_game_examples(record, max_positions_per_game=32))

    assert len(examples) == 32


def test_long_game_samples_exactly_the_cap_deterministically_under_fixed_seed():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 10  # 40 plies
    record = _legal_record(*moves)

    examples_a = list(iter_game_examples(record, rng=random.Random(123)))
    examples_b = list(iter_game_examples(record, rng=random.Random(123)))
    examples_c = list(iter_game_examples(record, rng=random.Random(456)))

    assert len(examples_a) == 32
    # deterministic given a fixed seed
    assert [e["policy_target"] for e in examples_a] == [e["policy_target"] for e in examples_b]
    # a different seed samples a different subset (overwhelmingly likely for 40 choose 32)
    assert [e["policy_target"] for e in examples_a] != [e["policy_target"] for e in examples_c]


# --- eligible_ply_count: sampling stays within the time-pressure cutoff ----


def test_excluded_plies_past_the_cutoff_are_never_yielded():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 5  # 20 plies
    record = _legal_record(*moves, eligible_ply_count=6)

    examples = list(iter_game_examples(record))

    assert len(examples) == 6  # under the cap, so every eligible ply is used


def test_max_positions_sampling_operates_over_eligible_plies_only():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 15  # 60 plies, but only the first 40 are eligible
    record = _legal_record(*moves, eligible_ply_count=40)

    # Spy on the rng actually used for sampling, to directly confirm the
    # population sampled from is range(eligible_ply_count), not range(len(moves)).
    rng = random.Random(1)
    captured = {}
    original_sample = rng.sample

    def spy_sample(population, k):
        captured["population"] = list(population)
        captured["k"] = k
        return original_sample(population, k)

    rng.sample = spy_sample

    examples = list(iter_game_examples(record, rng=rng))

    assert captured["population"] == list(range(40))
    assert captured["k"] == 32
    assert len(examples) == 32  # capped, even though 40 plies are eligible


def test_fewer_eligible_plies_than_cap_yields_all_of_them():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 15  # 60 plies, far more than the 32-position cap
    record = _legal_record(*moves, eligible_ply_count=10)

    examples = list(iter_game_examples(record))

    assert len(examples) == 10


def test_record_without_eligible_ply_count_field_treats_every_ply_as_eligible():
    # Backward-compatible fallback for hand-built records missing the field.
    record = {
        "white_elo": 1500,
        "black_elo": 1500,
        "result": "1-0",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
    }

    examples = list(iter_game_examples(record))

    assert len(examples) == 4


def test_history_length_capped_at_eight():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "e1g1", "d7d6"]
    record = _legal_record(*moves)

    examples = list(iter_game_examples(record))

    for ex in examples:
        assert ex["x"].shape == (64, 12 * HISTORY_LENGTH)


# --- iter_game_examples: 5% random history masking (Chessformer App. E) ----


class _FixedMaskRng:
    """Stand-in exposing only the random.Random calls history masking uses.

    `trigger` controls whether `rng.random() < history_mask_prob` fires;
    `keep_previous` controls how many previous boards `rng.randint(0, n)`
    reports as retained (clamped into range, like the real randint).
    """

    def __init__(self, trigger: bool, keep_previous: int = 0):
        self._trigger = trigger
        self._keep_previous = keep_previous

    def random(self):
        return 0.0 if self._trigger else 1.0

    def randint(self, a, b):
        return max(a, min(self._keep_previous, b))


_MASK_TEST_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]


def _boards_after(moves):
    board = chess.Board()
    boards = [board.copy(stack=False)]
    for uci in moves:
        board.push_uci(uci)
        boards.append(board.copy(stack=False))
    return boards


def test_history_masking_skipped_when_augmentation_does_not_trigger():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)  # P0..P5, last ply's history is P0..P4

    examples = list(
        iter_game_examples(record, rng=_FixedMaskRng(trigger=False, keep_previous=0))
    )

    assert torch.equal(examples[-1]["x"], encode_history(boards[:5]))


def test_history_masking_keeps_current_board_regardless_of_retained_count():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)
    current_board_encoding = encode_history([boards[4]])[:, -12:]

    for keep_previous in range(5):
        examples = list(
            iter_game_examples(
                record, rng=_FixedMaskRng(trigger=True, keep_previous=keep_previous)
            )
        )
        assert torch.equal(examples[-1]["x"][:, -12:], current_board_encoding)


def test_history_masking_retains_most_recent_previous_boards():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)

    examples = list(
        iter_game_examples(record, rng=_FixedMaskRng(trigger=True, keep_previous=2))
    )

    # ply 4's history is P0..P4; keeping 2 previous boards keeps P2, P3, P4.
    assert torch.equal(examples[-1]["x"], encode_history(boards[2:5]))


def test_history_masking_zero_retained_uses_only_current_board():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)

    examples = list(
        iter_game_examples(record, rng=_FixedMaskRng(trigger=True, keep_previous=0))
    )

    assert torch.equal(examples[-1]["x"], encode_history([boards[4]]))


def test_history_masking_retaining_all_available_matches_unmasked_history():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)

    examples = list(
        iter_game_examples(record, rng=_FixedMaskRng(trigger=True, keep_previous=4))
    )

    assert torch.equal(examples[-1]["x"], encode_history(boards[:5]))


def test_history_masking_still_pads_from_earliest_retained_board():
    record = _legal_record(*_MASK_TEST_MOVES)
    boards = _boards_after(_MASK_TEST_MOVES)

    examples = list(
        iter_game_examples(record, rng=_FixedMaskRng(trigger=True, keep_previous=1))
    )

    # ply 4's history is P0..P4; keeping 1 previous board keeps P3, P4, then
    # existing padding prepends copies of P3 (not the original game start P0).
    assert torch.equal(examples[-1]["x"], encode_history(boards[3:5]))


def test_history_masking_is_deterministic_under_fixed_rng_seed():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    moves = cycle * 5  # 20 plies, well under the sampling cap
    record = _legal_record(*moves)

    examples_a = list(
        iter_game_examples(record, history_mask_prob=1.0, rng=random.Random(123))
    )
    examples_b = list(
        iter_game_examples(record, history_mask_prob=1.0, rng=random.Random(123))
    )
    examples_c = list(
        iter_game_examples(record, history_mask_prob=1.0, rng=random.Random(456))
    )

    xs_a = [e["x"] for e in examples_a]
    xs_b = [e["x"] for e in examples_b]
    xs_c = [e["x"] for e in examples_c]

    assert all(torch.equal(a, b) for a, b in zip(xs_a, xs_b))
    assert any(not torch.equal(a, c) for a, c in zip(xs_a, xs_c))


def test_encode_history_stays_deterministic_and_unaffected_by_global_rng_state():
    # encode_history() is the deterministic encoder also used for eval/inference;
    # history masking must live in iter_game_examples, not leak in here.
    boards = _boards_after(_MASK_TEST_MOVES)

    random.seed(0)
    first = encode_history(boards)
    random.random()  # perturb any accidentally-shared global RNG state
    random.random()
    second = encode_history(boards)

    assert torch.equal(first, second)


# --- iter_shard_records -------------------------------------------------


def test_iter_shard_records_round_trips_through_parquet(tmp_path):
    records = [
        {"white_elo": 1500, "black_elo": 1600, "result": "1-0", "moves": ["e2e4"]},
        {"white_elo": 1200, "black_elo": 1300, "result": "0-1", "moves": ["d2d4", "d7d5"]},
    ]
    path = tmp_path / "shard.parquet"
    write_shard(records, path)

    read_back = list(iter_shard_records(path, batch_size=1))

    assert read_back == records


# --- shuffle_buffer -------------------------------------------------------


def test_shuffle_buffer_preserves_every_element_exactly_once():
    items = list(range(200))
    rng = random.Random(42)

    output = list(shuffle_buffer(iter(items), buffer_size=16, rng=rng))

    assert sorted(output) == items
    assert len(output) == len(items)


def test_shuffle_buffer_passthrough_when_disabled():
    items = list(range(20))
    rng = random.Random(0)

    output = list(shuffle_buffer(iter(items), buffer_size=1, rng=rng))

    assert output == items


def test_shuffle_buffer_actually_reorders():
    items = list(range(200))
    rng = random.Random(7)

    output = list(shuffle_buffer(iter(items), buffer_size=50, rng=rng))

    assert output != items
    assert sorted(output) == items


# --- ChessDataset ----------------------------------------------------------


def _make_shard(tmp_path, name, white_elo):
    path = tmp_path / name
    write_shard(
        [{"white_elo": white_elo, "black_elo": 999, "result": "1-0", "moves": ["e2e4"]}],
        path,
    )
    return path


def test_chess_dataset_raises_when_no_shards_found(tmp_path):
    with pytest.raises(ValueError):
        ChessDataset(tmp_path)


def test_chess_dataset_rejects_negative_shuffle_buffer(tmp_path):
    _make_shard(tmp_path, "shard_00000.parquet", 1111)

    with pytest.raises(ValueError):
        ChessDataset(tmp_path, shuffle_buffer_size=-1)


def test_chess_dataset_single_process_reads_all_shards(tmp_path, monkeypatch):
    _make_shard(tmp_path, "shard_00000.parquet", 1111)
    _make_shard(tmp_path, "shard_00001.parquet", 2222)

    monkeypatch.setattr(
        "amadeus_counterpoint.data.dataset.get_worker_info", lambda: None
    )

    dataset = ChessDataset(tmp_path, shuffle_buffer_size=0)
    player_elos = sorted(ex["player_elo"] for ex in dataset)

    assert player_elos == [1111, 2222]


def test_chess_dataset_workers_receive_disjoint_shards(tmp_path, monkeypatch):
    _make_shard(tmp_path, "shard_00000.parquet", 1111)
    _make_shard(tmp_path, "shard_00001.parquet", 2222)
    _make_shard(tmp_path, "shard_00002.parquet", 3333)

    dataset = ChessDataset(tmp_path, shuffle_buffer_size=0)

    def elos_for_worker(worker_id, num_workers):
        monkeypatch.setattr(
            "amadeus_counterpoint.data.dataset.get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=num_workers),
        )
        return sorted(ex["player_elo"] for ex in dataset)

    worker0 = elos_for_worker(0, 2)
    worker1 = elos_for_worker(1, 2)

    assert set(worker0).isdisjoint(worker1)
    assert sorted(worker0 + worker1) == [1111, 2222, 3333]


def test_chess_dataset_extra_workers_get_no_shards_without_crashing(tmp_path, monkeypatch):
    _make_shard(tmp_path, "shard_00000.parquet", 1111)

    dataset = ChessDataset(tmp_path, shuffle_buffer_size=0)

    monkeypatch.setattr(
        "amadeus_counterpoint.data.dataset.get_worker_info",
        lambda: SimpleNamespace(id=4, num_workers=5),
    )

    assert list(dataset) == []
