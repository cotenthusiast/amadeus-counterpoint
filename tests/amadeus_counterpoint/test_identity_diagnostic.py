import torch
from torch.utils.data import DataLoader

from _helpers import CONFIG
from amadeus_counterpoint.data.style_dataset import (
    StyleDataset,
    build_style_datasets,
    filter_records_by_player,
    list_decisions,
)
from amadeus_counterpoint.evaluation.identity_diagnostic import (
    RunningStats,
    build_method1_val_dataset,
    build_method2_val_datasets,
    evaluate_generic,
    evaluate_method1_representation,
    evaluate_method2_row,
    summarize_overall,
    summarize_row,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN

STYLE_DIM = 16
NUM_PLAYERS = 3


def _games_for_player(player_id, n, seed_offset=0):
    """n short, slightly varied games for one player (mover always white)."""
    records = []
    move_pool = [
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
    ]
    for i in range(n):
        moves = move_pool[(i + seed_offset) % len(move_pool)]
        records.append({
            "player_id": player_id,
            "mover_color": "white",
            "white_elo": 2000 + 10 * i,
            "black_elo": 1900 + 5 * i,
            "result": "1-0",
            "moves": moves,
        })
    return records


def _unbalanced_dataset_for_player(records, player_id):
    """A plain, non-phase-balanced StyleDataset over one player's records --
    used by the evaluate_* function tests below, which only need SOME
    non-empty dataset to drive the forward passes, not a production-faithful
    split. All the synthetic games here are only 6 plies (well within
    "opening" under data.phase's classifier), so balance_by_phase -- which
    requires all three phases present -- would zero everything out;
    build_method1_val_dataset/build_method2_val_datasets are exercised
    separately, on their own terms, by the split-reconstruction tests below."""
    player_records = filter_records_by_player(records, player_id)
    decisions = list_decisions(player_records)
    return StyleDataset(player_records, decisions)


def _all_player_records(n_per_player=12):
    records = []
    for pid in range(NUM_PLAYERS):
        records.extend(_games_for_player(pid, n_per_player, seed_offset=pid))
    return records


# --- validation-split reconstruction ----------------------------------------


def test_method1_val_dataset_matches_filter_then_split_exactly():
    """build_method1_val_dataset must reproduce train_method1_player.py's own
    call sequence byte-for-byte: filter to one player FIRST, then split."""
    records = _all_player_records()

    expected_records = filter_records_by_player(records, 1)
    _, expected_val, _ = build_style_datasets(
        expected_records, train_fraction=0.8, split_seed=0, balance_seed=0
    )

    actual_val = build_method1_val_dataset(records, 1, train_fraction=0.8, split_seed=0, balance_seed=0)

    assert actual_val.decisions == expected_val.decisions
    assert actual_val.records == expected_val.records


def test_method2_val_datasets_match_joint_split_then_filter_exactly():
    """build_method2_val_datasets must reproduce train_method2.py's own call
    sequence: split ALL players jointly first, then partition by player."""
    records = _all_player_records()

    _, expected_joint_val, _ = build_style_datasets(
        records, train_fraction=0.8, split_seed=0, balance_seed=0
    )
    expected_val_for_1 = [
        (gi, ply) for gi, ply in expected_joint_val.decisions
        if expected_joint_val.records[gi]["player_id"] == 1
    ]

    per_player = build_method2_val_datasets(records, [0, 1, 2], train_fraction=0.8, split_seed=0, balance_seed=0)

    assert per_player[1].decisions == expected_val_for_1


def test_method1_and_method2_reconstructions_genuinely_differ_for_the_same_player():
    """Documents the real divergence found before implementation: solo
    (filter-then-split) and joint (split-then-filter) reconstructions are NOT
    interchangeable for the same player/seed, because split_games_by_player
    consumes one shared RNG across players in sorted player_id order.

    Checked at the split_games_by_player layer directly (before phase
    balancing) -- these synthetic games are all 6-ply "opening"-only, so
    balance_by_phase (which build_method1_val_dataset/build_method2_val_datasets
    both apply) would zero out both sides' val sets, making a
    post-balancing comparison vacuous rather than a real test of the
    divergence."""
    from amadeus_counterpoint.data.style_dataset import split_games_by_player

    records = _all_player_records()

    _, solo_val, _ = split_games_by_player(filter_records_by_player(records, 1), train_fraction=0.8, seed=0)
    _, joint_val, _ = split_games_by_player(records, train_fraction=0.8, seed=0)
    joint_val_for_1 = [r for r in joint_val if r["player_id"] == 1]

    solo_moves = sorted(tuple(r["moves"]) for r in solo_val)
    joint_moves = sorted(tuple(r["moves"]) for r in joint_val_for_1)

    assert solo_moves and joint_moves  # sanity: both non-empty before comparing
    assert solo_moves != joint_moves


# --- generic baseline is genuinely unwrapped ---------------------------------


def test_evaluate_generic_matches_a_direct_unwrapped_base_call():
    base = Chessformer(**CONFIG)
    val_dataset = _unbalanced_dataset_for_player(_all_player_records(), 0)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    stats = evaluate_generic(base, dataloader, torch.device("cpu"))

    # Recompute directly with the raw base, no wrapper, real recorded Elos --
    # if evaluate_generic ever started routing through a wrapper/override,
    # this would diverge.
    base.eval()
    total_nll = 0.0
    correct = 0
    n = 0
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in dataloader:
            policy_logits, _ = base(batch["x"], batch["player_elo"], batch["opponent_elo"])
            masked = policy_logits.masked_fill(~batch["legal_mask"], torch.finfo(policy_logits.dtype).min)
            total_nll += F.cross_entropy(masked, batch["policy_target"], reduction="sum").item()
            correct += (masked.argmax(dim=-1) == batch["policy_target"]).sum().item()
            n += batch["policy_target"].shape[0]

    assert stats.n == n
    assert stats.correct == correct
    assert abs(stats.nll_sum - total_nll) < 1e-4


# --- Method 1 embedding override actually takes effect end-to-end -----------


def test_evaluate_method1_representation_reflects_which_wrapper_is_passed():
    base = Chessformer(**CONFIG)
    val_dataset = _unbalanced_dataset_for_player(_all_player_records(), 0)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    device = torch.device("cpu")

    wrapper_a = PersonalizedChessformer(base, nominal_elo=1200.0, identity="A")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=1200.0, identity="B")
    # Force genuinely different learned representations (as if from two
    # independently trained checkpoints), not just relying on random init.
    with torch.no_grad():
        wrapper_a.z_player.fill_(0.5)
        wrapper_b.z_player.fill_(-0.5)

    stats_a = evaluate_method1_representation(wrapper_a, dataloader, device)
    stats_b = evaluate_method1_representation(wrapper_b, dataloader, device)

    assert stats_a.n == stats_b.n == len(val_dataset)
    assert stats_a.mean_nll != stats_b.mean_nll


# --- Method 2 player_id override + representation-independent coverage ------


def test_evaluate_method2_row_reflects_player_id_override_and_shares_coverage():
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=NUM_PLAYERS, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.5)
    with torch.no_grad():
        table.embeddings.weight[0].fill_(1.0)
        table.embeddings.weight[1].fill_(-1.0)

    val_dataset = _unbalanced_dataset_for_player(_all_player_records(), 0)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    row = evaluate_method2_row(base, cnn, table, residual, k=3, dataloader=dataloader,
                                representation_ids=[0, 1], device=torch.device("cpu"))

    assert row.per_representation[0].n == row.per_representation[1].n == len(val_dataset)
    # Different style-table rows must produce different oracle-expanded
    # candidate-conditioned scores/loss.
    assert row.per_representation[0].mean_nll != row.per_representation[1].mean_nll
    assert 0.0 <= row.coverage <= 1.0

    # candidate_matched_generic has no style residual at all -- it must not
    # depend on which representation_ids were requested, and must differ
    # from both style-conditioned NLLs (residual is nonzero here).
    assert row.candidate_matched_generic.n == len(val_dataset)
    assert row.candidate_matched_generic.mean_nll != row.per_representation[0].mean_nll
    assert row.candidate_matched_generic.mean_nll != row.per_representation[1].mean_nll

    # Deployable accuracy: bounded, tracked per representation, and -- since
    # it's the raw-top-K-only (no oracle append) view -- never higher than
    # raw coverage would allow (a target outside raw top-K can never be the
    # deployable pick).
    for pid in (0, 1):
        acc = row.deployable_per_representation[pid].accuracy
        assert 0.0 <= acc <= 1.0
        assert row.deployable_per_representation[pid].n == len(val_dataset)
        assert acc <= row.coverage + 1e-9


def test_deployable_accuracy_never_credits_an_uncovered_target():
    """With k=1, only one candidate is ever a real (non-appended) contender,
    so raw coverage is necessarily low -- deployable accuracy must never
    exceed it, since an uncovered target can't be the deployable pick."""
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=NUM_PLAYERS, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.5)

    val_dataset = _unbalanced_dataset_for_player(_all_player_records(), 0)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    row = evaluate_method2_row(base, cnn, table, residual, k=1, dataloader=dataloader,
                                representation_ids=[0], device=torch.device("cpu"))

    assert row.deployable_per_representation[0].accuracy <= row.coverage + 1e-9


# --- rank/delta arithmetic ----------------------------------------------------


def test_summarize_row_rank_and_deltas_on_known_values():
    generic = RunningStats(nll_sum=20.0, correct=5, n=10)  # mean_nll=2.0
    per_representation = {
        0: RunningStats(nll_sum=5.0, correct=9, n=10),   # correct player: mean_nll=0.5 (best)
        1: RunningStats(nll_sum=30.0, correct=2, n=10),  # mean_nll=3.0
        2: RunningStats(nll_sum=15.0, correct=4, n=10),  # mean_nll=1.5
    }

    row = summarize_row(true_player=0, generic=generic, per_representation=per_representation)

    assert row["correct_nll"] == 0.5
    assert row["generic_nll"] == 2.0
    assert row["mean_wrong_nll"] == (3.0 + 1.5) / 2
    assert row["correct_rank_among_8"] == 1  # lowest NLL among {0.5, 3.0, 1.5}
    assert row["generic_minus_correct_nll"] == 2.0 - 0.5
    assert row["mean_wrong_minus_correct_nll"] == (3.0 + 1.5) / 2 - 0.5


def test_summarize_row_rank_when_correct_representation_is_not_best():
    generic = RunningStats(nll_sum=10.0, correct=5, n=10)
    per_representation = {
        0: RunningStats(nll_sum=25.0, correct=1, n=10),  # correct player: worst of the 3
        1: RunningStats(nll_sum=5.0, correct=9, n=10),
        2: RunningStats(nll_sum=15.0, correct=4, n=10),
    }

    row = summarize_row(true_player=0, generic=generic, per_representation=per_representation)

    assert row["correct_rank_among_8"] == 3
    assert row["generic_minus_correct_nll"] < 0  # generic beat "correct" here


def test_summarize_overall_aggregates_rows():
    rows = [
        {"correct_rank_among_8": 1, "generic_minus_correct_nll": 1.0, "mean_wrong_minus_correct_nll": 2.0},
        {"correct_rank_among_8": 3, "generic_minus_correct_nll": -0.5, "mean_wrong_minus_correct_nll": 0.5},
    ]

    overall = summarize_overall(rows)

    assert overall["num_players"] == 2
    assert overall["correct_rank_1_count"] == 1
    assert overall["mean_correct_rank"] == 2.0
    assert overall["mean_generic_minus_correct_nll"] == 0.25
    assert overall["mean_wrong_minus_correct_nll_avg"] == 1.25
