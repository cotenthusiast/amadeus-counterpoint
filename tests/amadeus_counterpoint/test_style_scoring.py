import torch
import torch.nn.functional as F

from _helpers import CONFIG, FakeBase
from amadeus_counterpoint.models.candidates import topk_target_coverage
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.models.style_scoring import score_candidates

STYLE_DIM = 16


def build_base() -> Chessformer:
    return Chessformer(**CONFIG)


def build_style(num_players: int = 4, style_dim: int = STYLE_DIM):
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=num_players, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    return cnn, table, residual


def random_batch(B: int, num_players: int = 4):
    x = torch.randn(B, 64, CONFIG["input_dim"])
    player_elo = torch.randint(0, 5000, (B,)).float()
    opponent_elo = torch.randint(0, 5000, (B,)).float()
    player_id = torch.randint(0, num_players, (B,))

    legal_mask = torch.zeros(B, 4352, dtype=torch.bool)
    for b in range(B):
        legal_indices = torch.randperm(4352)[:10]
        legal_mask[b, legal_indices] = True

    return x, player_elo, opponent_elo, player_id, legal_mask


def first_legal_index(legal_mask_row: torch.Tensor) -> torch.Tensor:
    return legal_mask_row.nonzero()[0, 0]


# --- shapes, real (random) base ---------------------------------------------


def test_scoring_shapes_inference_mode_no_target():
    base = build_base()
    cnn, table, residual = build_style()
    x, player_elo, opponent_elo, player_id, legal_mask = random_batch(3)

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask, k=5
    )

    assert output.candidate_indices.shape == (3, 5)
    assert output.candidate_scores.shape == (3, 5)
    assert output.candidate_valid.shape == (3, 5)
    assert output.local_target_index is None
    assert output.masked_logits.shape == (3, 4352)


def test_scoring_shapes_with_target_widen_to_k_plus_1():
    base = build_base()
    cnn, table, residual = build_style()
    x, player_elo, opponent_elo, player_id, legal_mask = random_batch(3)
    target = torch.stack([first_legal_index(legal_mask[b]) for b in range(3)])

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask,
        k=5, target_index=target,
    )

    assert output.candidate_indices.shape == (3, 6)
    assert output.candidate_scores.shape == (3, 6)
    assert output.candidate_valid.shape == (3, 6)
    assert output.local_target_index.shape == (3,)


def test_valid_slots_finite_invalid_slots_negative_infinity():
    base = build_base()
    cnn, table, residual = build_style()
    x, player_elo, opponent_elo, player_id, legal_mask = random_batch(2)
    target = torch.stack([first_legal_index(legal_mask[b]) for b in range(2)])

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask,
        k=5, target_index=target,
    )

    valid_scores = output.candidate_scores[output.candidate_valid]
    invalid_scores = output.candidate_scores[~output.candidate_valid]

    assert torch.isfinite(valid_scores).all()
    assert torch.all(invalid_scores == float("-inf"))


def test_target_always_valid_and_finite_after_append():
    base = build_base()
    cnn, table, residual = build_style()
    x, player_elo, opponent_elo, player_id, legal_mask = random_batch(3)
    target = torch.stack([first_legal_index(legal_mask[b]) for b in range(3)])

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask,
        k=5, target_index=target,
    )

    for b in range(3):
        local = output.local_target_index[b].item()
        assert output.candidate_indices[b, local].item() == target[b].item()
        assert output.candidate_valid[b, local].item()
        assert torch.isfinite(output.candidate_scores[b, local])


# --- deterministic difficult cases (fake base) ------------------------------

_LEGAL = [10, 20, 30, 40, 50]
_VALUES = [9.0, 7.0, 5.0, 3.0, 1.0]  # strictly descending, rank == list position


def _fixture(target_index: int, k: int = 3):
    policy_logits = torch.zeros(1, 4352)
    for idx, val in zip(_LEGAL, _VALUES):
        policy_logits[0, idx] = val

    legal_mask = torch.zeros(1, 4352, dtype=torch.bool)
    legal_mask[0, _LEGAL] = True

    base = FakeBase(policy_logits)
    cnn, table, residual = build_style(num_players=1)

    x = torch.randn(1, 64, CONFIG["input_dim"])
    player_elo = torch.tensor([1500.0])
    opponent_elo = torch.tensor([1500.0])
    player_id = torch.tensor([0])
    target = torch.tensor([target_index])

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask,
        k=k, target_index=target,
    )
    coverage = topk_target_coverage(output.masked_logits, k=k, target_index=target)
    return output, coverage, target


def test_target_outside_topk_is_appended_ce_finite_coverage_false():
    output, coverage, target = _fixture(target_index=50, k=3)  # rank 5th, outside top-3

    assert output.candidate_indices[0, :3].tolist() == [10, 20, 30]  # top-3 by raw base logit
    assert output.candidate_indices[0, 3].item() == 50  # appended
    assert output.candidate_valid[0, 3].item()
    assert output.local_target_index.item() == 3

    ce = F.cross_entropy(output.candidate_scores, output.local_target_index)
    assert torch.isfinite(ce)
    assert not coverage.item()


def test_target_inside_topk_not_duplicated_ce_finite_coverage_true():
    output, coverage, target = _fixture(target_index=20, k=3)  # rank 2nd, already in top-3

    assert output.candidate_indices[0, :3].tolist() == [10, 20, 30]
    assert not output.candidate_valid[0, 3].item()  # padded slot unused, no duplicate
    local = output.local_target_index.item()
    assert local == 1
    assert output.candidate_indices[0, local].item() == 20

    ce = F.cross_entropy(output.candidate_scores, output.local_target_index)
    assert torch.isfinite(ce)
    assert coverage.item()


def test_scoring_backward_succeeds():
    base = build_base()
    cnn, table, residual = build_style()
    x, player_elo, opponent_elo, player_id, legal_mask = random_batch(3)
    target = torch.stack([first_legal_index(legal_mask[b]) for b in range(3)])

    output = score_candidates(
        base, cnn, table, residual, x, player_elo, opponent_elo, player_id, legal_mask,
        k=5, target_index=target,
    )
    loss = F.cross_entropy(output.candidate_scores, output.local_target_index)
    loss.backward()

    assert any(p.grad is not None and torch.any(p.grad != 0) for p in cnn.parameters())
