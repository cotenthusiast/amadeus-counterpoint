"""Production-integration tests for the frozen strength-aware guardrail
wired into the batched generation path (guarded_generation.py +
experiment.py's generate_method1_cell_batched/generate_method2_cell_batched).

No real Stockfish process is spawned here: `FakeEnginePool` stands in for
`guarded_generation.StockfishEnginePool` with the exact same
`evaluate_many(tasks, depth) -> list[list[float]]` interface, so these
tests are fast and portable. The REAL end-to-end check against the actual
Stockfish 19 binary and real production checkpoints is the separate
reference-equivalence pass run on the cluster (see
FINAL_SAMPLER_REPORT.md's production-integration section), not this file.
"""

import chess
import pytest
import torch

from _helpers import CONFIG, replay_and_check_legal
from amadeus_counterpoint.chess import create_board
from amadeus_counterpoint.encoding import move_to_policy_index
from amadeus_counterpoint.evaluation.generation import guarded_generation
from amadeus_counterpoint.evaluation.generation.experiment import (
    generate_method1_cell_batched,
    generate_method2_cell_batched,
)
from amadeus_counterpoint.evaluation.generation.strength_guardrail import (
    FROZEN_PRODUCTION_CONFIG,
    StrengthGuardrailConfig,
    strength_reweight,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN

DYAD = "0_1"
ROOT_SEED = 20260913
STYLE_DIM = 16


@pytest.fixture(autouse=True)
def _cap_plies_for_fixed_logit_fakes(monkeypatch):
    """Every `_FixedLogitsChessformer` fake in this file returns the SAME
    logits regardless of board state (full position-independence, unlike a
    real model) -- past the very first ply, decoding those same policy
    indices against a DIFFERENT board can produce an illegal move. Capping
    MAX_PLIES to 1 keeps every test in this file to exactly the first move,
    which is all any of them actually inspect; harmless for the tests that
    use a real (tiny) Chessformer instead, which just get shorter games."""
    monkeypatch.setattr(guarded_generation, "MAX_PLIES", 1)


class FakeEnginePool:
    """`engine_pool` test double: `evaluate_many` returns a fixed loss per
    candidate UCI, looked up from a caller-supplied score table, with NO
    dependency on board FEN -- exercises the SELECTION mechanism against
    known candidate qualities without needing a real engine. `raise_on_call`
    lets a test simulate an engine-worker failure."""

    def __init__(self, score_by_uci=None, raise_on_call=False):
        self.score_by_uci = score_by_uci or {}
        self.raise_on_call = raise_on_call
        self.calls = []

    def evaluate_many(self, tasks, depth):
        if self.raise_on_call:
            raise RuntimeError("simulated engine worker failure")
        self.calls.append((list(tasks), depth))
        results = []
        for _fen, ucis in tasks:
            scores = [self.score_by_uci.get(u, 0.0) for u in ucis]
            best = max(scores) if scores else 0.0
            results.append([max(0.0, best - s) for s in scores])
        return results


class _FixedLogitsChessformer(Chessformer):
    """Real nn.Module (so `next(model.parameters())` works, matching
    production code's device detection), but forward() always returns
    caller-supplied FIXED logits regardless of input, instead of a real
    (randomly-initialized) forward pass -- full control over top-K ranking.

    Returns `override_logits` when a `player_emb_override` is passed (i.e.
    when called AS a PersonalizedChessformer's wrapped base, with its
    z_player substituted in) and `fixed_logits` otherwise (i.e. when called
    as the plain generic policy, via `wrapper.base(...)` or directly) -- so
    a test can assert the guarded path picked up the RIGHT one instead of
    both branches trivially agreeing because they returned the same thing.
    """

    def __init__(self, *args, fixed_logits=None, override_logits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_logits = fixed_logits
        self.override_logits = override_logits if override_logits is not None else fixed_logits
        self.calls = []

    def forward(self, x, player_elo, opponent_elo, player_emb_override=None):
        self.calls.append({
            "player_elo": player_elo.clone(), "opponent_elo": opponent_elo.clone(),
            "override": None if player_emb_override is None else player_emb_override.clone(),
        })
        batch = x.shape[0]
        logits = self.override_logits if player_emb_override is not None else self.fixed_logits
        return logits.expand(batch, -1).clone(), torch.zeros(batch, 3)


def _ranked_logits_favoring(moves_in_rank_order):
    """Strictly-decreasing logits over the given moves (index 0 = best),
    -inf everywhere else -- mirrors test_method2_generation.py's
    `_ranked_start_position_logits` pattern."""
    board = create_board()
    logits = torch.full((1, 4352), float("-inf"))
    n = len(moves_in_rank_order)
    for rank, uci in enumerate(moves_in_rank_order):
        move = chess.Move.from_uci(uci)
        logits[0, move_to_policy_index(move, board)] = float(n - rank)
    return logits


START_TOP5_UCIS = ["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"]


def _fixed_base(fixed_logits):
    return _FixedLogitsChessformer(**CONFIG, fixed_logits=fixed_logits)


# ---------------------------------------------------------------------------
# frozen spec
# ---------------------------------------------------------------------------

def test_frozen_production_config_matches_spec():
    assert FROZEN_PRODUCTION_CONFIG.k == 5
    assert FROZEN_PRODUCTION_CONFIG.lam == 2.0
    assert FROZEN_PRODUCTION_CONFIG.cheap_depth == 8
    assert FROZEN_PRODUCTION_CONFIG.threads == 1
    assert FROZEN_PRODUCTION_CONFIG.hash_mb == 128


# ---------------------------------------------------------------------------
# generic guarded path
# ---------------------------------------------------------------------------

def test_generic_guarded_calls_engine_pool_and_produces_legal_games():
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    pool = FakeEnginePool(score_by_uci={"d2d4": 100.0})  # make d4 the cheap-best, not e4
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = guarded_generation.play_games_guarded(base, [1600.0] * 6, [1600.0] * 6, list(range(6)), config, pool)

    assert len(pool.calls) > 0, "guardrail was never invoked"
    for g in games:
        replay_and_check_legal(g["moves"])


def test_generic_guarded_probabilities_sum_to_one_and_all_nonzero():
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    # one candidate catastrophically bad under the cheap search
    pool = FakeEnginePool(score_by_uci={"e2e4": 0.0, "d2d4": 0.0, "g1f3": 0.0, "c2c4": 0.0, "b2b3": -100000.0})
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    # drive the mechanism directly to inspect q, not just play a game
    x_indices, x_moves, p_behavior = guarded_generation.top_k_candidates(
        torch.softmax(base.fixed_logits.squeeze(0), dim=-1), create_board(), 5,
    )
    ucis = [m.uci() for m in x_moves]
    best = max(pool.score_by_uci.get(u, 0.0) for u in ucis)
    cheap_losses = [max(0.0, best - pool.score_by_uci.get(u, 0.0)) for u in ucis]
    q = strength_reweight(p_behavior, cheap_losses, config.lam)

    assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(q > 0.0)


def test_lambda_zero_integration_reproduces_topk_renormalized_behavior():
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    pool = FakeEnginePool(score_by_uci={"b2b3": 500.0})  # would matter a lot at lambda>0
    config_zero = StrengthGuardrailConfig(lam=0.0, k=5, cheap_depth=8)

    seed = 42
    games_guarded = guarded_generation.play_games_guarded(base, [1600.0], [1600.0], [seed], config_zero, pool)

    # manual reference: top-5 by p_behavior, renormalized, sampled with the same seed
    probs_full = torch.softmax(base.fixed_logits.squeeze(0), dim=-1)
    indices, moves, p_behavior = guarded_generation.top_k_candidates(probs_full, create_board(), 5)
    q_ref = strength_reweight(p_behavior, [0.0] * len(p_behavior), 0.0)
    generator = torch.Generator().manual_seed(seed)
    local_idx = int(torch.multinomial(q_ref, num_samples=1, generator=generator).item())
    expected_first_move = moves[local_idx].uci()

    assert games_guarded[0]["moves"][0] == expected_first_move


def test_lambda_zero_ignores_cheap_loss_entirely():
    p = [0.5, 0.3, 0.2]
    q_with_extreme_loss = strength_reweight(p, [0.0, 99999.0, 99999.0], 0.0)
    q_without_loss = strength_reweight(p, [0.0, 0.0, 0.0], 0.0)
    assert torch.allclose(q_with_extreme_loss, q_without_loss, atol=1e-9)


# ---------------------------------------------------------------------------
# fixed seeds / determinism / batched-vs-reference agreement
# ---------------------------------------------------------------------------

def test_fixed_seeds_reproduce_guarded_sampling():
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    pool = FakeEnginePool(score_by_uci={"d2d4": 40.0, "g1f3": -20.0})
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    seeds = [7, 11, 23, 99]
    games_a = guarded_generation.play_games_guarded(base, [1600.0] * 4, [1600.0] * 4, seeds, config, pool)
    games_b = guarded_generation.play_games_guarded(base, [1600.0] * 4, [1600.0] * 4, seeds, config, pool)
    assert [g["moves"] for g in games_a] == [g["moves"] for g in games_b]


def test_batched_and_single_game_reference_agree():
    """A batch of size 1 through the production guarded path must match a
    hand-rolled single-position reference built directly from
    strength_guardrail's primitives -- the same "batched vs unbatched
    agree" property the frozen module's own tests already establish for
    strength_reweight, now checked at the integration level."""
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    pool = FakeEnginePool(score_by_uci={"e2e4": 10.0, "d2d4": 30.0, "g1f3": -5.0, "c2c4": 0.0, "b2b3": -50.0})
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)
    seed = 123

    games = guarded_generation.play_games_guarded(base, [1600.0], [1600.0], [seed], config, pool)

    board = create_board()
    probs_full = torch.softmax(base.fixed_logits.squeeze(0), dim=-1)
    indices, moves, p_behavior = guarded_generation.top_k_candidates(probs_full, board, config.k)
    ucis = [m.uci() for m in moves]
    best = max(pool.score_by_uci.get(u, 0.0) for u in ucis)
    cheap_losses = [max(0.0, best - pool.score_by_uci.get(u, 0.0)) for u in ucis]
    q_ref = strength_reweight(p_behavior, cheap_losses, config.lam)
    generator = torch.Generator().manual_seed(seed)
    local_idx = int(torch.multinomial(q_ref, num_samples=1, generator=generator).item())

    assert games[0]["moves"][0] == moves[local_idx].uci()


# ---------------------------------------------------------------------------
# engine failure propagation
# ---------------------------------------------------------------------------

def test_engine_worker_failure_propagates_not_silently_falls_back():
    base = _fixed_base(_ranked_logits_favoring(START_TOP5_UCIS))
    pool = FakeEnginePool(raise_on_call=True)
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    with pytest.raises(RuntimeError, match="simulated engine worker failure"):
        guarded_generation.play_games_guarded(base, [1600.0], [1600.0], [1], config, pool)


# ---------------------------------------------------------------------------
# Method 1: correct player's own policy is used, both sides guarded
# ---------------------------------------------------------------------------

def _build_wrapper_with_split_logits(base_logits, override_logits, nominal_elo=1600.0):
    """`base_logits` is what a generic (no-override) forward returns --
    used for the opponent side in single-sided personalization.
    `override_logits` is what the SAME underlying model returns once the
    wrapper's z_player override is applied -- used for the wrapper's own
    plies. Distinct on purpose so a test can tell which one actually
    reached the guardrail, instead of both branches trivially matching."""
    base = _FixedLogitsChessformer(**CONFIG, fixed_logits=base_logits, override_logits=override_logits)
    return PersonalizedChessformer(base, nominal_elo=nominal_elo, identity="test-player"), base


def test_method1_guarded_uses_wrappers_own_policy_not_generic():
    generic_logits = _ranked_logits_favoring(["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"])
    wrapper_own_logits = _ranked_logits_favoring(["b2b3", "c2c4", "g1f3", "d2d4", "e2e4"])  # reversed
    wrapper, _ = _build_wrapper_with_split_logits(generic_logits, wrapper_own_logits)
    pool = FakeEnginePool()  # all cheap losses 0 -> guardrail is pass-through, ranking alone decides
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = guarded_generation.play_games_method1_guarded(
        wrapper, chess.WHITE, [1600.0] * 30, list(range(30)), config, pool,
    )
    for g in games:
        replay_and_check_legal(g["moves"])
    first_moves = [g["moves"][0] for g in games]
    # wrapper's OWN top-1 (b2b3) must be the dominant White first move across many seeds --
    # if the code had used generic_logits instead, e2e4 would dominate instead.
    from collections import Counter
    counts = Counter(first_moves)
    assert counts.most_common(1)[0][0] == "b2b3"


def test_method1_guarded_opponent_side_uses_generic_policy_not_wrapper():
    """The non-personalized (opponent) side in AG/GB must use the GENERIC
    policy, guarded identically -- not the wrapper's personalized policy,
    and not the old unguarded full-vocabulary sampler."""
    generic_logits = _ranked_logits_favoring(["b2b3", "c2c4", "g1f3", "d2d4", "e2e4"])
    wrapper_own_logits = _ranked_logits_favoring(["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"])
    wrapper, _ = _build_wrapper_with_split_logits(generic_logits, wrapper_own_logits)
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    # wrapper plays BLACK, so White's first move (ply 0) is the generic/opponent side
    games = guarded_generation.play_games_method1_guarded(
        wrapper, chess.BLACK, [1600.0] * 30, list(range(30)), config, pool,
    )
    for g in games:
        replay_and_check_legal(g["moves"])
    from collections import Counter
    white_first_moves = Counter(g["moves"][0] for g in games)
    # generic_logits favors b2b3 most -- if the wrapper's own policy had leaked into the
    # opponent's move instead, e2e4 would dominate here.
    assert white_first_moves.most_common(1)[0][0] == "b2b3"


def test_method1_ab_guarded_routes_each_side_to_its_own_wrapper():
    logits_a = _ranked_logits_favoring(["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"])
    logits_b = _ranked_logits_favoring(["b2b3", "c2c4", "g1f3", "d2d4", "e2e4"])
    wrapper_a, _ = _build_wrapper_with_split_logits(logits_a, logits_a, nominal_elo=1600.0)
    wrapper_b, _ = _build_wrapper_with_split_logits(logits_b, logits_b, nominal_elo=1900.0)
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = guarded_generation.play_games_method1_ab_guarded(
        wrapper_a, wrapper_b, chess.WHITE, list(range(10)), config, pool,
    )
    for g in games:
        replay_and_check_legal(g["moves"])
        assert g["white_elo"] == 1600.0 and g["black_elo"] == 1900.0


# ---------------------------------------------------------------------------
# Method 2: candidate construction + rerank preserved, invalid slots filtered
# ---------------------------------------------------------------------------

def build_real_style_stack(num_players=2, style_dim=STYLE_DIM):
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=num_players, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    return base, cnn, table, residual


def test_method2_guarded_produces_legal_games_both_conditions():
    base, cnn, table, residual = build_real_style_stack()
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = guarded_generation.play_games_method2_guarded(
        base, cnn, table, residual, player_id=0, player_color=chess.WHITE,
        player_elo=1600.0, opponent_elos=[1600.0] * 5, k=5, seeds=list(range(5)),
        config=config, engine_pool=pool,
    )
    for g in games:
        replay_and_check_legal(g["moves"])

    games_ab = guarded_generation.play_games_method2_ab_guarded(
        base, cnn, table, residual, player_id_a=0, player_id_b=1, a_color=chess.WHITE,
        elo_a=1600.0, elo_b=1900.0, k=5, seeds=list(range(5)), config=config, engine_pool=pool,
    )
    for g in games_ab:
        replay_and_check_legal(g["moves"])


def test_method2_guarded_never_offers_invalid_padding_candidate_to_engine():
    """Method 2's raw top-K can contain invalid padding slots when fewer
    than K legal moves exist (a real bug an earlier diagnostic script hit
    by forgetting this filter -- see guarded_generation._method2_candidates'
    docstring). Drive a near-stalemate-ish endgame position with very few
    legal moves and confirm every task handed to the engine pool only
    contains genuinely legal UCIs."""
    base, cnn, table, residual = build_real_style_stack()
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    # K(=5) legal moves is already rare for a starting position with a real
    # (if tiny) model; instead, directly exercise _method2_candidates with a
    # position that has fewer than 5 legal moves (verified: exactly 3, and
    # a genuinely valid/reachable position, unlike an adjacent-kings FEN).
    board = chess.Board(fen="7k/8/8/8/8/8/8/K7 w - - 0 1")  # bare king in the corner, 3 legal moves
    from amadeus_counterpoint.encoding import encode_history, legal_move_mask
    x = encode_history([board.copy(stack=False)]).unsqueeze(0)
    mask = legal_move_mask(board).unsqueeze(0)
    mover_elo_t = torch.tensor([1600.0])
    opp_elo_t = torch.tensor([1600.0])
    player_id_t = torch.tensor([0])

    candidates = guarded_generation._method2_candidates(
        base, cnn, table, residual, x, mover_elo_t, opp_elo_t, player_id_t, mask, 5,
    )
    (indices, p_behavior), = candidates
    assert len(indices) <= 3
    for idx in indices:
        from amadeus_counterpoint.encoding import policy_index_to_move
        move = policy_index_to_move(idx, board)
        assert move in board.legal_moves


# ---------------------------------------------------------------------------
# mover POV / mate-score correctness (unit-level, FakeEngine, no subprocess)
# ---------------------------------------------------------------------------

class _FakeChessEngine:
    def __init__(self, scores_by_fen):
        self.scores_by_fen = scores_by_fen

    def analyse(self, board, limit):
        import chess.engine as ce
        key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        return {"score": ce.PovScore(ce.Cp(self.scores_by_fen[key]), chess.WHITE)}


def test_pool_worker_eval_mover_pov_via_real_worker_function(monkeypatch):
    """Exercises guarded_generation._pool_worker_eval directly (bypassing
    multiprocessing.Pool) with a fake engine injected into the module-level
    _ENGINE global -- confirms the worker function correctly reuses
    strength_guardrail.candidate_cheap_losses (mover-POV-fixed) rather than
    re-deriving orientation itself."""
    board = create_board()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    board_e4 = board.copy(); board_e4.push(e4)
    board_d4 = board.copy(); board_d4.push(d4)

    fake_engine = _FakeChessEngine({
        board_e4.board_fen() + " b": 50,
        board_d4.board_fen() + " b": 10,
    })
    monkeypatch.setattr(guarded_generation, "_ENGINE", fake_engine)

    losses = guarded_generation._pool_worker_eval((board.fen(), ["e2e4", "d2d4"], 8))
    assert losses[0] == pytest.approx(0.0)
    assert losses[1] == pytest.approx(40.0)


def test_pool_worker_eval_handles_black_to_move_orientation(monkeypatch):
    board = create_board()
    board.push(chess.Move.from_uci("e2e4"))  # Black to move now
    e5 = chess.Move.from_uci("e7e5")
    c5 = chess.Move.from_uci("c7c5")
    board_e5 = board.copy(); board_e5.push(e5)
    board_c5 = board.copy(); board_c5.push(c5)

    # White-POV: after ...e5, White is +30 (worse for Black); after ...c5, White is +5 (better for Black)
    fake_engine = _FakeChessEngine({
        board_e5.board_fen() + " w": 30,
        board_c5.board_fen() + " w": 5,
    })
    monkeypatch.setattr(guarded_generation, "_ENGINE", fake_engine)

    losses = guarded_generation._pool_worker_eval((board.fen(), ["e7e5", "c7c5"], 8))
    # mover is Black: best (lowest White-eval) is c5 (+5) -> zero loss; e5 is 25cp worse for Black
    assert losses[1] == pytest.approx(0.0)
    assert losses[0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# experiment.py dispatch: GG/AG/GB/AB route to the guarded functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
def test_experiment_cell_batched_dispatches_to_guardrail_when_provided(condition):
    base = Chessformer(**CONFIG)
    wrapper_a = PersonalizedChessformer(base, nominal_elo=1600.0, identity="Player Zero")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=1900.0, identity="Player One")
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, condition, 1600.0, 1900.0,
        DYAD, n_games=4, root_seed=ROOT_SEED, checkpoint_identity="test",
        guardrail=config, engine_pool=pool,
    )
    assert len(games) == 4
    for g in games:
        replay_and_check_legal(g["moves"])
    assert len(pool.calls) > 0, "guardrail was never invoked through experiment.py's dispatch"


def test_experiment_cell_batched_none_guardrail_is_unchanged_old_behavior():
    """guardrail=None (the default) must still call the exact old unguarded
    functions -- a regression here would mean the additive parameter isn't
    actually additive."""
    base = Chessformer(**CONFIG)
    wrapper_a = PersonalizedChessformer(base, nominal_elo=1600.0, identity="Player Zero")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=1900.0, identity="Player One")

    games_default = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, "AB", 1600.0, 1900.0,
        DYAD, n_games=3, root_seed=ROOT_SEED, checkpoint_identity="test",
    )
    games_explicit_none = generate_method1_cell_batched(
        wrapper_a, wrapper_b, base, chess.WHITE, "AB", 1600.0, 1900.0,
        DYAD, n_games=3, root_seed=ROOT_SEED, checkpoint_identity="test",
        guardrail=None, engine_pool=None,
    )
    assert [g["moves"] for g in games_default] == [g["moves"] for g in games_explicit_none]


def test_guardrail_and_engine_pool_must_be_supplied_together():
    base = Chessformer(**CONFIG)
    wrapper_a = PersonalizedChessformer(base, nominal_elo=1600.0, identity="Player Zero")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=1900.0, identity="Player One")
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    with pytest.raises(ValueError, match="together"):
        generate_method1_cell_batched(
            wrapper_a, wrapper_b, base, chess.WHITE, "GG", 1600.0, 1900.0,
            DYAD, n_games=1, root_seed=ROOT_SEED, checkpoint_identity="test",
            guardrail=config, engine_pool=None,
        )


@pytest.mark.parametrize("condition", ["GG", "AG", "GB", "AB"])
def test_experiment_method2_cell_batched_dispatches_to_guardrail(condition):
    base, cnn, table, residual = build_real_style_stack()
    pool = FakeEnginePool()
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    games = generate_method2_cell_batched(
        base, cnn, table, residual, player_id_a=0, player_id_b=1, a_color=chess.WHITE,
        condition=condition, elo_a=1600.0, elo_b=1900.0, k=5, dyad=DYAD,
        n_games=4, root_seed=ROOT_SEED, checkpoint_identity="test",
        guardrail=config, engine_pool=pool,
    )
    assert len(games) == 4
    for g in games:
        replay_and_check_legal(g["moves"])
    assert len(pool.calls) > 0
