"""Stage-4 regression coverage for the ORDINARY (non-personalized) generation
path in single.py / batch.py.

Neither file is modified by any personalization stage; these tests run the
real end-to-end mask/softmax/sample/decode path against a real (tiny) model
to prove it still behaves exactly as before the Method-1/Method-2 additions.
No pre-existing test file is modified here.
"""

from _helpers import CONFIG, replay_and_check_legal
from amadeus_counterpoint.evaluation.generation import batch, single
from amadeus_counterpoint.models.chessformer import Chessformer


def test_play_game_still_produces_only_legal_moves():
    model = Chessformer(**CONFIG)

    game = single.play_game(model, white_elo=1500.0, black_elo=1600.0, seed=1)

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_play_game_still_seeded_reproducible():
    model = Chessformer(**CONFIG)

    game_a = single.play_game(model, white_elo=1500.0, black_elo=1600.0, seed=5)
    game_b = single.play_game(model, white_elo=1500.0, black_elo=1600.0, seed=5)

    assert game_a == game_b


def test_play_games_batched_matches_single_game_for_the_same_seed():
    model = Chessformer(**CONFIG)

    single_game = single.play_game(model, white_elo=1500.0, black_elo=1600.0, seed=11)
    batched = batch.play_games(model, white_elos=[1500.0], black_elos=[1600.0], seeds=[11])

    assert batched[0] == single_game


def test_play_games_batched_each_game_legal_and_independent_of_batch_position():
    model = Chessformer(**CONFIG)

    games = batch.play_games(
        model,
        white_elos=[1500.0, 1700.0],
        black_elos=[1600.0, 1800.0],
        seeds=[21, 22],
    )

    for game in games:
        assert len(game["moves"]) > 0
        replay_and_check_legal(game["moves"])

    # Re-running the second game alone (its own seed) must match its result
    # when generated as part of a two-game batch -- proving independence
    # from batch position/order, exactly as batch.play_games documents.
    alone = single.play_game(model, white_elo=1700.0, black_elo=1800.0, seed=22)
    assert games[1] == alone


# --- device handling ---------------------------------------------------------


def test_play_game_follows_model_device_not_cpu_default():
    """Regression test for a real bug found during Kelvin2 GPU generation
    benchmarking: play_game created x/Elo tensors and the legal mask with no
    device argument (implicit CPU default), which crashed against a
    CUDA-resident model. `meta` is a real, distinct device requiring no GPU:
    it reproduces the same "expected all tensors on the same device" failure
    CPU-vs-CUDA did. A full game can't complete on meta (sampling needs real
    probability values, which meta tensors don't have -- `.cpu()` on a meta
    tensor raises NotImplementedError, not the original RuntimeError), so
    this checks that play_game gets PAST the forward pass and masking (the
    parts the fix touches) and fails only at that expected, unrelated point --
    proving the device-mismatch bug itself is gone.
    """
    model = Chessformer(**CONFIG).to("meta")

    try:
        single.play_game(model, white_elo=1500.0, black_elo=1600.0, seed=1)
        raise AssertionError("expected NotImplementedError from meta tensor sampling")
    except NotImplementedError as e:
        assert "meta tensor" in str(e)
    except RuntimeError as e:
        raise AssertionError(
            f"device-mismatch regression: play_game raised the original bug's "
            f"error instead of getting past the forward pass: {e}"
        )


def test_play_games_batch_follows_model_device_not_cpu_default():
    """Same regression, for the batched player (evaluation.generation.batch)."""
    model = Chessformer(**CONFIG).to("meta")

    try:
        batch.play_games(model, white_elos=[1500.0], black_elos=[1600.0], seeds=[1])
        raise AssertionError("expected NotImplementedError from meta tensor sampling")
    except NotImplementedError as e:
        assert "meta tensor" in str(e)
    except RuntimeError as e:
        raise AssertionError(
            f"device-mismatch regression: play_games raised the original bug's "
            f"error instead of getting past the forward pass: {e}"
        )
