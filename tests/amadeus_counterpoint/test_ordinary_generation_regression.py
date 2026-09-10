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
