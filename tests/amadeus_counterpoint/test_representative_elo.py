from amadeus_counterpoint.evaluation.representative_elo import compute_representative_elos


def test_median_uses_one_rating_observation_per_game_not_per_move():
    records = [
        # Own ratings: 2700, 2800, 2900 -> median 2800, regardless of how
        # many moves each game record happens to carry.
        {"player_id": 0, "mover_color": "white", "white_elo": 2700, "black_elo": 2600, "moves": ["e2e4"] * 40},
        {"player_id": 0, "mover_color": "white", "white_elo": 2800, "black_elo": 2600, "moves": ["e2e4"]},
        {"player_id": 0, "mover_color": "black", "white_elo": 2600, "black_elo": 2900, "moves": ["e2e4"]},
    ]

    assert compute_representative_elos(records) == {0: 2800}


def test_uses_the_targets_own_rating_not_the_opponents():
    records = [
        {"player_id": 0, "mover_color": "white", "white_elo": 2700, "black_elo": 1500},
        {"player_id": 0, "mover_color": "black", "white_elo": 1500, "black_elo": 2900},
    ]

    # Median of the target's own ratings (2700, 2900) = 2800, not the
    # opponent's (1500).
    assert compute_representative_elos(records) == {0: 2800.0}


def test_separate_players_get_independent_medians():
    records = [
        {"player_id": 0, "mover_color": "white", "white_elo": 2700, "black_elo": 2000},
        {"player_id": 0, "mover_color": "white", "white_elo": 2900, "black_elo": 2000},
        {"player_id": 1, "mover_color": "black", "white_elo": 2000, "black_elo": 2500},
    ]

    assert compute_representative_elos(records) == {0: 2800.0, 1: 2500}
