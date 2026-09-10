"""Synthetic GG/AG/GB/AB game generation across the 28 sealed dyads.

Experiment glue, not a framework: for each dyad, each color orientation, and
each of the four conditions, call the existing generation function and
record the result. This module never generates a move itself -- it only
decides WHICH existing generator to call and WHAT seed to give it, then
shapes the result into `evaluation.artifacts.GAME_SCHEMA`'s field set.

Conditions:
    GG = generic A, generic B       (ordinary base generation, no personalization)
    AG = personalized A, generic B
    GB = generic A, personalized B
    AB = personalized A, personalized B

A/B identity is fixed per dyad (A = the lower player_id, B = the higher,
matching `data.sealed_dyads.dyad_key`) and never changes with color; only
`a_color` changes between the two orientations. Every condition still
conditions the model on each player's OWN representative Elo (per the
frozen protocol) -- "generic" means "no personalized representation", not
"no Elo".

Two separate entry points, one per method, because Method 1 and Method 2
have genuinely different personalization objects (a `PersonalizedChessformer`
per player vs. one shared style stack). A production run of ONE method
against `n_games_per_orientation=5000` produces
`28 dyads * 2 orientations * 4 conditions * 5000 = 1,120,000` games for THAT
method alone; calling both entry points does not share or halve that count.
"""

import itertools

import chess

from amadeus_counterpoint.data.sealed_dyads import A_WHITE, B_WHITE, dyad_key
from amadeus_counterpoint.evaluation.generation import single
from amadeus_counterpoint.evaluation.generation.method1_personalized import (
    play_game_method1,
    play_game_method1_ab,
)
from amadeus_counterpoint.evaluation.generation.method2_personalized import (
    play_game_method2,
    play_game_method2_ab,
)
from amadeus_counterpoint.evaluation.seeding import derive_seed
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN

CONDITIONS = ("GG", "AG", "GB", "AB")
ORIENTATIONS = (A_WHITE, B_WHITE)

GENERIC_REPRESENTATION_IDENTITY = "global"


def _game_record(
    game, game_index, seed, dyad, condition, orientation,
    checkpoint_identity, white_representation_identity, black_representation_identity,
) -> dict:
    """Shape one generator's output dict into `artifacts.GAME_SCHEMA`'s fields."""
    return {
        "game_index": game_index,
        "seed": seed,
        "dyad": dyad,
        "condition": condition,
        "orientation": orientation,
        "result": game["result"],
        "censored": game["censored"],
        "moves": game["moves"],
        "white_elo": game["white_elo"],
        "black_elo": game["black_elo"],
        "checkpoint_identity": checkpoint_identity,
        "white_representation_identity": white_representation_identity,
        "black_representation_identity": black_representation_identity,
    }


def generate_method1_cell(
    wrapper_a: PersonalizedChessformer,
    wrapper_b: PersonalizedChessformer,
    base: Chessformer,
    a_color: chess.Color,
    condition: str,
    elo_a: float,
    elo_b: float,
    dyad: str,
    n_games: int,
    root_seed: int,
    checkpoint_identity: str,
) -> list[dict]:
    """Generate one (dyad, condition, orientation) cell's games, Method 1.

    `a_color` fixes the orientation (`chess.WHITE` = A_WHITE, `chess.BLACK` =
    B_WHITE). Pulled out of `generate_method1_experiment` so a production
    runner can generate/write/skip one cell at a time without regenerating
    the whole 28-dyad grid.
    """
    orientation = A_WHITE if a_color == chess.WHITE else B_WHITE
    b_color = chess.BLACK if a_color == chess.WHITE else chess.WHITE

    games = []
    for game_index in range(n_games):
        seed = derive_seed(
            root_seed=root_seed,
            dyad=f"method1__{dyad}",
            condition=condition,
            orientation=orientation,
            game_index=game_index,
        )

        if condition == "GG":
            white_elo = elo_a if a_color == chess.WHITE else elo_b
            black_elo = elo_b if a_color == chess.WHITE else elo_a
            game = single.play_game(base, white_elo, black_elo, seed)
            a_repr, b_repr = GENERIC_REPRESENTATION_IDENTITY, GENERIC_REPRESENTATION_IDENTITY
        elif condition == "AG":
            game = play_game_method1(wrapper_a, a_color, opponent_elo=elo_b, seed=seed)
            a_repr, b_repr = wrapper_a.identity, GENERIC_REPRESENTATION_IDENTITY
        elif condition == "GB":
            game = play_game_method1(wrapper_b, b_color, opponent_elo=elo_a, seed=seed)
            a_repr, b_repr = GENERIC_REPRESENTATION_IDENTITY, wrapper_b.identity
        else:  # AB
            game = play_game_method1_ab(wrapper_a, wrapper_b, a_color, seed)
            a_repr, b_repr = wrapper_a.identity, wrapper_b.identity

        white_repr, black_repr = (
            (a_repr, b_repr) if a_color == chess.WHITE else (b_repr, a_repr)
        )

        games.append(_game_record(
            game, game_index, seed, dyad, condition, orientation,
            checkpoint_identity, white_repr, black_repr,
        ))

    return games


def generate_method1_experiment(
    wrappers: dict[int, PersonalizedChessformer],
    base: Chessformer,
    player_ids: list[int],
    representative_elos: dict[int, float],
    n_games_per_orientation: int,
    root_seed: int,
    checkpoint_identity: str,
) -> list[dict]:
    """Generate GG/AG/GB/AB games for every unordered dyad among `player_ids`, Method 1.

    `wrappers[player_id]` must be that player's already-loaded
    `PersonalizedChessformer` (see `checkpoint_loading.
    load_method1_wrapper_for_generation`); `base` is the shared frozen
    Broadcast-adapted Chessformer used for the GG (fully generic) games.
    `representative_elos[player_id]` is that player's one fixed Elo-
    conditioning scalar, used for every condition and orientation.

    Returns a flat list of dicts, each shaped exactly like
    `evaluation.artifacts.GAME_SCHEMA`'s fields.
    """
    games = []

    for player_id_a, player_id_b in itertools.combinations(sorted(player_ids), 2):
        dyad = dyad_key(player_id_a, player_id_b)
        wrapper_a = wrappers[player_id_a]
        wrapper_b = wrappers[player_id_b]
        elo_a = representative_elos[player_id_a]
        elo_b = representative_elos[player_id_b]

        for orientation in ORIENTATIONS:
            a_color = chess.WHITE if orientation == A_WHITE else chess.BLACK

            for condition in CONDITIONS:
                games.extend(generate_method1_cell(
                    wrapper_a, wrapper_b, base, a_color, condition,
                    elo_a, elo_b, dyad, n_games_per_orientation,
                    root_seed, checkpoint_identity,
                ))

    return games


def generate_method2_cell(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    player_id_a: int,
    player_id_b: int,
    a_color: chess.Color,
    condition: str,
    elo_a: float,
    elo_b: float,
    k: int,
    dyad: str,
    n_games: int,
    root_seed: int,
    checkpoint_identity: str,
) -> list[dict]:
    """Generate one (dyad, condition, orientation) cell's games, Method 2.

    `a_color` fixes the orientation (`chess.WHITE` = A_WHITE, `chess.BLACK` =
    B_WHITE). Pulled out of `generate_method2_experiment` so a production
    runner can generate/write/skip one cell at a time without regenerating
    the whole 28-dyad grid.
    """
    orientation = A_WHITE if a_color == chess.WHITE else B_WHITE
    b_color = chess.BLACK if a_color == chess.WHITE else chess.WHITE

    games = []
    for game_index in range(n_games):
        seed = derive_seed(
            root_seed=root_seed,
            dyad=f"method2__{dyad}",
            condition=condition,
            orientation=orientation,
            game_index=game_index,
        )

        if condition == "GG":
            white_elo = elo_a if a_color == chess.WHITE else elo_b
            black_elo = elo_b if a_color == chess.WHITE else elo_a
            game = single.play_game(base, white_elo, black_elo, seed)
            a_repr, b_repr = GENERIC_REPRESENTATION_IDENTITY, GENERIC_REPRESENTATION_IDENTITY
        elif condition == "AG":
            game = play_game_method2(
                base, cnn, table, residual,
                player_id=player_id_a, player_color=a_color,
                player_elo=elo_a, opponent_elo=elo_b,
                k=k, seed=seed,
            )
            a_repr, b_repr = str(player_id_a), GENERIC_REPRESENTATION_IDENTITY
        elif condition == "GB":
            game = play_game_method2(
                base, cnn, table, residual,
                player_id=player_id_b, player_color=b_color,
                player_elo=elo_b, opponent_elo=elo_a,
                k=k, seed=seed,
            )
            a_repr, b_repr = GENERIC_REPRESENTATION_IDENTITY, str(player_id_b)
        else:  # AB
            game = play_game_method2_ab(
                base, cnn, table, residual,
                player_id_a=player_id_a, player_id_b=player_id_b,
                a_color=a_color, elo_a=elo_a, elo_b=elo_b,
                k=k, seed=seed,
            )
            a_repr, b_repr = str(player_id_a), str(player_id_b)

        white_repr, black_repr = (
            (a_repr, b_repr) if a_color == chess.WHITE else (b_repr, a_repr)
        )

        games.append(_game_record(
            game, game_index, seed, dyad, condition, orientation,
            checkpoint_identity, white_repr, black_repr,
        ))

    return games


def generate_method2_experiment(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    player_ids: list[int],
    representative_elos: dict[int, float],
    k: int,
    n_games_per_orientation: int,
    root_seed: int,
    checkpoint_identity: str,
) -> list[dict]:
    """Generate GG/AG/GB/AB games for every unordered dyad among `player_ids`, Method 2.

    `base`/`cnn`/`table`/`residual` are the one shared, already-loaded
    Method-2 style stack (see `checkpoint_loading.
    load_method2_style_stack_for_generation`) -- the same objects are used
    for every dyad; a player is selected only via their `player_id` row in
    `table`. `representative_elos[player_id]` is that player's one fixed
    Elo-conditioning scalar, used for every condition and orientation.

    Returns a flat list of dicts, each shaped exactly like
    `evaluation.artifacts.GAME_SCHEMA`'s fields.
    """
    games = []

    for player_id_a, player_id_b in itertools.combinations(sorted(player_ids), 2):
        dyad = dyad_key(player_id_a, player_id_b)
        elo_a = representative_elos[player_id_a]
        elo_b = representative_elos[player_id_b]

        for orientation in ORIENTATIONS:
            a_color = chess.WHITE if orientation == A_WHITE else chess.BLACK

            for condition in CONDITIONS:
                games.extend(generate_method2_cell(
                    base, cnn, table, residual, player_id_a, player_id_b,
                    a_color, condition, elo_a, elo_b, k, dyad,
                    n_games_per_orientation, root_seed, checkpoint_identity,
                ))

    return games
