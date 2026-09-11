"""Production Method-1 synthetic generation: one shard per
(dyad, condition, orientation) cell, skipping cells that already have a
completed artifact.

Loads all 8 players' Method-1 checkpoints once, then generates the full
28-dyad x 2-orientation x 4-condition grid at --games-per-orientation games
per cell (production default: 5000). This is a LONG-RUNNING production
script -- it is not run as part of this implementation session.

    python scripts/generate_method1_production.py \\
        --base-checkpoint checkpoints/broadcast_adaptation/step_00200000.pt \\
        --player-checkpoints '{"0": "checkpoints/method1/player_0.pt", "1": "checkpoints/method1/player_1.pt"}' \\
        --identities '{"0": "Magnus Carlsen", "1": "Wesley So"}' \\
        --representative-elos configs/representative_elos.json \\
        --output-root artifacts/synthetic \\
        --root-seed 20260910 \\
        --checkpoint-identity method1-2026-09-10 \\
        --protocol-version 1 --code-commit <git-sha>

Optional cell-level filtering, for independent SLURM array jobs over the
28-dyad x 2-orientation x 4-condition grid: --only-dyad-a/--only-dyad-b
(both, restricts to exactly that unordered pair), --only-orientation
(A_WHITE or B_WHITE), --only-condition (GG/AG/GB/AB). Any combination may be
supplied; omitted filters process every value as before. Filtering only
skips which (dyad, condition, orientation) iterations run -- the seed/
metadata/output path for any cell that DOES run are computed exactly the
same way as in the unfiltered grid, so one filtered invocation for a cell
produces byte-identical output to that cell running inside the full loop.
"""

import argparse
import itertools
import json
from pathlib import Path

import chess
import torch

from amadeus_counterpoint.data.sealed_dyads import A_WHITE, dyad_key
from amadeus_counterpoint.evaluation.generation.checkpoint_loading import (
    load_method1_wrapper_for_generation,
)
from amadeus_counterpoint.evaluation.generation.experiment import CONDITIONS, ORIENTATIONS, generate_method1_cell
from amadeus_counterpoint.evaluation.generation.production import cell_is_complete, write_cell
from amadeus_counterpoint.models.chessformer import Chessformer

# Must match the architecture the Broadcast-adapted base checkpoint was
# trained with (scripts/train.py's "Chessformer 79M" configuration).
D_MODEL = 1024
NUM_HEADS = 32
NUM_LAYERS = 8
D1 = 32
D2 = 128
D3 = 128
D_FF = 2048
ELO_DIM = 128
HEAD_HID_DIM = 1024
RAW_INPUT_DIM = 96
DROPOUT = 0.0

PRODUCTION_GAMES_PER_ORIENTATION = 5000


def build_base(checkpoint_path, device) -> Chessformer:
    base = Chessformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
        d1=D1, d2=D2, d3=D3, d_ff=D_FF, head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM, elo_dim=ELO_DIM,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    base.load_state_dict(payload["model"])
    return base.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--player-checkpoints", type=str, required=True,
        help='JSON string: {"<player_id>": "<method1 checkpoint path>", ...}',
    )
    parser.add_argument(
        "--identities", type=str, required=True,
        help='JSON string: {"<player_id>": "<identity string>", ...}, must match each checkpoint',
    )
    parser.add_argument("--representative-elos", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--games-per-orientation", type=int, default=PRODUCTION_GAMES_PER_ORIENTATION)
    parser.add_argument("--checkpoint-identity", type=str, required=True)
    parser.add_argument("--protocol-version", type=str, required=True)
    parser.add_argument("--code-commit", type=str, required=True)
    parser.add_argument("--temperature", type=str, default="1.0")
    parser.add_argument(
        "--only-dyad-a", type=int, default=None,
        help="Restrict to the one dyad containing this player_id and --only-dyad-b "
        "(both required together). Default: process every dyad.",
    )
    parser.add_argument("--only-dyad-b", type=int, default=None)
    parser.add_argument(
        "--only-orientation", type=str, default=None, choices=ORIENTATIONS,
        help="Restrict to one orientation. Default: process both.",
    )
    parser.add_argument(
        "--only-condition", type=str, default=None, choices=CONDITIONS,
        help="Restrict to one condition. Default: process all four.",
    )
    args = parser.parse_args()

    if (args.only_dyad_a is None) != (args.only_dyad_b is None):
        raise SystemExit("--only-dyad-a and --only-dyad-b must be given together")
    only_dyad = (
        {args.only_dyad_a, args.only_dyad_b} if args.only_dyad_a is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    player_checkpoints = json.loads(args.player_checkpoints)
    identities = json.loads(args.identities)
    representative_elos = {
        int(player_id): elo
        for player_id, elo in json.loads(args.representative_elos.read_text()).items()
    }

    base = build_base(args.base_checkpoint, device)

    wrappers = {}
    for player_id_str, checkpoint_path in player_checkpoints.items():
        player_id = int(player_id_str)
        wrappers[player_id] = load_method1_wrapper_for_generation(
            base, checkpoint_path,
            nominal_elo=representative_elos[player_id],
            identity=identities[player_id_str],
        )

    player_ids = sorted(wrappers)
    metadata = {
        "protocol_version": args.protocol_version,
        "code_commit": args.code_commit,
        "root_seed": str(args.root_seed),
        "temperature": args.temperature,
    }

    for player_id_a, player_id_b in itertools.combinations(player_ids, 2):
        if only_dyad is not None and {player_id_a, player_id_b} != only_dyad:
            continue

        dyad = dyad_key(player_id_a, player_id_b)
        wrapper_a, wrapper_b = wrappers[player_id_a], wrappers[player_id_b]
        elo_a, elo_b = representative_elos[player_id_a], representative_elos[player_id_b]

        for orientation in ORIENTATIONS:
            if args.only_orientation is not None and orientation != args.only_orientation:
                continue

            a_color = chess.WHITE if orientation == A_WHITE else chess.BLACK

            for condition in CONDITIONS:
                if args.only_condition is not None and condition != args.only_condition:
                    continue

                if cell_is_complete(args.output_root, "method1", dyad, condition, orientation):
                    print(f"skip (already complete): method1 {dyad} {condition} {orientation}")
                    continue

                games = generate_method1_cell(
                    wrapper_a, wrapper_b, base, a_color, condition,
                    elo_a, elo_b, dyad, args.games_per_orientation,
                    args.root_seed, args.checkpoint_identity,
                )
                path = write_cell(
                    args.output_root, "method1", dyad, condition, orientation, games, metadata
                )
                print(f"wrote {path} ({len(games)} games)")


if __name__ == "__main__":
    main()
