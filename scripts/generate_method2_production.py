"""Production Method-2 synthetic generation: one shard per
(dyad, condition, orientation) cell, skipping cells that already have a
completed artifact.

Loads the one shared Method-2 style stack once, then generates the full
28-dyad x 2-orientation x 4-condition grid at --games-per-orientation games
per cell (production default: 5000). This is a LONG-RUNNING production
script -- it is not run as part of this implementation session.

    python scripts/generate_method2_production.py \\
        --base-checkpoint checkpoints/broadcast_adaptation/step_00200000.pt \\
        --method2-checkpoint checkpoints/method2/joint.pt \\
        --representative-elos configs/representative_elos.json \\
        --player-ids 0,1,2,3,4,5,6,7 \\
        --output-root artifacts/synthetic \\
        --root-seed 20260910 \\
        --checkpoint-identity method2-2026-09-10 \\
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

Games are generated through the batched path (generate_method2_cell_batched),
internally chunked to --generation-batch-size games active on the GPU at
once (default 128). Chunking only changes how many games one batched model
call advances at a time -- the full per-cell seed list is still derived up
front from the global game_index exactly as an unchunked run would, so a
cell's output (games, seeds, metadata) does not depend on the chosen
--generation-batch-size.
"""

import argparse
import itertools
import json
from pathlib import Path

import chess
import torch

from amadeus_counterpoint.data.sealed_dyads import A_WHITE, dyad_key
from amadeus_counterpoint.evaluation.generation.checkpoint_loading import (
    load_method2_style_stack_for_generation,
)
from amadeus_counterpoint.evaluation.generation.experiment import (
    CONDITIONS,
    ORIENTATIONS,
    generate_method2_cell_batched,
)
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
PRODUCTION_GENERATION_BATCH_SIZE = 128


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
    parser.add_argument("--method2-checkpoint", type=Path, required=True)
    parser.add_argument("--representative-elos", type=Path, required=True)
    parser.add_argument("--player-ids", type=str, required=True, help="comma-separated, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--games-per-orientation", type=int, default=PRODUCTION_GAMES_PER_ORIENTATION)
    parser.add_argument(
        "--generation-batch-size", type=int, default=PRODUCTION_GENERATION_BATCH_SIZE,
        help="Max games active on the GPU at once per cell (internal chunking of "
        "--games-per-orientation; does not change seeds/metadata/output). Default: 128.",
    )
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

    player_ids = sorted(int(p) for p in args.player_ids.split(","))
    representative_elos = {
        int(player_id): elo
        for player_id, elo in json.loads(args.representative_elos.read_text()).items()
    }

    base = build_base(args.base_checkpoint, device)
    cnn, table, residual, _ = load_method2_style_stack_for_generation(
        base, args.method2_checkpoint, num_players=args.num_players, style_dim=args.style_dim,
    )

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
        elo_a, elo_b = representative_elos[player_id_a], representative_elos[player_id_b]

        for orientation in ORIENTATIONS:
            if args.only_orientation is not None and orientation != args.only_orientation:
                continue

            a_color = chess.WHITE if orientation == A_WHITE else chess.BLACK

            for condition in CONDITIONS:
                if args.only_condition is not None and condition != args.only_condition:
                    continue

                if cell_is_complete(args.output_root, "method2", dyad, condition, orientation):
                    print(f"skip (already complete): method2 {dyad} {condition} {orientation}")
                    continue

                games = generate_method2_cell_batched(
                    base, cnn, table, residual, player_id_a, player_id_b,
                    a_color, condition, elo_a, elo_b, args.k, dyad,
                    args.games_per_orientation, args.root_seed, args.checkpoint_identity,
                    chunk_size=args.generation_batch_size,
                )
                path = write_cell(
                    args.output_root, "method2", dyad, condition, orientation, games, metadata
                )
                print(f"wrote {path} ({len(games)} games)")


if __name__ == "__main__":
    main()
