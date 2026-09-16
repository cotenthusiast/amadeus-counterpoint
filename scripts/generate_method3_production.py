"""Production exploratory Method-3 hybrid generation.

M3 composes the existing Method-1 personalized base policy with the existing
Method-2 style/reranking and frozen sampling pipeline. It loads checkpoints
only for inference; there is deliberately no training or optimizer path.

The output is a new ``method3_hybrid`` tree. Existing Method-1/Method-2
artifacts are never read for resumability and are never overwritten.
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
    load_method2_style_stack_for_generation,
)
from amadeus_counterpoint.evaluation.generation.experiment import (
    CONDITIONS,
    ORIENTATIONS,
    generate_method3_cell_batched,
)
from amadeus_counterpoint.evaluation.generation.guarded_generation import (
    StockfishEnginePool,
)
from amadeus_counterpoint.evaluation.generation.production import (
    cell_is_complete,
    write_cell,
)
from amadeus_counterpoint.evaluation.generation.strength_guardrail import (
    FROZEN_PRODUCTION_CONFIG,
)
from amadeus_counterpoint.models.chessformer import Chessformer

# Must match the architecture used by the existing Method-2 production
# runner and the Broadcast-adapted base checkpoint.
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
M3_METHOD_IDENTIFIER = "method3_hybrid"


def build_base(checkpoint_path, device) -> Chessformer:
    base = Chessformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
        d1=D1, d2=D2, d3=D3, d_ff=D_FF, head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM, elo_dim=ELO_DIM,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    base.load_state_dict(payload["model"])
    return base.to(device)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--player-checkpoints", type=str, required=True,
        help='JSON string: {"<player_id>": "<Method-1 checkpoint path>", ...}',
    )
    parser.add_argument(
        "--identities", type=str, required=True,
        help='JSON string: {"<player_id>": "<checkpoint identity>", ...}',
    )
    parser.add_argument("--method2-checkpoint", type=Path, required=True)
    parser.add_argument("--representative-elos", type=Path, required=True)
    parser.add_argument("--player-ids", type=str, required=True)
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--games-per-orientation", type=int, default=PRODUCTION_GAMES_PER_ORIENTATION)
    parser.add_argument(
        "--generation-batch-size", type=int, default=PRODUCTION_GENERATION_BATCH_SIZE,
        help="Maximum games active on the GPU per cell; does not change seeds or metadata.",
    )
    parser.add_argument("--checkpoint-identity", type=str, required=True)
    parser.add_argument("--protocol-version", type=str, required=True)
    parser.add_argument("--code-commit", type=str, required=True)
    parser.add_argument("--temperature", type=str, default="1.0")
    parser.add_argument("--only-dyad-a", type=int, default=None)
    parser.add_argument("--only-dyad-b", type=int, default=None)
    parser.add_argument("--only-orientation", type=str, choices=ORIENTATIONS, default=None)
    parser.add_argument("--only-condition", type=str, choices=CONDITIONS, default=None)
    parser.add_argument(
        "--use-strength-guardrail", action="store_true",
        help="Use the same frozen Stockfish/strength-aware sampler as guarded M2 production.",
    )
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--gcc-lib-path", type=str, default=None)
    parser.add_argument("--n-engine-workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.use_strength_guardrail and not args.stockfish_path:
        raise SystemExit("--stockfish-path is required when --use-strength-guardrail is set")
    if (args.only_dyad_a is None) != (args.only_dyad_b is None):
        raise SystemExit("--only-dyad-a and --only-dyad-b must be given together")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player_checkpoints = json.loads(args.player_checkpoints)
    identities = json.loads(args.identities)
    player_ids = sorted(int(player_id) for player_id in args.player_ids.split(","))
    representative_elos = {
        int(player_id): elo
        for player_id, elo in json.loads(args.representative_elos.read_text()).items()
    }

    base = build_base(args.base_checkpoint, device)
    wrappers = {}
    for player_id in player_ids:
        player_id_str = str(player_id)
        if player_id_str not in player_checkpoints or player_id_str not in identities:
            raise SystemExit(f"missing Method-1 checkpoint and identity for player {player_id}")
        wrapper = load_method1_wrapper_for_generation(
            base, player_checkpoints[player_id_str],
            nominal_elo=representative_elos[player_id], identity=identities[player_id_str],
        )
        wrapper.requires_grad_(False)
        wrappers[player_id] = wrapper

    cnn, table, residual, m2_metadata = load_method2_style_stack_for_generation(
        base, args.method2_checkpoint, num_players=args.num_players, style_dim=args.style_dim,
    )
    if m2_metadata["k"] is not None and m2_metadata["k"] != args.k:
        raise SystemExit(f"M2 checkpoint K={m2_metadata['k']} does not match requested K={args.k}")
    cnn.requires_grad_(False)
    table.requires_grad_(False)
    residual.requires_grad_(False)

    metadata = {
        "protocol_version": args.protocol_version,
        "code_commit": args.code_commit,
        "root_seed": str(args.root_seed),
        "temperature": args.temperature,
        "strength_guardrail": "on" if args.use_strength_guardrail else "off",
        "method_identifier": M3_METHOD_IDENTIFIER,
        "seed_namespace": "method3_hybrid__<dyad>",
        "composition": "M1-personalized-base-before-M2-legal-topK-style-rerank",
        "base_checkpoint": str(args.base_checkpoint),
        "m1_player_checkpoints": json.dumps(player_checkpoints, sort_keys=True),
        "m1_identities": json.dumps(identities, sort_keys=True),
        "m2_checkpoint": str(args.method2_checkpoint),
        "m2_k": str(args.k),
        "m2_style_dim": str(args.style_dim),
        "m2_checkpoint_k": str(m2_metadata["k"]),
        "guardrail_config": repr(FROZEN_PRODUCTION_CONFIG) if args.use_strength_guardrail else "off",
    }

    guardrail = FROZEN_PRODUCTION_CONFIG if args.use_strength_guardrail else None
    engine_pool = None
    if args.use_strength_guardrail:
        engine_pool = StockfishEnginePool(
            n_workers=args.n_engine_workers, stockfish_path=args.stockfish_path,
            gcc_lib_path=args.gcc_lib_path, threads=guardrail.threads, hash_mb=guardrail.hash_mb,
        )
        print(f"strength guardrail ON: {guardrail}, {args.n_engine_workers} engine workers")

    try:
        _run_cells(args, player_ids, wrappers, base, cnn, table, residual,
                   representative_elos, metadata, guardrail, engine_pool)
    finally:
        if engine_pool is not None:
            engine_pool.close()
            print("engine pool closed")


def _run_cells(args, player_ids, wrappers, base, cnn, table, residual,
               representative_elos, metadata, guardrail, engine_pool):
    only_dyad = {args.only_dyad_a, args.only_dyad_b} if args.only_dyad_a is not None else None
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
                if cell_is_complete(args.output_root, M3_METHOD_IDENTIFIER, dyad, condition, orientation):
                    print(f"skip (already complete): {M3_METHOD_IDENTIFIER} {dyad} {condition} {orientation}")
                    continue

                games = generate_method3_cell_batched(
                    wrappers[player_id_a], wrappers[player_id_b], base, cnn, table, residual,
                    player_id_a, player_id_b, a_color, condition, elo_a, elo_b, args.k, dyad,
                    args.games_per_orientation, args.root_seed, args.checkpoint_identity,
                    chunk_size=args.generation_batch_size, guardrail=guardrail, engine_pool=engine_pool,
                )
                path = write_cell(
                    args.output_root, M3_METHOD_IDENTIFIER, dyad, condition, orientation, games, metadata,
                )
                print(f"wrote {path} ({len(games)} games)")


if __name__ == "__main__":
    main()
