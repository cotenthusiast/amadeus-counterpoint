"""Post-hoc cross-player identity diagnostic for Method 1 / Method 2.

Diagnostic only: does NOT retrain anything, does NOT modify any checkpoint,
does NOT touch sealed target-vs-target games (structurally impossible -- see
evaluation.identity_diagnostic's module docstring), and does NOT decide
anything about the experimental protocol. For each true player P, evaluates
P's existing non-sealed held-out validation positions under (1) the generic
frozen Broadcast base, (2) P's own learned representation, and (3) every
other player's representation, producing an 8x8 cross-player matrix per
method plus rank/delta summary statistics.

    python scripts/evaluate_personalization_identity.py \\
        --records-path data/style_records.json \\
        --base-checkpoint checkpoints/broadcast_adaptation/step_00200000.pt \\
        --method1-checkpoint-dir checkpoints/method1 \\
        --method2-checkpoint checkpoints/method2/joint.pt \\
        --targets-config configs/broadcast_targets.json \\
        --output-json identity_diagnostic.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.broadcast_ingest import load_game_records
from amadeus_counterpoint.data.broadcast_targets import load_broadcast_targets
from amadeus_counterpoint.evaluation.generation.checkpoint_loading import (
    load_method1_wrapper_for_generation,
    load_method2_style_stack_for_generation,
)
from amadeus_counterpoint.evaluation.identity_diagnostic import (
    GENERIC,
    build_method1_val_dataset,
    build_method2_val_datasets,
    evaluate_generic,
    evaluate_method1_representation,
    evaluate_method2_row,
    summarize_overall,
    summarize_row,
)
from amadeus_counterpoint.models.chessformer import Chessformer

# Must match the architecture every checkpoint here was trained with
# (scripts/train.py's "Chessformer 79M" configuration).
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

DEFAULT_STYLE_DIM = 32
DEFAULT_NUM_PLAYERS = 8


def build_base(checkpoint_path, device) -> Chessformer:
    base = Chessformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
        d1=D1, d2=D2, d3=D3, d_ff=D_FF, head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM, elo_dim=ELO_DIM,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    base.load_state_dict(payload["model"])
    base.requires_grad_(False)
    return base.to(device)


def peek_method1_checkpoint_identity(path, map_location=None) -> tuple[str, float]:
    """Read a Method-1 checkpoint's own saved identity/nominal_elo without
    constructing a wrapper -- used to feed load_method1_wrapper_for_generation
    the exact values the checkpoint itself claims, so the caller's mapping
    can never drift from what was actually trained."""
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint["identity"], checkpoint["nominal_elo"]


def print_matrix(title: str, rows: list[dict], id_to_name: dict[int, str]) -> None:
    print(f"\n=== {title}: NLL matrix (rows=true player, cols=representation) ===")
    header = "true \\ repr".ljust(24) + "".join(id_to_name[pid][:10].rjust(11) for pid in sorted(id_to_name))
    print(header)
    for row in rows:
        name = id_to_name[row["true_player"]][:22].ljust(24)
        cells = "".join(
            f"{row['nll_by_representation'][pid]:11.4f}" for pid in sorted(id_to_name)
        )
        print(name + cells)

    print(f"\n=== {title}: per-player summary ===")
    print(
        f"{'player':<24}{'generic':>10}{'correct':>10}{'mean_wrong':>12}"
        f"{'rank':>6}{'Δgeneric':>11}{'Δwrong':>9}"
    )
    for row in rows:
        name = id_to_name[row["true_player"]][:22].ljust(24)
        print(
            f"{name}{row['generic_nll']:>10.4f}{row['correct_nll']:>10.4f}"
            f"{row['mean_wrong_nll']:>12.4f}{row['correct_rank_among_8']:>6d}"
            f"{row['generic_minus_correct_nll']:>11.4f}{row['mean_wrong_minus_correct_nll']:>9.4f}"
        )


def print_method2_extra_metrics(rows: list[dict], id_to_name: dict[int, str]) -> None:
    print("\n=== METHOD 2: full-action baseline (descriptive only), coverage, deployable accuracy ===")
    print(
        f"{'player':<24}{'full_generic':>13}{'cand_matched':>13}"
        f"{'coverage':>10}{'deployable_acc':>16}"
    )
    for row in rows:
        name = id_to_name[row["true_player"]][:22].ljust(24)
        print(
            f"{name}{row['full_generic_nll']:>13.4f}{row['candidate_matched_generic_nll']:>13.4f}"
            f"{row['raw_topk_coverage']:>10.4f}{row['deployable_top1_accuracy_correct']:>16.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--method1-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--method2-checkpoint", type=Path, required=True)
    parser.add_argument("--targets-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--style-dim", type=int, default=DEFAULT_STYLE_DIM)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    targets = load_broadcast_targets(args.targets_config)
    canonical_id_to_name = {t["player_id"]: t["canonical_name"] for t in targets}
    player_ids = sorted(canonical_id_to_name)

    records = load_game_records(args.records_path)
    base = build_base(args.base_checkpoint, device)

    # -------------------------------------------------------------------
    # Method 1
    # -------------------------------------------------------------------
    print("\n### METHOD 1 ###")
    wrappers = {}
    for pid in player_ids:
        ckpt_path = args.method1_checkpoint_dir / f"player_{pid}.pt"
        identity, nominal_elo = peek_method1_checkpoint_identity(ckpt_path, map_location=device)
        if identity != canonical_id_to_name[pid]:
            raise ValueError(
                f"player_id={pid}: checkpoint identity {identity!r} does not match "
                f"targets-config canonical_name {canonical_id_to_name[pid]!r}"
            )
        wrappers[pid] = load_method1_wrapper_for_generation(
            base, ckpt_path, nominal_elo=nominal_elo, identity=identity, map_location=device,
        )

    method1_rows = []
    for true_pid in player_ids:
        val_dataset = build_method1_val_dataset(
            records, true_pid, train_fraction=args.train_fraction,
            split_seed=args.split_seed, balance_seed=args.balance_seed,
        )
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        generic_stats = evaluate_generic(base, dataloader, device)
        per_representation = {
            pid: evaluate_method1_representation(wrapper, dataloader, device)
            for pid, wrapper in wrappers.items()
        }
        method1_rows.append(summarize_row(true_pid, generic_stats, per_representation))
        print(f"  player {true_pid} ({canonical_id_to_name[true_pid]}): "
              f"n={method1_rows[-1]['n_positions']} done")

    method1_overall = summarize_overall(method1_rows)
    print_matrix("METHOD 1", method1_rows, canonical_id_to_name)

    # -------------------------------------------------------------------
    # Method 2
    # -------------------------------------------------------------------
    print("\n### METHOD 2 ###")
    cnn, table, residual, metadata = load_method2_style_stack_for_generation(
        base, args.method2_checkpoint, num_players=DEFAULT_NUM_PLAYERS,
        style_dim=args.style_dim, map_location=device,
    )
    k = metadata["k"]
    checkpoint_player_id_map = metadata["player_id_map"]
    if checkpoint_player_id_map is not None:
        checkpoint_id_to_name = {v: k_ for k_, v in checkpoint_player_id_map.items()}
        if checkpoint_id_to_name != canonical_id_to_name:
            raise ValueError(
                f"method2 checkpoint's player_id_map {checkpoint_id_to_name} does not "
                f"match targets-config {canonical_id_to_name}"
            )
    print(f"  k={k} (from checkpoint)")

    per_player_val = build_method2_val_datasets(
        records, player_ids, train_fraction=args.train_fraction,
        split_seed=args.split_seed, balance_seed=args.balance_seed,
    )

    method2_rows = []
    for true_pid in player_ids:
        val_dataset = per_player_val[true_pid]
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Descriptive only -- full 4352-action space, NOT used for deltas/rank
        # (see identity_diagnostic's module docstring for why not).
        full_generic_stats = evaluate_generic(base, dataloader, device)

        row_stats = evaluate_method2_row(base, cnn, table, residual, k, dataloader, player_ids, device)

        # Deltas/rank use the candidate-matched baseline: same oracle-expanded
        # K(+1) set every style representation is scored on, so it's
        # genuinely comparable to per_representation's NLLs.
        summary = summarize_row(true_pid, row_stats.candidate_matched_generic, row_stats.per_representation)
        summary["full_generic_nll"] = full_generic_stats.mean_nll
        summary["full_generic_accuracy"] = full_generic_stats.accuracy
        summary["candidate_matched_generic_nll"] = row_stats.candidate_matched_generic.mean_nll
        summary["candidate_matched_generic_accuracy"] = row_stats.candidate_matched_generic.accuracy
        summary["raw_topk_coverage"] = row_stats.coverage
        summary["deployable_top1_accuracy_by_representation"] = {
            pid: stats.accuracy for pid, stats in row_stats.deployable_per_representation.items()
        }
        summary["deployable_top1_accuracy_correct"] = row_stats.deployable_per_representation[true_pid].accuracy
        method2_rows.append(summary)
        print(f"  player {true_pid} ({canonical_id_to_name[true_pid]}): "
              f"n={summary['n_positions']} coverage={row_stats.coverage:.4f} "
              f"deployable_acc={summary['deployable_top1_accuracy_correct']:.4f} done")

    method2_overall = summarize_overall(method2_rows)
    print_matrix("METHOD 2 (candidate-matched, oracle-expanded)", method2_rows, canonical_id_to_name)
    print_method2_extra_metrics(method2_rows, canonical_id_to_name)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    output = {
        "id_to_name": canonical_id_to_name,
        "method1": {"rows": method1_rows, "overall": method1_overall},
        "method2": {
            "rows": method2_rows, "overall": method2_overall,
            "k": k,
            "note": (
                "generic_nll/generic_accuracy, nll_by_representation, "
                "accuracy_by_representation, rank, and deltas are all on the "
                "CANDIDATE-MATCHED oracle-expanded K(+1) set (same set for "
                "base/correct/wrong -- see candidate_matched_generic_*), NOT "
                "the full 4352-action space, and NOT directly comparable to "
                "method1's full-action metrics. full_generic_nll/"
                "full_generic_accuracy are the descriptive full-action-space "
                "baseline only, never subtracted for deltas/rank. "
                "deployable_top1_accuracy_by_representation is the separate, "
                "real-inference metric: raw top-K only, no target append, "
                "automatically wrong if the true move isn't in the raw top-K."
            ),
        },
    }
    args.output_json.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
