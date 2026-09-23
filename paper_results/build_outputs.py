"""Build paper-facing Amadeus result audits and LaTeX tables."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
AUDIT = ROOT.parent / "results_audit.md"
INVENTORY = ROOT / "artifact_inventory.json"
METHOD_FILES = {
    "M1": SOURCE / "method1_results.json",
    "M2": SOURCE / "method2_results.json",
    "M3": SOURCE / "method3_final_complete_results.json",
}
METRICS = {"wdl": "WDL-TV", "opening": "Opening-family TV"}
CONDITIONS = ("GG", "AG", "GB", "AB")
NAMES = {
    0: "Magnus Carlsen",
    1: "Wesley So",
    2: "Levon Aronian",
    3: "Fabiano Caruana",
    4: "Maxime Vachier-Lagrave",
    5: "Hikaru Nakamura",
    6: "Ian Nepomniachtchi",
    7: "Alireza Firouzja",
}
SHORT = {
    0: "Carlsen",
    1: "So",
    2: "Aronian",
    3: "Caruana",
    4: "Vachier-Lagrave",
    5: "Nakamura",
    6: "Nepomniachtchi",
    7: "Firouzja",
}
REMOTE = {
    "M1": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method1_results.json",
    "M2": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method2_results.json",
    "M3": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_final_complete_results.json",
    "identity": "/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json",
    "m1_marginal": "/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/nonsealed_real_comparison.json",
    "manifest": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md",
    "production_manifest": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/production_manifest.json",
    "m3_provenance": "/mnt/scratch2/users/40482774/paper_artifacts/m3_provenance_2026-09-17.json",
    "m3_provisional": "/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_provisional_results_missing_5__6_GB.json",
    "historical": "/mnt/scratch2/users/40482774/paper_artifacts/historical_round_robin_2019/analysis.json",
}


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def load_sources() -> dict[str, dict]:
    return {
        method: json.loads(path.read_text())
        for method, path in METHOD_FILES.items()
    }


def dyad_label(dyad: str) -> str:
    a, b = (int(part) for part in dyad.split("__"))
    return f"{SHORT[a]}--{SHORT[b]}"


def aggregate_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for method in ("M1", "M2", "M3"):
        for metric, label in METRICS.items():
            values = {
                condition: results[method]["equal_weight_aggregate"][condition][metric]
                for condition in CONDITIONS
            }
            rows.append({
                "method": method,
                "metric": label,
                "values": values,
                "ab_minus_gg": values["AB"] - values["GG"],
            })
    return rows


def dyad_rows(results: dict[str, dict], method: str, metric: str) -> list[dict]:
    rows = []
    for dyad, values in results[method]["per_dyad"].items():
        gg = values["GG"][metric]["mean"]
        ab = values["AB"][metric]["mean"]
        rows.append({
            "dyad": dyad,
            "label": dyad_label(dyad),
            "gg": gg,
            "ab": ab,
            "change": ab - gg,
        })
    return sorted(rows, key=lambda row: row["dyad"])


def dyad_summary(results: dict[str, dict], method: str, metric: str) -> dict:
    rows = dyad_rows(results, method, metric)
    changes = [row["change"] for row in rows]
    return {
        "dyads": len(rows),
        "improved_count": sum(change < 0 for change in changes),
        "improved_fraction": sum(change < 0 for change in changes) / len(changes),
        "mean": statistics.mean(changes),
        "median": statistics.median(changes),
        "min": min(changes),
        "max": max(changes),
        "largest_improvements": sorted(rows, key=lambda row: row["change"])[:3],
        "largest_regressions": sorted(rows, key=lambda row: row["change"], reverse=True)[:3],
    }


def inventory() -> dict:
    return {
        "authority_rule": "Use complete final sealed result JSONs; exclude historical tournament outputs.",
        "authoritative": [
            {
                "method": method,
                "remote_path": REMOTE[method],
                "local_mirror": str(METHOD_FILES[method]),
                "status": "finalized and complete",
            }
            for method in ("M1", "M2", "M3")
        ] + [
            {
                "role": "individual validation",
                "remote_path": REMOTE["identity"],
                "local_mirror": str(SOURCE / "personalization_identity_diagnostic.json"),
                "status": "finalized nonsealed diagnostic",
            },
            {
                "role": "M1 nonsealed marginal fidelity",
                "remote_path": REMOTE["m1_marginal"],
                "local_mirror": str(SOURCE / "m1_nonsealed_real_comparison.json"),
                "status": "finalized preliminary individual-fidelity diagnostic",
            },
            {"role": "sealed pre-evaluation manifest", "remote_path": REMOTE["manifest"]},
            {"role": "sealed production manifest", "remote_path": REMOTE["production_manifest"]},
            {"role": "M3 provenance", "remote_path": REMOTE["m3_provenance"]},
        ],
        "superseded_or_excluded": [
            {"path": REMOTE["m3_provisional"], "reason": "Superseded provisional M3 file; missing 5__6 GB."},
            {"path": REMOTE["historical"], "reason": "Historical tournament experiment explicitly excluded from the paper."},
            {
                "path": "/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/",
                "reason": "Preliminary M1 artifact, not the final sealed pairwise source.",
            },
        ],
        "known_conflicts_or_manual_checks": [
            "M3 provisional and complete files differ because the provisional file excluded 5__6 GB; use only the complete file.",
            "M1/M2 final evaluator commit is 980405aa..., while M3 generation provenance is 51ff259... and M3 final evaluation JSON is 5b56fb....",
            "The M1 nonsealed marginal artifact uses a self-defined opening taxonomy, not the pinned sealed taxonomy.",
            "Final JSONs contain per-dyad bootstrap intervals but no aggregate confidence-interval fields.",
        ],
    }


def individual_section(identity: dict) -> str:
    m1 = identity["method1"]
    m2 = identity["method2"]
    lines = [
        "# 1. Individual Behavioural Validation",
        "",
        "The authoritative individual-validation artifact is the nonsealed 8-way identity diagnostic. M1 uses legal-masked full-action policy NLL; M2 uses candidate-matched NLL on raw top-5 candidates with the observed move appended when absent. The NLL scales are not directly comparable.",
        "",
        "| Method | Positions / players | Correct representation rank 1 | Mean generic-minus-correct NLL | Mean wrong-minus-correct NLL |",
        "|---|---:|---:|---:|---:|",
        f"| M1 | {sum(row['n_positions'] for row in m1['rows']):,} / 8 | {m1['overall']['correct_rank_1_count']}/8 | {fmt(m1['overall']['mean_generic_minus_correct_nll'])} | {fmt(m1['overall']['mean_wrong_minus_correct_nll_avg'])} |",
        f"| M2 | {sum(row['n_positions'] for row in m2['rows']):,} / 8 | {m2['overall']['correct_rank_1_count']}/8 | {fmt(m2['overall']['mean_generic_minus_correct_nll'])} | {fmt(m2['overall']['mean_wrong_minus_correct_nll_avg'])} |",
        "",
        "| Player | Positions | M1 ΔNLL | M1 top-1 | M2 ΔNLL | M2 raw top-5 | M2 deployable top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r1, r2 in zip(m1["rows"], m2["rows"]):
        lines.append(
            f"| {NAMES[r1['true_player']]} | {r1['n_positions']:,} | {fmt(r1['generic_minus_correct_nll'])} | {r1['correct_accuracy']:.4f} | {fmt(r2['generic_minus_correct_nll'])} | {r2['raw_topk_coverage']:.4f} | {r2['deployable_top1_accuracy_correct']:.4f} |"
        )
    lines += [
        "",
        "M1 personalized generation was closer to held-out real individual-player distributions than the generic reference for 8/8 players on both first-move TV and opening-family TV. This is a nonsealed individual-fidelity diagnostic using a self-defined preliminary opening taxonomy; it is not the primary sealed interaction metric.",
        "",
        f"Sources: {REMOTE['identity']} and {REMOTE['m1_marginal']}.",
        "",
    ]
    return "\n".join(lines)


def pairwise_section(results: dict[str, dict]) -> str:
    lines = [
        "# 2. Sealed Pairwise Interaction Fidelity",
        "",
        "Total-variation distance is lower-is-better. Values use 50/50 orientation averaging and equal weighting across 28 dyads.",
        "",
        "| Method | Metric | GG | AG | GB | AB | AB − GG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows(results):
        values = row["values"]
        lines.append(
            f"| {row['method']} | {row['metric']} | {fmt(values['GG'])} | {fmt(values['AG'])} | {fmt(values['GB'])} | {fmt(values['AB'])} | {fmt(row['ab_minus_gg'])} |"
        )
    lines += [
        "",
        "The evaluator saved 10,000 whole-real-game bootstrap replicates per dyad and metric. Per-dyad intervals are present in the result JSONs; aggregate CI fields are not present and are not invented here.",
        "",
        f"Sources: {REMOTE['M1']}, {REMOTE['M2']}, {REMOTE['M3']}, and {REMOTE['manifest']}.",
        "",
    ]
    return "\n".join(lines)


def dyad_section(results: dict[str, dict]) -> str:
    lines = [
        "# 3. Dyad-Level Variation",
        "",
        "The following tables report AB − GG; negative values indicate lower AB TV distance.",
        "",
    ]
    for method in ("M1", "M2", "M3"):
        lines.append(f"## {method}")
        for metric, label in METRICS.items():
            summary = dyad_summary(results, method, metric)
            lines += [
                f"### {label}",
                "",
                f"{summary['improved_count']}/28 dyads ({summary['improved_fraction']:.1%}) improved; mean {fmt(summary['mean'])}, median {fmt(summary['median'])}, range [{fmt(summary['min'])}, {fmt(summary['max'])}].",
                "",
                "| Dyad | GG | AB | AB − GG |",
                "|---|---:|---:|---:|",
            ]
            for row in dyad_rows(results, method, metric):
                lines.append(f"| {row['label']} | {fmt(row['gg'])} | {fmt(row['ab'])} | {fmt(row['change'])} |")
            best = ", ".join(f"{row['label']} ({fmt(row['change'])})" for row in summary["largest_improvements"])
            worst = ", ".join(f"{row['label']} ({fmt(row['change'])})" for row in summary["largest_regressions"])
            lines += ["", f"Largest improvements: {best}.", f"Largest regressions: {worst}.", ""]
    lines.append(f"Source per-dyad data: {REMOTE['M1']}, {REMOTE['M2']}, {REMOTE['M3']}.")
    return "\n".join(lines) + "\n"


def m3_section(results: dict[str, dict]) -> str:
    m1 = results["M1"]["equal_weight_aggregate"]
    m2 = results["M2"]["equal_weight_aggregate"]
    m3 = results["M3"]["equal_weight_aggregate"]
    return "\n".join([
        "# 4. Exploratory Hybrid Model (M3)",
        "",
        "M3 is a post-hoc exploratory hybrid combining the already-trained M1 player vectors with the already-trained M2 candidate reranker. It introduced no new training or scientific hyperparameters.",
        "",
        "| Metric | M3 GG | M3 AG | M3 GB | M3 AB | AB − GG | M1 AB | M2 AB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| WDL-TV | {m3['GG']['wdl']:.6f} | {m3['AG']['wdl']:.6f} | {m3['GB']['wdl']:.6f} | {m3['AB']['wdl']:.6f} | {m3['AB']['wdl']-m3['GG']['wdl']:.6f} | {m1['AB']['wdl']:.6f} | {m2['AB']['wdl']:.6f} |",
        f"| Opening-family TV | {m3['GG']['opening']:.6f} | {m3['AG']['opening']:.6f} | {m3['GB']['opening']:.6f} | {m3['AB']['opening']:.6f} | {m3['AB']['opening']-m3['GG']['opening']:.6f} | {m1['AB']['opening']:.6f} | {m2['AB']['opening']:.6f} |",
        "",
        "M3 has its own generated GG baseline. The provisional file with missing 5__6 GB is superseded by the complete result file.",
        "",
        f"Sources: {REMOTE['M3']} and {REMOTE['m3_provenance']}.",
        "",
    ])


def counts_section() -> str:
    return "\n".join([
        "# 5. Dataset / Evaluation Counts",
        "",
        "- 8 target players and 28 unordered dyads.",
        "- 4 conditions × 2 orientations × 5,000 games = 224 cells and 1,120,000 games per method.",
        "- M1/M2 production: 224/224 cells each and 2,240,000 games combined. M3 final evaluation reports 28 evaluated dyads and no skipped cells.",
        "- Sealed evaluator cache: 1,609 deduplicated real games; exact pre-dedup counts are not recoverable from the cache alone.",
        "- Bootstrap: 10,000 whole-real-game replicates per dyad/metric.",
        "- Censoring: censored synthetic games are excluded from denominators and never silently counted as draws.",
        "",
        f"Sources: {REMOTE['manifest']} and {REMOTE['production_manifest']}.",
        "",
    ])


def conflict_section() -> str:
    return "\n".join([
        "# 6. Potential Conflicts / Things to Verify Manually",
        "",
        "1. M3 provisional versus complete: use only method3_final_complete_results.json.",
        "2. M1/M2 final evaluation commit 980405aa..., M3 generation provenance 51ff259..., and M3 final evaluator commit 5b56fb... should be preserved as separate provenance fields.",
        "3. M1 nonsealed marginal opening TV uses a self-defined taxonomy; do not merge it with final sealed opening-family TV.",
        "4. Aggregate bootstrap confidence intervals are not saved in the final JSONs; only per-dyad intervals are available.",
        "5. Historical Croatia/Sinquefield tournament artifacts are excluded from all paper outputs.",
        "",
    ])


def latex_tables(results: dict[str, dict], identity: dict) -> None:
    m1 = identity["method1"]["overall"]
    m2 = identity["method2"]["overall"]
    m1_top1 = sum(row["correct_accuracy"] for row in identity["method1"]["rows"]) / 8
    m2_top1 = sum(row["deployable_top1_accuracy_correct"] for row in identity["method2"]["rows"]) / 8
    m2_cov = sum(row["raw_topk_coverage"] for row in identity["method2"]["rows"]) / 8
    (ROOT / "table_individual_validation.tex").write_text("\n".join([
        "% Generated from paper_results/source/personalization_identity_diagnostic.json",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Individual behavioral validation on held-out nonsealed data. M2 NLL is candidate-matched.}",
        "\\label{tab:individual-validation}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Rank 1 & Mean $\\Delta$NLL & Wrong$-$correct & Top-1 / coverage & Positions \\\\",
        "\\midrule",
        f"M1 & 8/8 & {m1['mean_generic_minus_correct_nll']:.6f} & {m1['mean_wrong_minus_correct_nll_avg']:.6f} & {m1_top1:.4f} & {sum(row['n_positions'] for row in identity['method1']['rows']):,} \\\\",
        f"M2 & 8/8 & {m2['mean_generic_minus_correct_nll']:.6f} & {m2['mean_wrong_minus_correct_nll_avg']:.6f} & {m2_top1:.4f} / {m2_cov:.4f} & {sum(row['n_positions'] for row in identity['method2']['rows']):,} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]))
    lines = [
        "% Generated from finalized sealed result JSONs.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Sealed target--target interaction fidelity. Lower TV is better; the final column is $AB-GG$.}",
        "\\label{tab:pairwise-main}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Method & Metric & GG & AG & GB & AB & $AB-GG$ \\\\",
        "\\midrule",
    ]
    for row in aggregate_rows(results):
        v = row["values"]
        metric = row["metric"].replace("Opening-family TV", "Opening-TV")
        lines.append(f"{row['method']} & {metric} & {v['GG']:.6f} & {v['AG']:.6f} & {v['GB']:.6f} & {v['AB']:.6f} & {row['ab_minus_gg']:.6f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (ROOT / "table_pairwise_main.tex").write_text("\n".join(lines))
    m1a = results["M1"]["equal_weight_aggregate"]["AB"]
    m2a = results["M2"]["equal_weight_aggregate"]["AB"]
    m3a = results["M3"]["equal_weight_aggregate"]["AB"]
    (ROOT / "table_m3.tex").write_text("\n".join([
        "% Generated from finalized sealed result JSONs. M3 is exploratory.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exploratory M3 hybrid compared with M1 and M2 AB conditions.}",
        "\\label{tab:m3}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & M1 AB & M2 AB & M3 AB \\\\",
        "\\midrule",
        f"WDL-TV & {m1a['wdl']:.6f} & {m2a['wdl']:.6f} & {m3a['wdl']:.6f} \\\\",
        f"Opening-TV & {m1a['opening']:.6f} & {m2a['opening']:.6f} & {m3a['opening']:.6f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]))
    lines = [
        "% Generated from finalized sealed result JSONs.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dyad-level $AB-GG$ variation; negative values indicate lower AB TV.}",
        "\\label{tab:dyad-summary}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Method & Metric & Improved & Mean & Median & Min & Max \\\\",
        "\\midrule",
    ]
    for method in ("M1", "M2", "M3"):
        for metric, label in METRICS.items():
            s = dyad_summary(results, method, metric)
            label = label.replace("Opening-family TV", "Opening-TV")
            lines.append(f"{method} & {label} & {s['improved_count']}/28 & {s['mean']:.6f} & {s['median']:.6f} & {s['min']:.6f} & {s['max']:.6f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (ROOT / "table_dyad_summary.tex").write_text("\n".join(lines))


def cheatsheet(results: dict[str, dict], identity: dict) -> None:
    lines = [
        "# Amadeus Results Cheatsheet",
        "",
        f"Primary sealed sources: {REMOTE['M1']}; {REMOTE['M2']}; {REMOTE['M3']}. Historical tournament outputs are excluded.",
        "",
        "## Individual validation",
        "",
        f"- M1: correct representation rank 1 for {identity['method1']['overall']['correct_rank_1_count']}/8; mean generic−correct NLL = {identity['method1']['overall']['mean_generic_minus_correct_nll']:.6f}.",
        f"- M2: correct representation rank 1 for {identity['method2']['overall']['correct_rank_1_count']}/8; mean candidate-matched generic−correct NLL = {identity['method2']['overall']['mean_generic_minus_correct_nll']:.6f}; raw top-5 coverage range = {min(row['raw_topk_coverage'] for row in identity['method2']['rows']):.4f}–{max(row['raw_topk_coverage'] for row in identity['method2']['rows']):.4f}.",
        "",
        "## Main interaction results",
        "",
    ]
    for method in ("M1", "M2", "M3"):
        lines.append(f"### {method}")
        for metric, label in METRICS.items():
            row = next(row for row in aggregate_rows(results) if row["method"] == method and row["metric"] == label)
            v = row["values"]
            lines.append(f"- {label}: GG {v['GG']:.6f}; AG {v['AG']:.6f}; GB {v['GB']:.6f}; AB {v['AB']:.6f}; AB−GG {row['ab_minus_gg']:.6f}.")
        lines.append("")
    lines += ["## Dyad-level", ""]
    for method in ("M1", "M2", "M3"):
        for metric, label in METRICS.items():
            s = dyad_summary(results, method, metric)
            lines.append(f"- {method} {label}: {s['improved_count']}/28 improve; median AB−GG {s['median']:.6f}; range [{s['min']:.6f}, {s['max']:.6f}].")
    lines += ["", "## Counts", "", "- 8 players, 28 dyads, 224 cells/method, 5,000 games/orientation, 1,120,000 games/method.", "- 10,000 whole-real-game bootstrap replicates per dyad/metric.", "- Sealed cache: 1,609 deduplicated real games.", ""]
    (ROOT / "results_cheatsheet.md").write_text("\n".join(lines))


def dyad_csv(results: dict[str, dict]) -> None:
    path = ROOT / "dyad_changes.csv"
    fields = ["method", "metric", "dyad", "player_a", "player_b", "gg", "ab", "ab_minus_gg"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in ("M1", "M2", "M3"):
            for metric in ("wdl", "opening"):
                for row in dyad_rows(results, method, metric):
                    a, b = row["dyad"].split("__")
                    writer.writerow({
                        "method": method,
                        "metric": METRICS[metric],
                        "dyad": row["dyad"],
                        "player_a": NAMES[int(a)],
                        "player_b": NAMES[int(b)],
                        "gg": f"{row['gg']:.12f}",
                        "ab": f"{row['ab']:.12f}",
                        "ab_minus_gg": f"{row['change']:.12f}",
                    })


def main() -> None:
    results = load_sources()
    identity = json.loads((SOURCE / "personalization_identity_diagnostic.json").read_text())
    (ROOT / "artifact_inventory.json").write_text(json.dumps(inventory(), indent=2) + "\n")
    latex_tables(results, identity)
    cheatsheet(results, identity)
    dyad_csv(results)
    sections = [
        "# Amadeus Results Audit",
        "",
        "Scope: latest finalized M1/M2/M3 sealed pairwise results and finalized nonsealed individual-validation artifacts. The historical Croatia GCT / Sinquefield tournament experiment is deliberately excluded.",
        "",
        "## Authority and source policy",
        "",
        "The complete final sealed JSONs are authoritative for pairwise results. Small summary mirrors under paper_results/source are used only to build these outputs; production games and checkpoints remain on Kelvin2.",
        "",
        individual_section(identity),
        pairwise_section(results),
        dyad_section(results),
        m3_section(results),
        counts_section(),
        conflict_section(),
    ]
    AUDIT.write_text("\n".join(sections))
    print(f"wrote {AUDIT}")
    print(f"wrote {ROOT / 'artifact_inventory.json'}")


if __name__ == "__main__":
    main()
