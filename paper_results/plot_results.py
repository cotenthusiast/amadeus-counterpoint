"""Plot finalized per-dyad GG-to-AB TV changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHOD_FILES = {
    "M1": "method1_results.json",
    "M2": "method2_results.json",
    "M3": "method3_final_complete_results.json",
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
COLORS = {"M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}
MARKERS = {"M1": "o", "M2": "s", "M3": "^"}


def label_dyad(dyad: str) -> str:
    a, b = (int(part) for part in dyad.split("__"))
    return f"{SHORT[a]}–{SHORT[b]}"


def load_changes(source_dir: Path, metric: str) -> dict[str, dict[str, float]]:
    changes: dict[str, dict[str, float]] = {}
    for method, filename in METHOD_FILES.items():
        data = json.loads((source_dir / filename).read_text())
        for dyad, values in data["per_dyad"].items():
            changes.setdefault(dyad, {})[method] = (
                values["AB"][metric]["mean"] - values["GG"][metric]["mean"]
            )
    return changes


def plot(source_dir: Path, output_dir: Path, metric: str, stem: str, title: str) -> None:
    changes = load_changes(source_dir, metric)
    dyads = sorted(changes, key=lambda dyad: sum(changes[dyad].values()) / 3)
    y = list(range(len(dyads)))
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    offsets = {"M1": -0.20, "M2": 0.0, "M3": 0.20}
    for method in ("M1", "M2", "M3"):
        ax.scatter(
            [changes[dyad][method] for dyad in dyads],
            [value + offsets[method] for value in y],
            s=30,
            color=COLORS[method],
            marker=MARKERS[method],
            alpha=0.9,
            zorder=3,
        )
    ax.axvline(0.0, color="#555555", linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label_dyad(dyad) for dyad in dyads], fontsize=7.5)
    ax.set_xlabel("AB − GG TV distance (negative is better)")
    ax.set_title(title)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    legend = [
        Line2D([0], [0], marker=MARKERS[m], color="w", markerfacecolor=COLORS[m], markersize=7, label=m)
        for m in ("M1", "M2", "M3")
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent / "source")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(args.source_dir, args.output_dir, "wdl", "fig_dyad_wdl_change", "Dyad-level change in WDL-TV")
    plot(args.source_dir, args.output_dir, "opening", "fig_dyad_opening_change", "Dyad-level change in opening-family TV")
    print(f"wrote figures under {args.output_dir}")


if __name__ == "__main__":
    main()
