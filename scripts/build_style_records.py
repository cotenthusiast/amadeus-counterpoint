"""Build Stage-6 personalization ingestion records from the raw Broadcast
PGN corpus, for all 8 targets at once.

Writes the `data.broadcast_ingest.StyleGameRecord` JSON file that
`train_method1_player.py`, `train_method2.py`, and
`derive_representative_elos.py` all read via `--records-path`.

Quarantine (`configs/quarantined_broadcast_games.json`) is always applied --
there is no flag to skip it. Without it, a one-target game whose opponent is
an unverified handle that might secretly be another target (see that file's
provenance) would silently enter the resolved target's personalization data.

    python scripts/build_style_records.py \\
        --broadcast-root /home/cotenthusiast/Data/lichess_broadcasts \\
        --output-path data/style_records.json
"""

import argparse
import glob
import json
from pathlib import Path

from amadeus_counterpoint.data.broadcast_ingest import iter_target_game_records, save_game_records
from amadeus_counterpoint.data.broadcast_targets import load_broadcast_targets, load_quarantined_game_urls

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broadcast-root", type=Path, required=True)
    parser.add_argument("--targets-config", type=Path, default=CONFIG_DIR / "broadcast_targets.json")
    parser.add_argument(
        "--quarantine-config", type=Path, default=CONFIG_DIR / "quarantined_broadcast_games.json"
    )
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    targets = load_broadcast_targets(args.targets_config)
    quarantined_game_urls = load_quarantined_game_urls(args.quarantine_config)
    pgn_paths = sorted(glob.glob(str(args.broadcast_root / "lichess_db_broadcast_*.pgn")))

    stats = {}
    records = list(iter_target_game_records(
        pgn_paths, targets, stats=stats, quarantined_game_urls=quarantined_game_urls,
    ))

    save_game_records(records, args.output_path)
    print(f"wrote {len(records)} records to {args.output_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
