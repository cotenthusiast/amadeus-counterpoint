"""Derive and save the frozen representative Elo mapping from cohort-excluded
Broadcast personalization records.

Records must come from data.broadcast_ingest (one-target, cohort-excluded
games only) -- never sealed target-vs-target games.

    python scripts/derive_representative_elos.py \\
        --records-path data/style_records.json \\
        --output-path configs/representative_elos.json
"""

import argparse
import json
from pathlib import Path

from amadeus_counterpoint.data.broadcast_ingest import load_game_records
from amadeus_counterpoint.evaluation.representative_elo import compute_representative_elos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    records = load_game_records(args.records_path)
    representative_elos = compute_representative_elos(records)

    args.output_path.write_text(json.dumps(representative_elos, indent=2, sort_keys=True))
    for player_id, elo in sorted(representative_elos.items()):
        print(f"player_id={player_id}: representative_elo={elo}")


if __name__ == "__main__":
    main()
