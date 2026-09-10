"""Tests against the REAL curated `configs/broadcast_targets.json`, not a
synthetic fixture -- these exist to pin down the exact behavior of the
2026-09-09 alias curation pass (Eval Stage A.2), so a future edit to the
config can't silently regress a specific verified string or reintroduce a
deliberately excluded one.

`match_target` itself is exact-FIDE-ID-then-exact-alias with no fuzzy or
substring matching; that behavior is already covered generically against a
synthetic fixture in `test_broadcast_targets.py`. This file is about the
curated data, not the matching algorithm.
"""

from amadeus_counterpoint.data.broadcast_ingest import classify_game, iter_target_game_records
from amadeus_counterpoint.data.broadcast_targets import (
    load_broadcast_targets,
    load_quarantined_game_urls,
    match_target,
)

CONFIG_PATH = "configs/broadcast_targets.json"
QUARANTINE_PATH = "configs/quarantined_broadcast_games.json"
TARGETS = load_broadcast_targets(CONFIG_PATH)

MAGNUS_CARLSEN = 0
WESLEY_SO = 1
LEVON_ARONIAN = 2
FABIANO_CARUANA = 3
MAXIME_VACHIER_LAGRAVE = 4
HIKARU_NAKAMURA = 5
IAN_NEPOMNIACHTCHI = 6
ALIREZA_FIROUZJA = 7


def test_no_alias_is_shared_across_two_targets():
    alias_to_targets: dict[str, set[int]] = {}
    for target in TARGETS:
        for alias in target["aliases"]:
            alias_to_targets.setdefault(alias, set()).add(target["player_id"])

    shared = {alias: pids for alias, pids in alias_to_targets.items() if len(pids) > 1}
    assert shared == {}


def test_no_target_has_a_duplicate_alias_in_its_own_list():
    for target in TARGETS:
        aliases = target["aliases"]
        assert len(aliases) == len(set(aliases)), target["canonical_name"]


# --- newly curated aliases resolve to exactly the intended target ----------


def test_title_prefixed_variants_resolve_correctly():
    assert match_target("GM Magnus Carlsen", "", TARGETS) == MAGNUS_CARLSEN
    assert match_target("GM Wesley So", "", TARGETS) == WESLEY_SO
    assert match_target("GM Levon Aronian", "", TARGETS) == LEVON_ARONIAN
    assert match_target("GM Fabiano Caruana", "", TARGETS) == FABIANO_CARUANA
    assert match_target("GM Maxime Vachier-Lagrave", "", TARGETS) == MAXIME_VACHIER_LAGRAVE
    assert match_target("GM Hikaru Nakamura", "", TARGETS) == HIKARU_NAKAMURA
    assert match_target("GM Ian Nepomniachtchi", "", TARGETS) == IAN_NEPOMNIACHTCHI
    assert match_target("GM Alireza Firouzja", "", TARGETS) == ALIREZA_FIROUZJA


def test_doubled_whitespace_country_code_variants_resolve_correctly():
    assert match_target("Carlsen Magnus  (NOR)", "", TARGETS) == MAGNUS_CARLSEN
    assert match_target("Aronian Levon  (ARM)", "", TARGETS) == LEVON_ARONIAN
    assert match_target("Caruana Fabiano  (USA)", "", TARGETS) == FABIANO_CARUANA
    assert match_target("Vachier-Lagrave Maxime  (FRA)", "", TARGETS) == MAXIME_VACHIER_LAGRAVE
    assert match_target("Nakamura Hikaru  (USA)", "", TARGETS) == HIKARU_NAKAMURA
    assert match_target("Nepomniachtchi Ian  (CFR)", "", TARGETS) == IAN_NEPOMNIACHTCHI
    assert match_target("Firouzja Alireza  (FRA)", "", TARGETS) == ALIREZA_FIROUZJA


def test_name_order_and_concatenation_variants_resolve_correctly():
    assert match_target("So Wesley", "", TARGETS) == WESLEY_SO
    assert match_target("LevonAronian", "", TARGETS) == LEVON_ARONIAN
    assert match_target("Ian Nepomniachtchi", "", TARGETS) == IAN_NEPOMNIACHTCHI
    assert match_target("Ian_Nepomniachtchi", "", TARGETS) == IAN_NEPOMNIACHTCHI


def test_federation_transfer_and_broadcast_artifact_variants_resolve_correctly():
    # Aronian Levon (USA): Levon Aronian's real federation transfer to the US.
    assert match_target("Aronian Levon  (USA)", "", TARGETS) == LEVON_ARONIAN
    # A broadcast-specific formatting artifact (club/ID token inserted between
    # first and last name); still unambiguously Maxime Vachier-Lagrave.
    assert match_target("Maxime ASN Vachier Lagrave", "", TARGETS) == MAXIME_VACHIER_LAGRAVE
    # A duplicated-name data-entry glitch; still unambiguously Hikaru Nakamura.
    assert match_target("Nakamura, Nakamura, Hikaru", "", TARGETS) == HIKARU_NAKAMURA


def test_included_handles_resolve_correctly():
    assert match_target("GMWSO", "", TARGETS) == WESLEY_SO
    assert match_target("LyonBeast", "", TARGETS) == MAXIME_VACHIER_LAGRAVE
    assert match_target("lachesisQ", "", TARGETS) == IAN_NEPOMNIACHTCHI


def test_kelvin2_synced_verified_aliases_resolve_correctly():
    # 2026-09-10 sync from the Kelvin2 amadeus-broadcast-data audit's
    # verified_name_strings (verified tier only -- see broadcast_targets.json
    # provenance for what was deliberately left out).
    assert match_target("MagzyBogues", "", TARGETS) == MAGNUS_CARLSEN
    assert match_target("GMHikaru", "", TARGETS) == HIKARU_NAKAMURA
    assert match_target("STL_Nakamura", "", TARGETS) == HIKARU_NAKAMURA
    assert match_target("STL_Caruana", "", TARGETS) == FABIANO_CARUANA
    assert match_target("M. Vachier-Lagrave", "", TARGETS) == MAXIME_VACHIER_LAGRAVE
    assert match_target("FantasticStar", "", TARGETS) == ALIREZA_FIROUZJA


def test_kelvin2_conservative_tier_was_not_imported():
    # 'Hikaru' and 'Offerspill-M. Carlsen' are Kelvin2's
    # conservative_candidate_name_strings, not verified_name_strings --
    # deliberately not synced (see configs/broadcast_targets.json provenance).
    assert match_target("Hikaru", "", TARGETS) is None
    assert match_target("Offerspill-M. Carlsen", "", TARGETS) is None


def test_known_ambiguous_string_remains_unresolved():
    # A distinct middle name ("Øen") could plausibly name a different person;
    # deliberately excluded despite superficially resembling Magnus Carlsen.
    assert match_target("Magnus Øen Carlsen", "", TARGETS) is None


def test_zero_and_one_target_behavior_is_unchanged():
    assert match_target("Some Random Amateur", "", TARGETS) is None
    assert match_target("Magnus Carlsen", "", TARGETS) == MAGNUS_CARLSEN


def test_still_no_fuzzy_or_substring_matching_against_the_real_config():
    # A near-miss typo and a superset string must both fail -- exact match only.
    assert match_target("Ian Nepomniachtch", "", TARGETS) is None
    assert match_target("Ian Nepomniachtchi Junior", "", TARGETS) is None


# --- integration: the fix actually flips classification --------------------


def test_previously_misclassified_nepomniachtchi_game_is_now_target_vs_target():
    classification, player_id, mover_color = classify_game(
        "Magnus Carlsen", "Ian Nepomniachtchi", "", "", TARGETS
    )
    assert classification == "target_vs_target"
    assert player_id is None
    assert mover_color is None


def _pgn(headers, moves="1. e4 e5 2. Nf3 Nc6"):
    header_lines = "\n".join(f'[{k} "{v}"]' for k, v in headers.items())
    return f"{header_lines}\n\n{moves}\n\n"


def test_real_quarantine_config_loads_and_is_nonempty():
    quarantined = load_quarantined_game_urls(QUARANTINE_PATH)
    # 12 'pedestrian' games (Carlsen- and Nakamura-side) + 6 'STL_So' games.
    assert len(quarantined) == 18
    assert all(url.startswith("https://lichess.org/broadcast/") for url in quarantined)


def test_pedestrian_game_that_would_otherwise_leak_is_quarantined(tmp_path):
    # Reproduces the exact newly-created risk: adding 'MagzyBogues' as a
    # verified Carlsen alias means a game vs the never-identified
    # 'pedestrian' handle would otherwise silently become one-target
    # Carlsen data, even though 'pedestrian' might secretly be another
    # target (see configs/quarantined_broadcast_games.json).
    quarantined = load_quarantined_game_urls(QUARANTINE_PATH)
    headers = {
        "Event": "Test",
        "White": "MagzyBogues",
        "Black": "pedestrian",
        "Result": "1/2-1/2",
        "WhiteElo": "2853",
        "BlackElo": "2500",
        "GameURL": "https://lichess.org/broadcast/chessable-masters-semi-finals-day-1/-/enmZrVwl/QJb6t6uC",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_pgn(headers), encoding="utf-8")

    stats = {}
    records = list(
        iter_target_game_records([path], TARGETS, stats=stats, quarantined_game_urls=quarantined)
    )

    assert records == []
    assert stats["quarantined_games"] == 1
    assert stats["one_target_games"] == 0


def test_corrected_target_vs_target_game_is_excluded_from_personalization_ingestion(tmp_path):
    headers = {
        "Event": "Test",
        "White": "Magnus Carlsen",
        "Black": "Ian Nepomniachtchi",
        "Result": "1-0",
        "WhiteElo": "2853",
        "BlackElo": "2777",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_pgn(headers), encoding="utf-8")

    stats = {}
    records = list(iter_target_game_records([path], TARGETS, stats=stats))

    assert records == []
    assert stats["target_vs_target_games"] == 1
    assert stats["one_target_games"] == 0
