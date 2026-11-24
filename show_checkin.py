"""Print the data preparation and validation progress for the IPL check-in."""

from collections import Counter
import csv
import gzip
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "ipl_deliveries.csv.gz"
SCORE_GROUPS = {"pre": (44, 49), "post": (50, 55)}


def main():
    if not DATASET.is_file():
        raise SystemExit("Dataset missing. Run: python build_ipl_deliveries.py ipl_json .")

    matches = set()
    verified_chases = set()
    counts = Counter()
    batter_innings = {"pre": set(), "post": set()}
    examples = {}
    total_deliveries = 0

    with gzip.open(DATASET, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            total_deliveries += 1
            match_id = row["match_id"]
            matches.add(match_id)
            eligible_chase = row["chase_status"] == "eligible"
            if eligible_chase:
                verified_chases.add(match_id)

            side = row["score_window"]
            if side in SCORE_GROUPS:
                counts["regulation", side] += 1
                batter_innings[side].add((match_id, row["innings"], row["batter_id"]))
                if eligible_chase:
                    counts["chase", side] += 1

            if match_id == "1082593" and row["legal"] == "1":
                if row["innings"] == "1" and row["batter"] == "KD Karthik":
                    if row["actual_delivery"] in {"19.4", "19.5"}:
                        examples[row["actual_delivery"]] = row
                if row["innings"] == "2" and row["batter"] == "CA Lynn":
                    if row["actual_delivery"] == "6.6":
                        examples["6.6"] = row

    four, wicket, chase = (examples[label] for label in ("19.4", "19.5", "6.6"))
    print("IPL HALF-CENTURY PROJECT | CHECK-IN")
    print()
    print("1. COMPARE DELIVERIES BY THE STRIKER'S SCORE BEFORE THE BALL")
    print("   44-49 before the milestone | 50-55 after the milestone")
    print("   Both groups use legal deliveries from regulation innings.")
    print()
    print("2. DELIVERY DATA AND SAMPLE COVERAGE")
    print(f"   {len(matches):,} matches | {total_deliveries:,} recorded deliveries")
    print(f"   {'Score':<10}{'Legal balls':>15}{'Batter-innings':>18}{'Chase balls':>15}")
    for side, (low, high) in SCORE_GROUPS.items():
        score = f"{low}-{high}"
        print(f"   {score:<10}{counts['regulation', side]:>15,}"
              f"{len(batter_innings[side]):>18,}{counts['chase', side]:>15,}")
    print("   Chase balls are a subset of legal balls.")
    print()
    print("3. VALIDATION EXAMPLE | DINESH KARTHIK")
    print(f"   Ball 19.4: starts on {four['batter_score_before']}, "
          f"scores {four['batter_runs']}, moves to {wicket['batter_score_before']}.")
    print(f"   Ball 19.5: starts on {wicket['batter_score_before']}, "
          f"{'dismissed' if wicket['striker_dismissed'] == '1' else 'continues batting'}, "
          f"labelled {wicket['score_window']}-50.")
    print()
    print("4. CHASE URGENCY EXAMPLE | CHRIS LYNN")
    print(f"   Required run rate is available for {len(verified_chases):,} standard chases.")
    print(f"   Lynn starts ball 6.6 on {chase['batter_score_before']}. "
          f"KKR needs {chase['runs_required_before']} off {chase['balls_remaining_before']} balls.")
    rate = float(chase["required_run_rate_before"])
    print(f"   Required run rate = 6 x {chase['runs_required_before']} "
          f"/ {chase['balls_remaining_before']} = {rate:.2f} runs per over.")
    print()


if __name__ == "__main__":
    main()
