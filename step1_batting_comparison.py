"""Summarize batting outcomes for legal deliveries starting at 44-49 or 50-55."""

from collections import Counter
import csv
import gzip
from pathlib import Path


DATASET = Path(__file__).resolve().with_name("ipl_deliveries.csv.gz")
OUTCOMES = (
    ("Batter runs per legal ball", "batter_runs", 1, "runs/ball"),
    ("Boundary rate", "boundary", 100, "pp"),
    ("Batting dot rate", "batting_dot", 100, "pp"),
    ("Striker-dismissal rate", "striker_dismissed", 100, "pp"),
)


def main():
    if not DATASET.is_file():
        raise SystemExit("Dataset missing. Run: python build_ipl_deliveries.py ipl_json .")

    totals = {"pre": Counter(), "post": Counter()}
    with gzip.open(DATASET, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            group = totals.get(row["score_window"])
            if group is None or row["regulation_legal"] != "1":
                continue
            group["balls"] += 1
            for _, field, _, _ in OUTCOMES:
                group[field] += int(row[field])

    pre, post = totals["pre"], totals["post"]
    if not pre["balls"] or not post["balls"]:
        raise SystemExit("Both score groups need legal deliveries to calculate differences.")

    print("IPL BATTING OUTCOMES BY SCORE BEFORE THE BALL")
    print("Regulation innings | Legal deliveries | Descriptive comparison")
    print()
    print(f"{'Outcome':<30}{'44-49':>10}{'50-55':>10}{'44-49 minus 50-55':>21}")
    print(f"{'Legal deliveries':<30}{pre['balls']:>10,}{post['balls']:>10,}")
    for label, field, scale, unit in OUTCOMES:
        before = scale * pre[field] / pre["balls"]
        after = scale * post[field] / post["balls"]
        suffix = "%" if scale == 100 else ""
        difference = f"{before - after:+.2f} {unit}"
        print(f"{label:<30}{before:>9.2f}{suffix or ' '}"
              f"{after:>9.2f}{suffix or ' '}{difference:>21}")
    print()
    print("pp = percentage points. A batting dot is a legal ball with zero batter runs.")


if __name__ == "__main__":
    main()
