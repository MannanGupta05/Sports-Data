"""Build and validate the IPL milestone dataset from a Cricsheet JSON directory."""

import argparse
from collections import Counter
import csv
import gzip
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory


PRE_RANGE = (44, 49)
POST_RANGE = (50, 55)
SCORE_RANGES = {"pre": PRE_RANGE, "post": POST_RANGE}

FIELDS = """
match_id date season innings batting_team batter batter_id bowler bowler_id
non_striker over actual_delivery delivery_sequence super_over
batter_score_before team_score_before legal_balls_used_before wickets_lost_before
wickets_remaining_before batter_runs extras_runs total_runs wide no_ball legal
boundary batting_dot striker_dismissed wickets_on_delivery wicket_events
batter_final_score regulation_legal score_window
chase_status recorded_target_runs recorded_target_overs target_runs_at_ball
balls_remaining_before runs_required_before required_run_rate_before
""".split()


def check(condition, description):
    if not condition:
        raise ValueError(f"Validation failed: {description}")


def deliveries(innings):
    for over in innings["overs"]:
        for ball in over["deliveries"]:
            yield over["over"], ball


def wicket_events(ball):
    return [w for w in ball.get("wickets", []) if w["kind"] != "retired hurt"]


def target_ball_count(overs, balls_per_over):
    whole, _, partial = str(overs).partition(".")
    last_balls = int(partial or "0")
    check(len(partial) <= 1 and last_balls < balls_per_over, f"target overs {overs}")
    return int(whole) * balls_per_over + last_balls


def chase_status(number, innings, info, first_total, balls_per_over):
    if innings.get("super_over"):
        return "super_over"
    if number != 2:
        return "not_a_chase"
    target = innings.get("target")
    if not target:
        return "missing_target"
    if info["outcome"].get("method") or info["outcome"].get("result") == "no result":
        return "adjusted_or_no_result"
    if target_ball_count(target["overs"], balls_per_over) != 20 * balls_per_over:
        return "shortened_allocation"
    if target["runs"] != first_total + 1:
        return "target_mismatch"
    if innings.get("miscounted_overs"):
        return "miscounted_overs"
    return "eligible"


def score_window(score):
    pre_min, pre_max = PRE_RANGE
    post_min, post_max = POST_RANGE
    if pre_min <= score <= pre_max:
        return "pre"
    if post_min <= score <= post_max:
        return "post"
    return ""


def match_rows(match_id, match):
    info = match["info"]
    people = info["registry"]["people"]
    balls_per_over = info["balls_per_over"]
    first_total = None
    for number, innings in enumerate(match["innings"], 1):
        super_over = int(innings.get("super_over", False))
        status = chase_status(number, innings, info, first_total, balls_per_over)
        target = innings.get("target", {})
        penalty = innings.get("penalty_runs", {})
        scores = Counter()
        team_score = penalty.get("pre", 0)
        legal_used = wickets_lost = 0
        rows = []

        for sequence, (over, ball) in enumerate(deliveries(innings), 1):
            batter = ball["batter"]
            score_before = scores[batter]
            runs = ball["runs"]
            batter_runs = runs["batter"]
            extras = ball.get("extras", {})
            wide = int("wides" in extras)
            no_ball = int("noballs" in extras)
            legal = int(not (wide or no_ball))
            dismissed = wicket_events(ball)
            striker_dismissed = int(any(w["player_out"] == batter for w in dismissed))
            eligible = int(legal and not super_over)
            check(runs["total"] == batter_runs + runs["extras"],
                  f"{match_id} innings {number} row {sequence} run balance")

            runs_required = balls_left = required_rate = target_at_ball = None
            if status == "eligible":
                target_at_ball = target["runs"]
                runs_required = target_at_ball - team_score
                balls_left = target_ball_count(target["overs"], balls_per_over) - legal_used
                check(balls_left > 0 and runs_required > 0,
                      f"{match_id} innings {number} row {sequence} chase state")
                required_rate = balls_per_over * runs_required / balls_left

            rows.append(dict(
                match_id=match_id, date=info["dates"][0], season=info["season"],
                innings=number, batting_team=innings["team"], batter=batter,
                batter_id=people[batter], bowler=ball["bowler"],
                bowler_id=people[ball["bowler"]], non_striker=ball["non_striker"],
                over=over + 1, actual_delivery=ball.get("actual_delivery", ""),
                delivery_sequence=sequence, super_over=super_over,
                batter_score_before=score_before, team_score_before=team_score,
                legal_balls_used_before=legal_used, wickets_lost_before=wickets_lost,
                wickets_remaining_before=(2 if super_over else 10) - wickets_lost,
                batter_runs=batter_runs, extras_runs=runs["extras"], total_runs=runs["total"],
                wide=wide, no_ball=no_ball, legal=legal,
                boundary=int(batter_runs in (4, 6) and not runs.get("non_boundary", False)),
                batting_dot=int(legal and batter_runs == 0),
                striker_dismissed=striker_dismissed, wickets_on_delivery=len(dismissed),
                wicket_events=json.dumps(
                    [[w["player_out"], w["kind"]] for w in ball.get("wickets", [])],
                    ensure_ascii=False,
                ),
                regulation_legal=eligible,
                score_window=score_window(score_before) if eligible else "",
                chase_status=status, recorded_target_runs=target.get("runs"),
                recorded_target_overs=target.get("overs"), target_runs_at_ball=target_at_ball,
                balls_remaining_before=balls_left, runs_required_before=runs_required,
                required_run_rate_before=required_rate,
            ))
            scores[batter] += batter_runs
            team_score += runs["total"]
            legal_used += legal
            wickets_lost += len(dismissed)

        if number == 1:
            first_total = team_score + penalty.get("post", 0)
        for row in rows:
            row["batter_final_score"] = scores[row["batter"]]
        yield number, status, rows, (team_score + penalty.get("post", 0), wickets_lost)


def validate_archive_examples(records, scorecards):
    def row(match_id, innings, label, predicate=lambda row: True):
        found = [r for r in records[match_id] if r["innings"] == innings
                 and r["actual_delivery"] == label and predicate(r)]
        check(len(found) == 1, f"{match_id} innings {innings} ball {label} lookup")
        return found[0]

    k = row("1082593", 1, "19.5")
    previous = row("1082593", 1, "19.4")
    check((previous["batter_score_before"], previous["batter_runs"], previous["boundary"])
          == (43, 4, 1), "Karthik at 43 hits a four")
    check((k["batter_score_before"], k["batter_final_score"], k["striker_dismissed"],
           k["score_window"]) == (47, 47, 1, "pre"),
          "Karthik's dismissal at 47 retained in the pre-50 sample")

    w = row("1082591", 1, "0.5", lambda r: r["wide"])
    after_wide = row("1082591", 1, "0.5", lambda r: r["legal"])
    check(w["legal"] == 0 and w["batter_runs"] == 0
          and after_wide["delivery_sequence"] == w["delivery_sequence"] + 1
          and after_wide["legal_balls_used_before"] == w["legal_balls_used_before"]
          and after_wide["team_score_before"] == w["team_score_before"] + 2,
          "wide preserves ball count and repeated label while adding team extras")
    n = row("1082591", 2, "9.1", lambda r: r["no_ball"])
    next_ball = row("1082591", 2, "9.1", lambda r: r["legal"])
    jadhav_returns = row("1082591", 2, "9.3")
    check(n["batter_runs"] == 1 and n["legal"] == 0
          and next_ball["balls_remaining_before"] == n["balls_remaining_before"]
          and next_ball["runs_required_before"] == n["runs_required_before"] - 2
          and jadhav_returns["batter"] == n["batter"]
          and jadhav_returns["batter_score_before"] == n["batter_score_before"] + 1,
          "scoring no-ball advances batter and team scores with no ball used")
    lb = row("1082591", 1, "0.6")
    check((lb["legal"], lb["batting_dot"], lb["total_runs"]) == (1, 1, 1),
          "leg-bye is a legal batting dot with a team run")
    ro = row("1082591", 2, "19.4")
    check(ro["wickets_on_delivery"] == 1 and ro["striker_dismissed"] == 0,
          "non-striker run-out counts as team wicket only")
    nb = row("1082594", 2, "12.4")
    check(nb["batter_runs"] == 4 and nb["boundary"] == 0,
          "four runs with non_boundary flag")
    opening = row("1082591", 2, "0.1")
    check(opening["target_runs_at_ball"] == 208 and opening["runs_required_before"] == 208
          and opening["balls_remaining_before"] == 120
          and abs(opening["required_run_rate_before"] - 10.4) < 1e-10,
          "208 from 120 balls starts at 10.40 RPO")
    check(target_ball_count("9.2", 6) == 56, "9.2 overs is 56 balls")
    shortened = next(r for r in records["1082648"] if r["innings"] == 2)
    check(shortened["recorded_target_runs"] == 48 and shortened["recorded_target_overs"] == 6
          and shortened["target_runs_at_ball"] is None
          and shortened["required_run_rate_before"] is None,
          "adjusted chase retains archive target with urgency blank")
    check(scorecards == {("1082591", 1): (207, 4), ("1082591", 2): (172, 10),
                         ("1082593", 1): (183, 4), ("1082593", 2): (184, 0)},
          "reconstructed team totals match both published IPL scorecards")
    kkr = [r for r in records["1082593"] if r["innings"] == 2 and r["score_window"] == "pre"]
    check(bool(kkr), "near-50 chase examples exist")
    return [
        ("Karthik hits four from 43", previous), ("Karthik dismissed on 47", k),
        ("Wide", w), ("Legal ball with same label", after_wide), ("Scoring no-ball", n),
        ("Next legal delivery", next_ball), ("Leg-bye", lb), ("Non-striker run-out", ro),
        ("Four runs without a boundary", nb), ("208-run chase opener", opening),
        ("Near-50 delivery in the chase", kkr[0]), ("Adjusted chase", shortened),
    ]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(source, destination):
    paths = sorted(source.glob("*.json"))
    check(bool(paths), f"no match JSON files in {source}")
    example_ids = {"1082591", "1082593", "1082594", "1082648"}
    examples = {}
    scorecards = {}
    totals = Counter()
    chase_log = []
    groups = {
        (sample, side): dict(matches=set(), batter_innings=set(), deliveries=0,
                             striker_dismissals=0)
        for sample, side in product(("all", "eligible_chases"), SCORE_RANGES)
    }

    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        compressed = stage / "ipl_deliveries.csv.gz"
        with gzip.open(compressed, "wt", encoding="utf-8", newline="", compresslevel=6) as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for path in paths:
                data = json.loads(path.read_text(encoding="utf-8"))
                totals["match_files"] += 1
                has_chase = False
                for number, status, rows, innings_totals in match_rows(path.stem, data):
                    writer.writerows(rows)
                    totals["innings"] += 1
                    totals["deliveries"] += len(rows)
                    if number == 2 and status != "super_over":
                        has_chase = True
                        target = data["innings"][1].get("target", {})
                        chase_log.append(dict(
                            match_id=path.stem, date=data["info"]["dates"][0], status=status,
                            outcome_method=data["info"]["outcome"].get("method", ""),
                            recorded_target_runs=target.get("runs"),
                            recorded_target_overs=target.get("overs"),
                            recorded_deliveries=len(rows),
                        ))
                    if path.stem in example_ids:
                        examples.setdefault(path.stem, []).extend(rows)
                        if path.stem in {"1082591", "1082593"} and number <= 2:
                            scorecards[path.stem, number] = innings_totals
                    for r in rows:
                        totals["legal_deliveries"] += r["legal"]
                        totals["regulation_legal_deliveries"] += r["regulation_legal"]
                        if status == "eligible":
                            totals["eligible_chase_deliveries"] += 1
                            totals["eligible_chase_legal_deliveries"] += r["legal"]
                        side = r["score_window"]
                        if side:
                            samples = ("all", "eligible_chases") if status == "eligible" else ("all",)
                            for sample in samples:
                                group = groups[sample, side]
                                group["deliveries"] += 1
                                group["matches"].add(path.stem)
                                group["batter_innings"].add((path.stem, number, r["batter_id"]))
                                group["striker_dismissals"] += r["striker_dismissed"]
                if not has_chase:
                    chase_log.append(dict(match_id=path.stem, date=data["info"]["dates"][0],
                                          status="no_second_innings", recorded_deliveries=0))

        summary = [dict(sample=sample, side=side,
                        score_min=SCORE_RANGES[side][0], score_max=SCORE_RANGES[side][1],
                        matches=len(group["matches"]), batter_innings=len(group["batter_innings"]),
                        deliveries=group["deliveries"], striker_dismissals=group["striker_dismissals"])
                   for (sample, side), group in groups.items()]
        if example_ids.issubset(examples):
            checked = validate_archive_examples(examples, scorecards)
            preview_fields = """
            example match_id innings actual_delivery delivery_sequence batter batter_score_before
            batter_runs legal boundary batting_dot striker_dismissed wickets_on_delivery
            score_window team_score_before target_runs_at_ball balls_remaining_before
            runs_required_before required_run_rate_before chase_status
            """.split()
            write_csv(stage / "validated_delivery_examples.csv", preview_fields,
                      [dict(r, example=label) for label, r in checked])
            example_status = "Passed all packaged IPL scorecard and delivery checks."
        else:
            example_status = "Example checks skipped because the four fixture files were absent."

        write_csv(stage / "sample_counts.csv", list(summary[0]), summary)
        write_csv(stage / "chase_eligibility.csv", """
                  match_id date status outcome_method recorded_target_runs recorded_target_overs
                  recorded_deliveries
                  """.split(), chase_log)
        status_counts = Counter(row["status"] for row in chase_log)
        check(sum(status_counts.values()) == len(paths), "one chase eligibility entry per match")
        log = [example_status, "",
               f"Starting score windows: {PRE_RANGE[0]}-{PRE_RANGE[1]} vs {POST_RANGE[0]}-{POST_RANGE[1]}",
               "", "Archive and sample audit",
               *(f"{key}: {value:,}" for key, value in totals.items()), "",
               "Chase eligibility by match",
               *(f"{key}: {value:,}" for key, value in sorted(status_counts.items()))]
        (stage / "validation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        for artifact in stage.iterdir():
            artifact.replace(destination / artifact.name)
    print("\n".join(log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    run(args.json_directory, args.output_directory)
