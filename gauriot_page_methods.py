"""Gauriot and Page (2015), Figures 1A and 2A, applied to IPL data."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import WLS


CUTOFF = 50
BANDWIDTH = 5
BIN_WIDTH = 1
MIN_FINAL_STRIKE_RATE = 70
FIRST_SCORE = 44
LAST_SCORE = 55


def dismissal_scores(deliveries):
    """Recover one final personal score per dismissed regulation batter-innings."""
    keys = ["match_id", "innings", "batter"]
    batting = deliveries.groupby(keys, sort=False)["batter_runs"].sum()
    records = []
    events = deliveries.loc[deliveries["wicket_events"].ne("[]"),
                            ["match_id", "innings", "wicket_events"]]
    for match_id, innings, wickets in events.itertuples(index=False, name=None):
        for batter, kind in json.loads(wickets):
            if kind != "retired hurt":
                # A non-striker can be dismissed before taking strike and have zero runs.
                final_score = batting.get((match_id, innings, batter), 0)
                records.append((match_id, innings, batter, final_score))
    return pd.DataFrame(records, columns=[*keys, "final_score"]).drop_duplicates(keys)


def local_density_limit(midpoints, density, right):
    """Triangular local-linear fit on one side of the McCrary cutoff."""
    distance = midpoints - CUTOFF
    side = distance >= 0 if right else distance < 0
    use = side & (np.abs(distance) < BANDWIDTH)
    x = distance[use]
    design = np.column_stack((np.ones(len(x)), x))
    root_weights = np.sqrt(1 - np.abs(x) / BANDWIDTH)
    estimates, _, rank, _ = np.linalg.lstsq(
        design * root_weights[:, None], density[use] * root_weights, rcond=None
    )
    if rank < 2:
        raise ValueError("Insufficient score bins for a local density fit.")
    return estimates


def save_threshold_plot(path, ylabel, score, values, fits, caption, bin_midpoint_offset=0):
    fig, ax = plt.subplots(figsize=(7.7, 4.7), layout="constrained")
    colors = ("#1f6384", "#ad5027")
    left = score < CUTOFF
    ax.scatter(score[left], values[left], color=colors[0], s=32, label="Below 50")
    ax.scatter(score[~left], values[~left], color=colors[1], s=32, label="50 and above")
    for (x, y), color in zip(fits, colors):
        ax.plot(x, y, color=color, linewidth=2.2)
    ax.axvline(CUTOFF, color="#606a76", linewidth=1, linestyle=":")
    ticks = np.arange(FIRST_SCORE, LAST_SCORE + 1)
    ax.set_xticks(ticks + bin_midpoint_offset, ticks)
    ax.set(xlim=(43.6, 56), ylabel=ylabel)
    ax.set_xlabel(caption, labelpad=9)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mc_crary(dismissals, output):
    """Estimate log density(50+) minus log density(50-) using McCrary (2008)."""
    scores = np.arange(FIRST_SCORE, LAST_SCORE + 1)
    counts = dismissals["final_score"].value_counts().reindex(scores, fill_value=0)
    n = len(dismissals)
    midpoints = scores + BIN_WIDTH / 2
    density = counts.to_numpy() / (n * BIN_WIDTH)
    left = local_density_limit(midpoints, density, right=False)
    right = local_density_limit(midpoints, density, right=True)
    f_left, f_right = left[0], right[0]
    if min(f_left, f_right) <= 0:
        raise ValueError("A fitted density at 50 was nonpositive, so its log is undefined.")
    jump = math.log(f_right / f_left)
    se = math.sqrt(24 / (5 * n * BANDWIDTH) * (1 / f_left + 1 / f_right))
    critical = norm.ppf(0.975)
    grids = [np.linspace(45, 50, 60), np.linspace(50, 55, 60)]
    fits = [(x, coefficients[0] + coefficients[1] * (x - CUTOFF))
            for x, coefficients in zip(grids, (left, right))]
    save_threshold_plot(
        output / "paper_dismissal_density.png", "Share of all dismissals per one-run bin",
        midpoints, density, fits, "Final score at dismissal",
        bin_midpoint_offset=BIN_WIDTH / 2,
    )
    pd.DataFrame({"final_score": scores, "bin_midpoint": midpoints,
                  "dismissal_count": counts.to_numpy(), "density": density}).to_csv(
        output / "paper_dismissal_bins.csv", index=False
    )
    local = np.abs(midpoints - CUTOFF) < BANDWIDTH
    return dict(
        method="McCrary test on final dismissal scores", unit="log density ratio",
        contrast="right minus left at 50", estimate=jump, standard_error=se,
        ci_low=jump - critical * se, ci_high=jump + critical * se,
        p_value=2 * norm.sf(abs(jump / se)), left_at_50=f_left, right_at_50=f_right,
        percent_jump=100 * math.expm1(jump), observations=int(counts.to_numpy()[local].sum()),
        total_dismissals=n, bandwidth_runs=BANDWIDTH, bin_width_runs=BIN_WIDTH,
    )


def strike_rate_discontinuity(legal, output):
    """Fit the published 5-run triangular design in IPL innings reaching 70."""
    plot_sample = legal.loc[legal["batter_final_score"].ge(MIN_FINAL_STRIKE_RATE)
                            & legal["batter_score_before"].between(FIRST_SCORE, LAST_SCORE)]
    sample = plot_sample.copy()
    sample["distance"] = sample["batter_score_before"] - CUTOFF
    sample = sample.loc[sample["distance"].abs().lt(BANDWIDTH)].copy()
    distance = sample["distance"]
    post = distance.ge(0).astype(float)
    design = pd.DataFrame({"constant": 1.0, "post": post, "distance": distance,
                           "post_x_distance": post * distance}, index=sample.index)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Insufficient strike-rate observations on both sides of 50.")
    model = WLS(sample["batter_runs"], design, weights=1 - distance.abs() / BANDWIDTH)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sample["match_id"]}, use_t=True)
    params = result.params
    intervals = result.conf_int().loc["post"]
    f_left, f_right = params["constant"], params["constant"] + params["post"]
    by_score = plot_sample.groupby("batter_score_before")["batter_runs"].agg(
        legal_balls="size", runs_per_ball="mean"
    )
    by_score.to_csv(output / "paper_strike_rate_by_score.csv")
    grids = [np.linspace(sample.loc[distance.lt(0), "batter_score_before"].min(), 50, 60),
             np.linspace(50, sample.loc[distance.ge(0), "batter_score_before"].max(), 60)]
    fits = [(grids[0], f_left + params["distance"] * (grids[0] - CUTOFF)),
            (grids[1], f_right + (params["distance"] + params["post_x_distance"])
             * (grids[1] - CUTOFF))]
    save_threshold_plot(
        output / "paper_strike_rate.png", "Batter runs per legal ball",
        by_score.index.to_numpy(), by_score["runs_per_ball"].to_numpy(), fits,
        "Striker's score before the ball | Batter-innings reaching at least 70",
    )
    return dict(
        method="Triangular local-linear strike-rate comparison", unit="runs per legal ball",
        contrast="right minus left at 50", estimate=params["post"], standard_error=result.bse["post"],
        ci_low=intervals.iloc[0], ci_high=intervals.iloc[1], p_value=result.pvalues["post"],
        left_at_50=f_left, right_at_50=f_right, percent_jump=100 * params["post"] / f_left,
        observations=len(sample), match_clusters=sample["match_id"].nunique(),
        bandwidth_runs=BANDWIDTH, minimum_final_score=MIN_FINAL_STRIKE_RATE,
    )


def run_paper_methods(regulation, output):
    """Run both of the paper's half-century batting comparisons."""
    density = mc_crary(dismissal_scores(regulation), output)
    strike_rate = strike_rate_discontinuity(regulation.loc[regulation["legal"].eq(1)], output)
    results = pd.DataFrame([density, strike_rate])
    results.to_csv(output / "paper_tests.csv", index=False)
    return results
