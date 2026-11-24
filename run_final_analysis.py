"""Run the IPL half-century models and the Gauriot-Page batting comparisons."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster

from gauriot_page_methods import run_paper_methods


PROJECT = Path(__file__).resolve().parent
OUTCOMES = {
    "batter_runs": ("Runs per legal ball", "runs per legal ball", 1),
    "boundary": ("Boundary rate", "percentage points", 100),
    "batting_dot": ("Batting dot rate", "percentage points", 100),
    "striker_dismissed": ("Striker-dismissal rate", "percentage points", 100),
}
COLUMNS = [
    "match_id", "innings", "batter", "batter_id", "super_over", "legal",
    "regulation_legal", "batter_score_before", "batter_final_score", "score_window",
    "over", "wickets_remaining_before", "chase_status", "required_run_rate_before",
    "wicket_events", *OUTCOMES,
]


def load_deliveries(path):
    deliveries = pd.read_csv(path, usecols=COLUMNS, dtype={"match_id": str})
    regulation = deliveries.loc[deliveries["super_over"].eq(0)].copy()
    sample = regulation.loc[regulation["regulation_legal"].eq(1)
                            & regulation["batter_score_before"].between(44, 55)].copy()
    sample["pre"] = sample["batter_score_before"].lt(50).astype(float)
    expected = np.where(sample["pre"].eq(1), "pre", "post")
    if not np.array_equal(expected, sample["score_window"].to_numpy()):
        raise ValueError("Delivery score windows disagree with the 44-49 vs 50-55 specification.")
    return regulation, sample


def design_for(sample, adjusted):
    design = sample[["pre"]].astype(float).copy()
    design.insert(0, "constant", 1.0)
    if adjusted:
        controls = pd.get_dummies(
            sample[["over", "wickets_remaining_before"]].astype("category"),
            drop_first=True, dtype=float,
        )
        design = design.join(controls)
    return design


def fit_outcomes(sample, design, fixed_effects=None, absorbed_parameters=0,
                 required_terms=("pre",)):
    response = sample[list(OUTCOMES)].astype(float)
    if fixed_effects is not None:
        # Removing group means is equivalent to adding one intercept for each group.
        design = design - design.groupby(fixed_effects).transform("mean")
        design = design.loc[:, design.abs().max().gt(1e-10)]
        response = response - response.groupby(fixed_effects).transform("mean")

    if np.linalg.matrix_rank(design) != design.shape[1] or not set(required_terms) <= set(design):
        raise ValueError("Model cannot separately estimate the score-group coefficient.")
    clusters = sample["match_id"]
    n, k = design.shape
    g = clusters.nunique()
    if g < 2 or n <= k + absorbed_parameters:
        raise ValueError("Insufficient observations or match clusters for inference.")
    # CR1 correction. Match-nested innings effects are excluded from the parameter penalty.
    correction = g / (g - 1) * (n - 1) / (n - k - absorbed_parameters)
    models = {}
    for field in OUTCOMES:
        result = OLS(response[field], design).fit()
        covariance = pd.DataFrame(cov_cluster(result, clusters, use_correction=False) * correction,
                                  index=design.columns, columns=design.columns)
        models[field] = (result.params, covariance, g - 1)
    return models


def contrast(model, weights, scale):
    coefficients, covariance, df = model
    terms = list(weights)
    vector = np.array(list(weights.values()))
    estimate = float(vector @ coefficients.loc[terms]) * scale
    se = np.sqrt(float(vector @ covariance.loc[terms, terms] @ vector)) * scale
    critical = t.ppf(0.975, df)
    p_value = 2 * t.sf(abs(estimate / se), df) if se > 0 else np.nan
    return dict(estimate=estimate, standard_error=se, ci_low=estimate - critical * se,
                ci_high=estimate + critical * se, p_value=p_value)


def summary_rows(sample, models, name, weights, contrast_label, unit_suffix=""):
    rows = []
    for field, (label, unit, scale) in OUTCOMES.items():
        rows.append(dict(
            specification=name, outcome=label, unit=unit + unit_suffix, contrast=contrast_label,
            legal_balls=len(sample), pre_balls=int(sample["pre"].sum()),
            post_balls=int(sample["pre"].eq(0).sum()), match_clusters=sample["match_id"].nunique(),
            **contrast(models[field], weights, scale),
        ))
    results = pd.DataFrame(rows)
    results["p_holm_four_outcomes"] = np.nan
    finite = results["p_value"].notna()
    results.loc[finite, "p_holm_four_outcomes"] = multipletests(
        results.loc[finite, "p_value"], method="holm"
    )[1]
    return results


def compare_all_innings(sample, output):
    raw = summary_rows(sample, fit_outcomes(sample, design_for(sample, adjusted=False)),
                       "Raw comparison", {"pre": 1}, "44-49 minus 50-55")
    for column, group in (("mean_44_49", 1), ("mean_50_55", 0)):
        raw[column] = [sample.loc[sample["pre"].eq(group), field].mean() * scale
                       for field, (_, _, scale) in OUTCOMES.items()]
    adjusted = summary_rows(sample, fit_outcomes(sample, design_for(sample, adjusted=True)),
                            "Over and wickets remaining fixed effects", {"pre": 1},
                            "44-49 minus 50-55")
    results = pd.concat([raw, adjusted], ignore_index=True)
    results.to_csv(output / "batting_comparison.csv", index=False)
    return results


def compare_placebos(regulation, main_comparison, output):
    """Apply the main specification at 40 and 60 and compare the gaps to 50."""
    thresholds = (40, 50, 60)
    windows = {}
    for threshold in thresholds:
        sample = regulation.loc[
            regulation["regulation_legal"].eq(1)
            & regulation["batter_score_before"].between(threshold - 6, threshold + 5)
        ].copy()
        sample["pre"] = sample["batter_score_before"].lt(threshold).astype(float)
        windows[threshold] = sample

    results = [
        main_comparison.assign(threshold=50, pre_range="44-49", post_range="50-55")
    ]
    spec_names = ("Raw comparison", "Over and wickets remaining fixed effects")
    for adjusted, specification in enumerate(spec_names):
        for threshold in (40, 60):
            sample = windows[threshold]
            pre_range = f"{threshold - 6}-{threshold - 1}"
            post_range = f"{threshold}-{threshold + 5}"
            models = fit_outcomes(sample, design_for(sample, adjusted=bool(adjusted)))
            summary = summary_rows(sample, models, specification, {"pre": 1},
                                   f"{pre_range} minus {post_range}")
            if not adjusted:
                for column, group in (("mean_44_49", 1), ("mean_50_55", 0)):
                    summary[column] = [
                        sample.loc[sample["pre"].eq(group), field].mean() * scale
                        for field, (_, _, scale) in OUTCOMES.items()
                    ]
            results.append(summary.assign(threshold=threshold, pre_range=pre_range,
                                          post_range=post_range))

    table = pd.concat(results, ignore_index=True).rename(
        columns={"mean_44_49": "pre_mean", "mean_50_55": "post_mean"}
    )
    table = pd.concat([table.loc[table["threshold"].eq(threshold)] for threshold in thresholds],
                      ignore_index=True)
    leading = ["threshold", "pre_range", "post_range"]
    table = table[leading + [column for column in table if column not in leading]]
    table.to_csv(output / "placebo_comparison.csv", index=False)

    # Each threshold has its own intercept, pre coefficient, and controls.
    # Clustering the stacked data jointly captures covariance across threshold
    # estimates, including deliveries included in more than one score window.
    stacked = pd.concat(windows.values(), ignore_index=True)
    direct = []
    for adjusted, specification in enumerate(spec_names):
        blocks = [design_for(windows[threshold], adjusted=bool(adjusted)).add_suffix(f"_{threshold}")
                  for threshold in thresholds]
        pooled = pd.concat(blocks, ignore_index=True).fillna(0)
        models = fit_outcomes(stacked, pooled, required_terms=tuple(f"pre_{k}" for k in thresholds))
        for threshold in (40, 60):
            for field, (label, unit, scale) in OUTCOMES.items():
                direct.append(dict(
                    threshold=threshold, reference_threshold=50, specification=specification,
                    outcome=label, unit=unit,
                    contrast=f"Gap at {threshold} minus gap at 50",
                    match_clusters=stacked["match_id"].nunique(),
                    **contrast(models[field], {f"pre_{threshold}": 1, "pre_50": -1}, scale),
                ))
    differences = pd.DataFrame(direct)
    differences["p_holm_eight_contrasts"] = differences.groupby("specification")[
        "p_value"
    ].transform(lambda p: multipletests(p, method="holm")[1])
    differences.to_csv(output / "placebo_gap_differences.csv", index=False)
    return table


def compare_chases(sample, output):
    chases = sample.loc[sample["chase_status"].eq("eligible")].copy()
    urgency = chases["required_run_rate_before"]
    if chases.empty or not np.isfinite(urgency).all() or urgency.le(0).any():
        raise ValueError("Eligible chases need finite positive pre-delivery required run rates.")
    center = urgency.median()
    design = design_for(chases, adjusted=True)
    design["required_rate_centered"] = urgency - center
    design["pre_x_required_rate"] = chases["pre"] * design["required_rate_centered"]
    models = fit_outcomes(chases, design)

    interaction = summary_rows(
        chases, models, "Required run rate interaction", {"pre_x_required_rate": 1},
        "Change in the 44-49 minus 50-55 gap", " per +1 run per over required",
    )
    interaction["required_rate_center"] = center
    interaction.to_csv(output / "chase_urgency_interaction.csv", index=False)

    # Display the overlapping central ranges so the chart avoids extrapolating to rare extremes.
    quantiles = chases.groupby("pre")["required_run_rate_before"].quantile([0.10, 0.90]).unstack()
    low, high = quantiles[0.10].max(), quantiles[0.90].min()
    if low >= high:
        raise ValueError("The two chase groups have no overlapping central range of required rates.")
    rates = np.linspace(low, high, 81)
    curves = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (field, (label, unit, scale)) in zip(axes.flat, OUTCOMES.items()):
        records = [
            dict(outcome=label, unit=unit, required_run_rate=rate,
                 **contrast(models[field], {"pre": 1, "pre_x_required_rate": rate - center}, scale))
            for rate in rates
        ]
        curve = pd.DataFrame(records)
        curves.append(curve)
        ax.plot(rates, curve["estimate"], color="#1f6384", linewidth=2)
        ax.fill_between(rates, curve["ci_low"], curve["ci_high"], color="#1f6384", alpha=0.17)
        ax.axhline(0, color="#606a76", linestyle=":", linewidth=1)
        ax.set_title(label, loc="left")
        ax.set(xlabel="Required runs per over before the ball", ylabel=unit)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("IPL chase urgency and the adjusted batting gap (44–49 minus 50–55)")
    fig.savefig(output / "chase_urgency.png", dpi=180)
    plt.close(fig)
    pd.concat(curves, ignore_index=True).to_csv(output / "chase_gaps_by_required_rate.csv", index=False)
    return interaction


def compare_robustness(sample, output):
    batter = sample["batter_id"]
    batter_innings = sample.groupby(["match_id", "innings", "batter_id"], sort=False).ngroup()
    paired = sample.loc[sample.groupby(batter_innings)["pre"].transform("nunique").eq(2)]
    within_batter = fit_outcomes(sample, design_for(sample, adjusted=True),
                                 fixed_effects=batter, absorbed_parameters=batter.nunique())
    within_innings = fit_outcomes(paired, design_for(paired, adjusted=True),
                                  fixed_effects=batter_innings.loc[paired.index])
    results = pd.concat([
        summary_rows(sample, within_batter, "Batter fixed effects with over and wickets",
                     {"pre": 1}, "44-49 minus 50-55"),
        summary_rows(paired, within_innings, "Batter-innings fixed effects with over and wickets",
                     {"pre": 1}, "44-49 minus 50-55"),
    ], ignore_index=True)
    results.to_csv(output / "robustness.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deliveries", type=Path, default=PROJECT / "ipl_deliveries.csv.gz")
    parser.add_argument("--output", type=Path, default=PROJECT / "final_results")
    args = parser.parse_args()
    regulation, sample = load_deliveries(args.deliveries)
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as temporary:
        staging = Path(temporary)
        comparison = compare_all_innings(sample, staging)
        placebos = compare_placebos(regulation, comparison, staging)
        urgency = compare_chases(sample, staging)
        robustness = compare_robustness(sample, staging)
        paper = run_paper_methods(regulation, staging)
        for artifact in staging.iterdir():
            artifact.replace(args.output / artifact.name)

    for title, frame, column in [
        ("IPL batting across the half-century | raw gaps", comparison.iloc[:4], "outcome"),
        ("IPL batting across the half-century | adjusted gaps", comparison.iloc[4:], "outcome"),
        ("IPL chase urgency interaction", urgency, "outcome"),
        ("IPL sensitivity | same batter", robustness.iloc[:4], "outcome"),
        ("IPL sensitivity | same batter and innings", robustness.iloc[4:], "outcome"),
        ("Gauriot and Page half-century methods on IPL data", paper, "method"),
    ]:
        print(f"\n{title}")
        print(frame[[column, "unit", "estimate", "ci_low", "ci_high", "p_value"]].to_string(
            index=False, float_format=lambda value: f"{value:.4g}"
        ))
    print("\nPlacebo checks at 40 and 60 | same score windows and models as at 50")
    for specification in placebos["specification"].unique():
        print(f"\n{specification}")
        subset = placebos.loc[placebos["specification"].eq(specification)]
        print(subset[["threshold", "outcome", "estimate", "ci_low", "ci_high", "p_value"]].to_string(
            index=False, float_format=lambda value: f"{value:.4g}"
        ))
    print(f"\nResults and charts saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
