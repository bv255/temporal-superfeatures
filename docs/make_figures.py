"""
Generates the paper's figures (primary effect bootstrap, fold effects, seed distribution,
operator frequency, predictor comparison) for ANY completed matched-seed run pair, not just
seeds 10/11/12 - the
previous version of this script was hardcoded to that one seed set and to the pre-2026-08-24
'ga_rank_ic_seed{N}' directory-naming convention, so it silently couldn't run against the newer
(and now headline) seed100-109 full pre-registered set without hand-editing.

Usage (run from docs/, matching this file's own relative sys.path insert below):
    python3 make_figures.py --seeds 100 101 102 103 104 105 106 107 108 109
    python3 make_figures.py --seeds 10 11 12 --fitness-metric rank_ic   # legacy pre-08-24 runs

Requires `research/compare_ga_runs.py --seeds ...` (same --seeds/--fitness-metric/--base) to have
already been run for this seed set - this script reads its output CSVs
(research/comparison_outputs/seeds_<...>/) rather than re-deriving the same numbers a second way,
so a figure and its companion table are always built from the identical computation.

Color story (consistent across all 4 figures, from the dataviz skill's reference palette):
  Temporal-OFF = categorical slot 1 (blue #2a78d6), Temporal-ON = categorical slot 2 (orange
  #eb6834) - this adjacent pair is pre-validated (CVD dE 9.1 light) in references/palette.md, so
  it's used as-is rather than re-run through the validator.
"""
import sys
import os
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_DIR = "/home/bvail/temporal-superfeatures/research"
OUT_DIR = "/home/bvail/temporal-superfeatures/docs/figures"
sys.path.insert(0, f"{RESEARCH_DIR}/src")
sys.path.insert(0, RESEARCH_DIR)
import compare_ga_runs as cgr

# run_output_dir()'s own existence-check fallback (added 2026-08-25) resolves its candidate paths
# relative to cwd, per its "callers always run this from research/" convention - chdir here so
# that holds even though this script's own convention is to be run from docs/.
os.chdir(RESEARCH_DIR)

# ---- palette (references/palette.md, categorical slots 1-2, light mode) ----
BLUE = "#2a78d6"    # Temporal-OFF
ORANGE = "#eb6834"  # Temporal-ON
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 11,
})


def style_axes(ax, hide_top_right=True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)


def load_matched_results(base, seeds, fitness_metric):
    """Same loading/matching logic as compare_ga_runs.py's own main_multi_seed, so the folds and
    (fold, seed) records feeding these figures are identical to what built the companion CSVs."""
    temporal_results_by_seed, no_temporal_results_by_seed, per_seed_matched = {}, {}, {}
    for seed in seeds:
        t_dir = cgr.run_output_dir(base, no_temporal=False, fitness_metric=fitness_metric, seed=seed)
        n_dir = cgr.run_output_dir(base, no_temporal=True, fitness_metric=fitness_metric, seed=seed)
        t_results = cgr.load_final_test_fold_results(f"{RESEARCH_DIR}/{t_dir}")
        n_results = cgr.load_final_test_fold_results(f"{RESEARCH_DIR}/{n_dir}")
        temporal_results_by_seed[seed] = t_results
        no_temporal_results_by_seed[seed] = n_results
        per_seed_matched[seed] = set(t_results) & set(n_results)
    matched_folds = sorted(set.intersection(*per_seed_matched.values())) if per_seed_matched else []
    return temporal_results_by_seed, no_temporal_results_by_seed, matched_folds


# =================================================================================================
# fig:primary_temporal_effect - bootstrap distribution of the seed-averaged pooled ON-OFF delta
# =================================================================================================
def make_primary_effect_figure(temporal_by_seed, no_temporal_by_seed, matched_folds, seeds, suffix):
    fold_month_deltas = cgr.seed_averaged_fold_month_deltas(
        temporal_by_seed, no_temporal_by_seed, "winner_monthly_ic", matched_folds, seeds)

    # Reproduce block_bootstrap_ic_delta's internals to get the raw replicate array (the function
    # itself only returns summary stats, which are already in primary_comparison_seed_averaged.csv
    # - this is the one figure that genuinely needs the full replicate distribution, not just its
    # summary, so it can't be built from that CSV alone).
    block_length, n_resamples, seed = 3, 3000, 42
    fold_names = list(fold_month_deltas.keys())
    all_deltas = [d for deltas in fold_month_deltas.values() for d in deltas]
    observed_delta = float(np.mean(all_deltas))
    rng = random.Random(seed)
    boot_deltas = []
    for _ in range(n_resamples):
        pooled = []
        for fold_name in fold_names:
            pooled.extend(cgr._moving_block_resample(fold_month_deltas[fold_name], block_length, rng))
        boot_deltas.append(float(np.mean(pooled)))
    boot_deltas = np.array(boot_deltas)
    lower_bound_95 = float(2 * observed_delta - np.percentile(boot_deltas, 95))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    ax.hist(boot_deltas, bins=40, color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=0.3)

    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="-")
    ax.axvline(observed_delta, color=ORANGE, linewidth=2)
    ax.axvline(lower_bound_95, color=INK_SECONDARY, linewidth=1.5, linestyle="--")

    ymax = ax.get_ylim()[1]
    ax.annotate(f"observed $\\hat\\delta$ = {observed_delta:.4f}", xy=(observed_delta, ymax * 0.96),
                xytext=(6, 0), textcoords="offset points", color=ORANGE, fontsize=10, va="top")
    ax.annotate(f"lower 95% bound = {lower_bound_95:.4f}", xy=(lower_bound_95, ymax * 0.86),
                xytext=(6, 0), textcoords="offset points", color=INK_SECONDARY, fontsize=10, va="top")
    ax.annotate("0 (null)", xy=(0, ymax * 0.76), xytext=(6, 0), textcoords="offset points",
                color=INK_MUTED, fontsize=10, va="top")

    ax.set_xlabel("Bootstrap replicate of pooled ON $-$ OFF monthly Rank IC delta")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Held-out temporal-ON minus temporal-OFF Rank IC effect\n"
                  f"(one-sided block bootstrap, seed-averaged over {len(seeds)} seeds)",
                  fontsize=12, color=INK_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/primary_temporal_effect_{suffix}.pdf")
    fig.savefig(f"{OUT_DIR}/primary_temporal_effect_{suffix}.png")
    plt.close(fig)
    print(f"wrote primary_temporal_effect_{suffix}.{{pdf,png}}")


# =================================================================================================
# fig:fold_effects - per-fold ON-OFF Rank IC delta, seed-averaged, colored by which arm it favors
# =================================================================================================
def make_fold_effects_figure(comparison_output_dir, seeds, suffix):
    fold_df = pd.read_csv(f"{comparison_output_dir}/seed_averaged_fold_rank_ic.csv")
    fold_df = fold_df[fold_df["fold_name"] != "AGGREGATE (pooled)"].reset_index(drop=True)

    labels = [f"Fold {i+1}\n({int(row.eval_year)})" for i, row in enumerate(fold_df.itertuples())]
    deltas = fold_df["rank_ic_delta_on_minus_off"].tolist()
    colors = [ORANGE if d > 0 else BLUE for d in deltas]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    x = np.arange(len(fold_df))
    ax.bar(x, deltas, color=colors, width=0.6, edgecolor="none", zorder=3)

    for xi, d in zip(x, deltas):
        va = "bottom" if d >= 0 else "top"
        offset = 0.0015 if d >= 0 else -0.0015
        ax.annotate(f"{d:+.4f}", xy=(xi, d + offset), ha="center", va=va,
                    fontsize=9.5, color=INK_PRIMARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Rank IC delta (ON $-$ OFF)")
    ax.set_title("Temporal-ON minus temporal-OFF Rank IC effect by final-test fold\n"
                  f"(seed-averaged over {len(seeds)} seeds)",
                  fontsize=12, color=INK_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["Favors temporal-ON", "Favors temporal-OFF"],
              loc="upper right", frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fold_effects_{suffix}.pdf")
    fig.savefig(f"{OUT_DIR}/fold_effects_{suffix}.png")
    plt.close(fig)
    print(f"wrote fold_effects_{suffix}.{{pdf,png}}")


# =================================================================================================
# fig:seed_distribution - per-seed arm-level mean Rank IC (fold-pooled), OFF vs ON
# =================================================================================================
def make_seed_distribution_figure(comparison_output_dir, seeds, suffix):
    pooled_df = pd.read_csv(f"{comparison_output_dir}/seed_level_rank_ic.csv")
    off_by_seed = pooled_df[pooled_df["arm"] == "temporal_off"].set_index("seed")["rank_ic"].to_dict()
    on_by_seed = pooled_df[pooled_df["arm"] == "temporal_on"].set_index("seed")["rank_ic"].to_dict()

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    x_off, x_on = 0.0, 1.0
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=len(seeds))

    off_vals = [off_by_seed[s] for s in seeds]
    on_vals = [on_by_seed[s] for s in seeds]

    ax.scatter(np.full(len(seeds), x_off) + jitter, off_vals, s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=0.8, zorder=3, label="Temporal-OFF")
    ax.scatter(np.full(len(seeds), x_on) + jitter, on_vals, s=70, color=ORANGE,
               edgecolor=SURFACE, linewidth=0.8, zorder=3, label="Temporal-ON")

    show_labels = len(seeds) <= 6  # per-point seed labels get unreadable beyond a handful
    if show_labels:
        for xi, s in zip(jitter, seeds):
            ax.annotate(f"seed {s}", xy=(x_off + xi, off_by_seed[s]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=8, color=INK_MUTED)
            ax.annotate(f"seed {s}", xy=(x_on + xi, on_by_seed[s]), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_MUTED)

    ax.hlines(np.mean(off_vals), x_off - 0.18, x_off + 0.18, color=BLUE, linewidth=2.5, zorder=4)
    ax.hlines(np.mean(on_vals), x_on - 0.18, x_on + 0.18, color=ORANGE, linewidth=2.5, zorder=4)

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([x_off, x_on])
    ax.set_xticklabels(["Temporal-OFF", "Temporal-ON"])
    ax.set_ylabel("Held-out Rank IC (fold-pooled, per seed)")
    ax.set_title("Distribution of held-out Rank IC across matched GA seeds\n"
                  f"(seeds {', '.join(str(s) for s in seeds)}; horizontal bar = mean)",
                  fontsize=12, color=INK_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/seed_distribution_{suffix}.pdf")
    fig.savefig(f"{OUT_DIR}/seed_distribution_{suffix}.png")
    plt.close(fig)
    print(f"wrote seed_distribution_{suffix}.{{pdf,png}}")


# =================================================================================================
# fig:operator_frequency - temporal operator usage among temporal-ON winners
# =================================================================================================
def make_operator_frequency_figure(comparison_output_dir, suffix):
    freq_df = pd.read_csv(f"{comparison_output_dir}/pooled_operator_frequency.csv")
    n_winners = int(freq_df["n_winners_total"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    x = np.arange(len(freq_df))
    ax.bar(x, freq_df["n_winners_using"], color=ORANGE, width=0.55, zorder=3)
    for xi, v, p in zip(x, freq_df["n_winners_using"], freq_df["pct_of_winners"]):
        ax.annotate(f"{v} ({p:.1f}%)", xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(freq_df["operator"])
    ax.set_ylabel(f"Number of winners (of {n_winners})")
    ax.set_ylim(0, freq_df["n_winners_using"].max() + 2)
    ax.set_title("Frequency of temporal operators among temporal-ON winning expressions\n"
                 f"({n_winners} winners pooled across matched folds and seeds)",
                 fontsize=12, color=INK_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/operator_frequency_{suffix}.pdf")
    fig.savefig(f"{OUT_DIR}/operator_frequency_{suffix}.png")
    plt.close(fig)
    print(f"wrote operator_frequency_{suffix}.{{pdf,png}}")


# =================================================================================================
# fig:predictor_comparison - GA vs. Fixed/Raw/Random, ON vs OFF, all on one shared matched date
# set per arm (ESTIMAND III N-way, predictor_comparison_matched.csv)
# =================================================================================================
def make_predictor_comparison_figure(comparison_output_dir, seeds, suffix):
    df = pd.read_csv(f"{comparison_output_dir}/predictor_comparison_matched.csv").set_index("arm")
    predictors = [c for c in df.columns if c != "n_months"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    x = np.arange(len(predictors))
    width = 0.35
    off_vals = df.loc["temporal_off", predictors].astype(float)
    on_vals = df.loc["temporal_on", predictors].astype(float)

    bars_off = ax.bar(x - width / 2, off_vals, width, color=BLUE, label="Temporal-OFF", zorder=3)
    bars_on = ax.bar(x + width / 2, on_vals, width, color=ORANGE, label="Temporal-ON", zorder=3)
    for bars in (bars_off, bars_on):
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 3 if h >= 0 else -3
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, offset),
                        textcoords="offset points", ha="center", va=va, fontsize=8.5, color=INK_PRIMARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(predictors)
    ax.set_ylabel("Rank IC (predictor-matched date set, per arm)")
    ax.set_title("GA winner vs. within-arm baselines, temporal-ON vs. temporal-OFF\n"
                  f"(seed-averaged over {len(seeds)} seeds; each arm on its own shared matched date set)",
                  fontsize=12, color=INK_PRIMARY, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/predictor_comparison_{suffix}.pdf")
    fig.savefig(f"{OUT_DIR}/predictor_comparison_{suffix}.png")
    plt.close(fig)
    print(f"wrote predictor_comparison_{suffix}.{{pdf,png}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                         help="matches compare_ga_runs.py's --fast (reads ga_fast/... comparison output)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                         help="the matched seed set to plot - must match a prior "
                              "'compare_ga_runs.py --seeds ...' invocation with the same seeds/"
                              "--fitness-metric, whose comparison_outputs/seeds_.../ this reads from")
    parser.add_argument("--fitness-metric", choices=["rmse", "rank_ic"], default=None,
                         help="pass 'rank_ic' only for a LEGACY pre-2026-08-24 run (e.g. seeds "
                              "10-13) - current runs (seed100+) use rank_ic implicitly with no "
                              "directory suffix, so this should stay unset for them. See "
                              "compare_ga_runs.py's run_output_dir docstring.")
    args = parser.parse_args()

    base = "ga_fast" if args.fast else "ga"
    label = cgr.comparison_output_label(fast=args.fast, fitness_metric=args.fitness_metric, seeds=args.seeds)
    comparison_output_dir = f"{RESEARCH_DIR}/comparison_outputs/{label}"
    if not os.path.isdir(comparison_output_dir):
        parser.error(f"{comparison_output_dir} doesn't exist - run "
                      f"'compare_ga_runs.py {'--fast ' if args.fast else ''}--seeds "
                      f"{' '.join(str(s) for s in args.seeds)}"
                      f"{' --fitness-metric ' + args.fitness_metric if args.fitness_metric else ''}' "
                      f"first to produce the tables these figures are built from.")

    temporal_by_seed, no_temporal_by_seed, matched_folds = load_matched_results(
        base, args.seeds, args.fitness_metric)
    if not matched_folds:
        parser.error(f"no final-test folds matched across all of seeds {args.seeds} - nothing to plot.")

    suffix = label
    make_primary_effect_figure(temporal_by_seed, no_temporal_by_seed, matched_folds, args.seeds, suffix)
    make_fold_effects_figure(comparison_output_dir, args.seeds, suffix)
    make_seed_distribution_figure(comparison_output_dir, args.seeds, suffix)
    make_operator_frequency_figure(comparison_output_dir, suffix)
    make_predictor_comparison_figure(comparison_output_dir, args.seeds, suffix)
