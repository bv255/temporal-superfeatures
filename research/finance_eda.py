"""
Finance-sector descriptive EDA for the temporal-superfeatures thesis.

Purpose: produce and save the descriptive statistics/plots a thesis chapter needs to describe
the Finance-sector dataset (coverage through time, market cap, feature missingness/variation/
distributions/correlation, the monthly-return target, fundamental reporting cadence, and
point-in-time filing delay) - NOT to make any feature-selection or modelling decision. Nothing
computed here is fed back into `run_preprocessing.py`/`run_ga.py`.

Run from `research/` (same cwd convention as `run_preprocessing.py`/`run_ga.py` - relative
output paths and the editable `superfeatures` install both depend on it):
    ~/pyspark-venv/bin/python3 finance_eda.py

ASSUMPTIONS made because the existing pipeline code doesn't make them explicit (see CLAUDE.md /
`superfeatures.preprocessing.utils.Utils` / `superfeatures.preprocessing.pipeline` for the
pipeline this script reads from):

  1. "Raw candidate fundamental feature set" = the columns of `cleaned_fundamentals`
     (post `Utils.drop_non_numeric_columns`, pre `interpolate_missing_values`/pre any per-fold
     missingness or static-feature filter) - i.e. exactly `candidate_feature_columns` as
     `run_preprocessing.py`'s driver computes it, before any fold ever sees the data. Sections
     5-8 (missingness/variation/distributions/correlation) describe this set.
  2. `pit_market_value` (point-in-time market cap) is computed internally by
     `Utils.get_fundamental_data()`/`get_price_data()` but filtered-on-then-dropped before
     either function returns - never an output column. This script recomputes it the same way
     the pipeline does, by calling `Utils._asof_carry_forward` (the pipeline's own helper, not a
     new implementation) on the fundamentals report grain (one value per report date).
  3. Because `get_fundamental_data()` already filters `pit_market_value > market_value_threshold`
     before returning, every fundamentals row analysed here already satisfies the threshold -
     "eligible companies above threshold through time" is therefore identical to the raw
     monthly/yearly company-coverage counts already in section 2. Section 4 instead
     characterises the level/spread of market cap *among* that already-eligible population,
     which is still informative (e.g. whether companies cluster near the threshold or are
     broadly distributed above it).
  4. "Fundamental observations" = one row per (fsym, date) in `Utils.get_fundamental_data()`'s
     output (one row per company per fiscal-period-end report), not the later monthly as-of-
     expanded grain built by `feature_selection_dataset`.
  5. Filing-date validity (section 11): a report is "invalid"/uses the pipeline's fallback lag
     when `greatest(ff_source_is_date, ff_source_bs_date, ff_source_cf_date)` is null OR implies
     a negative lag versus `date` - exactly the condition `Utils.get_fundamental_data()` uses
     internally to decide whether to trust the observed source dates or fall back to
     `date + fund_report_fallback_lag_days`. That boolean isn't exposed as an output column, so
     it's recomputed here from the same source columns (`ff_source_is_date`/`ff_source_bs_date`/
     `ff_source_cf_date`), not re-derived logic.
  6. Headline filing-delay statistics (mean/median/percentiles/threshold coverage) are computed
     over reports with a *valid* (non-fallback, non-negative) observed filing date only. Rows
     needing the fallback (missing source dates) are reported separately in the summary but
     excluded from the headline distribution; genuinely negative raw source-implied delays are
     written to `invalid_filing_delays.csv` and excluded entirely, per the task's "do not
     silently include them" instruction. Because the pipeline's own fallback always assigns a
     non-negative lag, no row in `fund_report_date - date` can itself be negative - the
     "impossible values" worth flagging are in the raw *source* dates, not the final column.
  7. Section 12's per-stage feature/row counts run `Utils.drop_na_features`/
     `Utils.remove_static_features` on the *whole* Finance history (not per-fold, train-only, as
     the real walk-forward pipeline correctly does) purely to produce descriptive stage counts.
     These calls are separate from, and never feed into, `run_fold()`'s own per-fold fits.
  8. "Company" for per-company statistics (sections 2, 3, 10) = `fsym` (security id), matching
     every per-company `Window`/`groupBy` in `Utils` (e.g. `align_data`, `remove_static_features`,
     `feature_selection_dataset` all partition by `fsym`, not `factset_entity_id`). Section 1
     separately reports unique `factset_entity_id` (company) vs. `fsym` (security) counts since
     they can differ (multiple share classes per entity).
  9. The monthly-return target (section 9) uses `Utils.calculate_monthly_returns()`'s output
     directly - already sanity-filtered for impossible/implausible returns by the pipeline
     itself, one row per (fsym, year, month), Finance-only via the same SQL-pushed
     `sector_filter` used everywhere else in `Utils`.
 10. Reporting cadence (section 10) uses the fundamentals' own period-end `date` column (how
     often a company files), while filing delay (section 11) is measured off `fund_report_date`
     (when a report becomes point-in-time known) - two different questions using two different
     date columns, matching the pipeline's own distinction between the two.
 11. "Eligible Finance companies per month/year" (section 2) is read off the walk-forward base
     checkpoint (`config.walk_forward_base_path`, i.e. `Utils.feature_selection_dataset`'s
     output) when it already exists on HDFS, since that checkpoint is exactly the pipeline's own
     point-in-time-eligible monthly panel - reusing it is both more correct (identical
     point-in-time definition) and cheaper than recomputing the as-of join. Falls back to
     building it fresh via `utils.feature_selection_dataset` if the checkpoint is absent.
 12. Company-level "years covered" (section 3) counts distinct calendar years with >=1
     fundamental report, not `(max_year - min_year + 1)`.

IMPLEMENTATION NOTES

- All Spark aggregations touching many columns are chunked (<=30 columns per `.select`/`.agg`),
  matching the mitigation already used throughout `Utils`/`ConsensusFeatureSelector` for the
  same reason (Spark whole-stage codegen can stall well before Java's 64KB per-method bytecode
  cap is hit).
- Histograms are built via `F.width_bucket` (pure Spark SQL aggregation) rather than collecting
  raw values - never `.rdd.flatMap(lambda ...)`, which needs a Python worker on the YARN
  executors and breaks (`/home/bvail` isn't NFS-shared to `skrzat*` - see CLAUDE.md).
- The only `.toPandas()` calls are on already-small, aggregated, or explicitly company-sampled
  frames (never the full fundamentals/price tables) - this script's own sampling, independent of
  `ConsensusFeatureSelector` (which no longer samples companies at all - see its class
  docstring).
- This script only ever reads from Hive/HDFS and writes small CSV/JSON/PNG files to local disk
  under `outputs/eda/finance/` - it never writes to HDFS and never touches source tables.
"""
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from superfeatures.config import FULL_CONFIG
from superfeatures.preprocessing import Utils

from run_preprocessing import _build_spark_session

# ---------------------------------------------------------------------- output layout ----
OUTPUT_ROOT = "outputs/eda/finance"
DIR_SCOPE = f"{OUTPUT_ROOT}/dataset_scope"
DIR_COVERAGE = f"{OUTPUT_ROOT}/company_coverage"
DIR_OBS_COVERAGE = f"{OUTPUT_ROOT}/observation_coverage"
DIR_MARKET_CAP = f"{OUTPUT_ROOT}/market_cap"
DIR_MISSINGNESS = f"{OUTPUT_ROOT}/feature_missingness"
DIR_VARIATION = f"{OUTPUT_ROOT}/feature_variation"
DIR_DISTRIBUTIONS = f"{OUTPUT_ROOT}/feature_distributions"
DIR_CORRELATION = f"{OUTPUT_ROOT}/feature_correlation"
DIR_RETURNS = f"{OUTPUT_ROOT}/monthly_return_target"
DIR_REPORTING_FREQ = f"{OUTPUT_ROOT}/reporting_frequency"
DIR_FILING_DELAY = f"{OUTPUT_ROOT}/filing_delay"
DIR_AUDIT = f"{OUTPUT_ROOT}/pipeline_audit"

ALL_DIRS = [
    OUTPUT_ROOT, DIR_SCOPE, DIR_COVERAGE, DIR_OBS_COVERAGE, DIR_MARKET_CAP,
    DIR_MISSINGNESS, DIR_VARIATION, DIR_DISTRIBUTIONS, DIR_CORRELATION,
    DIR_RETURNS, DIR_REPORTING_FREQ, DIR_FILING_DELAY, DIR_AUDIT,
]

PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
CHUNK_SIZE = 30          # matches Utils'/ConsensusFeatureSelector's wide-aggregation mitigation
STATS_CHUNK_SIZE = 12    # percentile_approx per column is heavier - smaller chunk


def ensure_output_dirs():
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  wrote {path}")


def save_df(df: pd.DataFrame, path, index=False):
    df.to_csv(path, index=index)
    print(f"  wrote {path} ({len(df)} rows)")


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


# ======================================================================= Spark helpers ====

def chunked_missing_counts(df, cols, chunk_size=CHUNK_SIZE):
    """Null count per column, chunked to avoid a single wide `.select()`. Returns
    (missing_counts: dict[col, int], total_rows: int)."""
    total_rows = df.count()
    counts = {}
    for i in range(0, len(cols), chunk_size):
        chunk = cols[i:i + chunk_size]
        row = df.select(
            [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in chunk]
        ).collect()[0].asDict()
        counts.update(row)
    return counts, total_rows


def chunked_descriptive_stats(df, cols, chunk_size=STATS_CHUNK_SIZE):
    """count/mean/std/min/percentiles/max/skewness per numeric column, chunked."""
    rows = []
    for i in range(0, len(cols), chunk_size):
        chunk = cols[i:i + chunk_size]
        aggs = []
        for c in chunk:
            safe = c.replace('`', '')
            aggs += [
                F.count(F.col(c)).alias(f"{c}__count"),
                F.mean(c).alias(f"{c}__mean"),
                F.stddev(c).alias(f"{c}__std"),
                F.min(c).alias(f"{c}__min"),
                F.max(c).alias(f"{c}__max"),
                F.skewness(c).alias(f"{c}__skew"),
                F.expr(f"percentile_approx(`{safe}`, array({','.join(str(p) for p in PERCENTILES)}), 10000)")
                 .alias(f"{c}__pcts"),
            ]
        row = df.select(*aggs).collect()[0].asDict()
        for c in chunk:
            pcts = row.get(f"{c}__pcts") or [None] * len(PERCENTILES)
            rows.append({
                'feature': c,
                'count': row.get(f"{c}__count"),
                'mean': row.get(f"{c}__mean"),
                'std': row.get(f"{c}__std"),
                'min': row.get(f"{c}__min"),
                'p1': pcts[0], 'p5': pcts[1], 'p25': pcts[2], 'median': pcts[3],
                'p75': pcts[4], 'p95': pcts[5], 'p99': pcts[6],
                'max': row.get(f"{c}__max"),
                'skewness': row.get(f"{c}__skew"),
            })
    return pd.DataFrame(rows)


def flag_any_nonnull(df, cols, entity_col, chunk_size=CHUNK_SIZE):
    """Count of distinct `entity_col` values having >=1 non-null value across `cols`, computed
    via chunked F.greatest (no per-row Python UDF, no full collect)."""
    flagged = df
    chunk_flag_cols = []
    for i in range(0, len(cols), chunk_size):
        chunk = cols[i:i + chunk_size]
        flag_name = f"_any_nonnull_chunk_{i}"
        flagged = flagged.withColumn(
            flag_name, F.greatest(*[F.col(c).isNotNull().cast('int') for c in chunk])
        )
        chunk_flag_cols.append(flag_name)
    flagged = flagged.withColumn('_any_nonnull', F.greatest(*[F.col(c) for c in chunk_flag_cols]))
    return flagged.filter(F.col('_any_nonnull') == 1).select(entity_col).distinct().count()


def spark_histogram(df, col_name, num_bins=50, lo=None, hi=None):
    """Histogram bin counts via F.width_bucket (pure Spark SQL aggregation - no RDD/lambda, so
    it's safe on YARN executors that don't have the driver's Python venv - see module docstring).
    Values below `lo`/above `hi` land in an explicit tail bucket rather than being dropped."""
    if lo is None or hi is None:
        row = df.select(F.min(col_name).alias('lo'), F.max(col_name).alias('hi')).collect()[0]
        lo = row['lo'] if lo is None else lo
        hi = row['hi'] if hi is None else hi
    if lo is None or hi is None or lo >= hi:
        return pd.DataFrame(columns=['bin_index', 'bin_left', 'bin_right', 'count'])

    bucketed = (
        df.select(F.col(col_name).cast('double').alias('v'))
        .where(F.col('v').isNotNull() & ~F.isnan('v'))
        .withColumn('bucket', F.width_bucket(F.col('v'), F.lit(lo), F.lit(hi), F.lit(num_bins)))
    )
    counts = {int(r['bucket']): r['count'] for r in bucketed.groupBy('bucket').count().collect()}
    width = (hi - lo) / num_bins
    rows = []
    for b in range(0, num_bins + 2):  # bucket 0 = below lo, num_bins+1 = above hi
        rows.append({
            'bin_index': b,
            'bin_left': lo + (b - 1) * width,
            'bin_right': lo + b * width,
            'count': counts.get(b, 0),
        })
    return pd.DataFrame(rows)


def sample_companies(df, fsym_col, seed, fraction=0.2, max_companies=400):
    """Deterministic company sample for the toPandas()-based correlation step - same
    seeded-random-sample idiom `ConsensusFeatureSelector` already uses, just with a larger
    fraction/cap since this is a one-off descriptive pass, not a per-fold refit."""
    all_companies = [r[fsym_col] for r in df.select(fsym_col).distinct().collect()]
    n = min(len(all_companies), max_companies, max(1, int(len(all_companies) * fraction)))
    rng = random.Random(seed)
    return sorted(rng.sample(all_companies, n))


# ============================================================================ sections ====

def section_dataset_scope(df_fundamentals, df_prices, monthly_returns, candidate_feature_columns):
    """1. DATASET SCOPE - headline counts a reader needs before anything else: how much data,
    over what period, for how many companies."""
    print("[1] dataset scope...")
    fund_dates = df_fundamentals.agg(
        F.min('date').alias('min_date'), F.max('date').alias('max_date')
    ).collect()[0]
    price_dates = df_prices.agg(
        F.min('p_date').alias('min_date'), F.max('p_date').alias('max_date')
    ).collect()[0]

    n_fund_obs = df_fundamentals.count()
    n_price_obs = df_prices.count()
    n_return_obs = monthly_returns.count()
    n_companies = df_fundamentals.select('factset_entity_id').distinct().count()
    n_securities = df_fundamentals.select('fsym').distinct().count()
    n_usable_companies = flag_any_nonnull(df_fundamentals, candidate_feature_columns, 'factset_entity_id')

    reports_per_company = (
        df_fundamentals.groupBy('fsym').agg(F.count('*').alias('n_reports'))
        .select(
            F.expr('percentile_approx(n_reports, 0.5, 10000)').alias('median'),
            F.min('n_reports').alias('min'),
            F.max('n_reports').alias('max'),
        ).collect()[0]
    )

    companies_per_month = (
        df_fundamentals.groupBy(F.year('date').alias('y'), F.month('date').alias('m'))
        .agg(F.countDistinct('fsym').alias('n_companies'))
    )
    companies_per_month_stats = companies_per_month.select(
        F.expr('percentile_approx(n_companies, 0.5, 10000)').alias('median'),
        F.min('n_companies').alias('min'),
        F.max('n_companies').alias('max'),
    ).collect()[0]

    summary = {
        'sector': 'Finance',
        'earliest_fundamental_date': str(fund_dates['min_date']),
        'latest_fundamental_date': str(fund_dates['max_date']),
        'earliest_price_date': str(price_dates['min_date']),
        'latest_price_date': str(price_dates['max_date']),
        'n_unique_finance_companies': n_companies,
        'n_unique_securities': n_securities,
        'n_raw_fundamental_observations': n_fund_obs,
        'n_price_observations': n_price_obs,
        'n_monthly_return_observations': n_return_obs,
        'n_raw_candidate_fundamental_features': len(candidate_feature_columns),
        'n_companies_with_usable_fundamental_observation': n_usable_companies,
        'fundamental_observations_per_company_median': reports_per_company['median'],
        'fundamental_observations_per_company_min': reports_per_company['min'],
        'fundamental_observations_per_company_max': reports_per_company['max'],
        'finance_companies_per_month_median': companies_per_month_stats['median'],
        'finance_companies_per_month_min': companies_per_month_stats['min'],
        'finance_companies_per_month_max': companies_per_month_stats['max'],
    }
    save_json(summary, f"{DIR_SCOPE}/dataset_summary.json")
    save_df(pd.DataFrame([summary]), f"{DIR_SCOPE}/dataset_summary.csv")
    return summary


def section_company_coverage(df_fundamentals, monthly_returns, eligible_monthly_df):
    """2. COMPANY COVERAGE THROUGH TIME - demonstrates how the usable Finance universe (and raw
    data volume feeding it) grows/shrinks over the sample period; motivates the walk-forward
    fold boundaries used downstream."""
    print("[2] company coverage through time...")

    fund_monthly = (
        df_fundamentals.groupBy(F.year('date').alias('year'), F.month('date').alias('month'))
        .agg(F.countDistinct('fsym').alias('n_reporting_companies'), F.count('*').alias('n_fundamental_observations'))
    )
    return_monthly = (
        monthly_returns.groupBy(F.col('year'), F.col('month'))
        .agg(F.count('*').alias('n_return_observations'))
    )
    eligible_monthly = (
        eligible_monthly_df.groupBy(F.year('target_date').alias('year'), F.month('target_date').alias('month'))
        .agg(F.countDistinct('fsym').alias('n_eligible_companies'))
    )

    monthly = (
        eligible_monthly.join(fund_monthly, on=['year', 'month'], how='outer')
        .join(return_monthly, on=['year', 'month'], how='outer')
        .orderBy('year', 'month')
        .fillna(0)
        .toPandas()
    )
    save_df(monthly, f"{DIR_COVERAGE}/company_coverage_monthly.csv")

    yearly = monthly.groupby('year', as_index=False).agg(
        n_eligible_companies=('n_eligible_companies', 'max'),
        n_reporting_companies=('n_reporting_companies', 'max'),
        n_fundamental_observations=('n_fundamental_observations', 'sum'),
        n_return_observations=('n_return_observations', 'sum'),
    )
    save_df(yearly, f"{DIR_COVERAGE}/company_coverage_yearly.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(monthly['year'].astype(str) + '-' + monthly['month'].astype(str) + '-01')
    ax.plot(x, monthly['n_eligible_companies'], label='Eligible Finance companies (point-in-time)')
    ax.plot(x, monthly['n_reporting_companies'], label='Companies with a report that month', alpha=0.6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of companies')
    ax.set_title('Finance-sector company coverage through time')
    ax.legend()
    save_fig(f"{DIR_COVERAGE}/finance_companies_over_time.png")

    return monthly, yearly


def section_observation_coverage(df_fundamentals):
    """3. OBSERVATION COVERAGE - per-company reporting history: how long has each company been
    covered, and how regularly, since this bounds how much temporal-operator history is
    available per company."""
    print("[3] per-company observation coverage...")

    from pyspark.sql.window import Window
    w = Window.partitionBy('fsym').orderBy('date')
    with_gaps = df_fundamentals.select('fsym', 'date').withColumn(
        'gap_days', F.datediff('date', F.lag('date', 1).over(w))
    )

    per_company_gap = with_gaps.groupBy('fsym').agg(
        F.expr('percentile_approx(gap_days, 0.5, 10000)').alias('median_days_between_reports')
    )

    per_company_year_counts = (
        df_fundamentals.groupBy('fsym', F.year('date').alias('year')).agg(F.count('*').alias('n_reports'))
    )
    per_company_year_agg = per_company_year_counts.groupBy('fsym').agg(
        F.min('n_reports').alias('min_reports_per_year'),
        F.max('n_reports').alias('max_reports_per_year'),
        F.countDistinct('year').alias('n_years_covered'),
    )

    per_company = (
        df_fundamentals.groupBy('fsym').agg(
            F.min('date').alias('first_fundamental_date'),
            F.max('date').alias('last_fundamental_date'),
            F.count('*').alias('n_fundamental_reports'),
        )
        .join(per_company_gap, on='fsym', how='left')
        .join(per_company_year_agg, on='fsym', how='left')
        .orderBy('fsym')
        .toPandas()
    )
    save_df(per_company, f"{DIR_OBS_COVERAGE}/company_observation_coverage.csv")

    summary = {
        'n_companies': int(len(per_company)),
        'n_fundamental_reports_median': float(per_company['n_fundamental_reports'].median()),
        'n_fundamental_reports_min': int(per_company['n_fundamental_reports'].min()),
        'n_fundamental_reports_max': int(per_company['n_fundamental_reports'].max()),
        'years_covered_median': float(per_company['n_years_covered'].median()),
        'years_covered_min': int(per_company['n_years_covered'].min()),
        'years_covered_max': int(per_company['n_years_covered'].max()),
        'median_days_between_reports_median_across_companies': float(
            per_company['median_days_between_reports'].dropna().median()
        ) if per_company['median_days_between_reports'].notna().any() else None,
        'min_reports_per_year_typical': float(per_company['min_reports_per_year'].median()),
        'max_reports_per_year_typical': float(per_company['max_reports_per_year'].median()),
    }
    save_json(summary, f"{DIR_OBS_COVERAGE}/observation_coverage_summary.json")
    return per_company, summary


def section_market_cap(utils, df_fundamentals):
    """4. MARKET CAPITALISATION - describes the size distribution of the already point-in-time-
    eligible Finance universe (see assumption 3 above for why "eligible above threshold" is
    identical to section 2's coverage counts). Reuses `Utils._asof_carry_forward` - the exact
    helper `get_fundamental_data()` itself uses internally - rather than re-deriving the
    point-in-time price-times-shares logic."""
    print("[4] market capitalisation...")

    # Keyed on 'fsym_id' (the REGIONAL id, from ff_basic_qf via fun.* in get_fundamental_data's
    # own query), not 'fsym' (the security id) - fp_basic_prices.fsym_id is also regional, so the
    # join keys must agree on 'fsym_id'. This is the exact mismatch Utils.get_fundamental_data's
    # own _asof_carry_forward call docstring warns about (joining security id against a
    # regional-keyed table almost never matches, leaving pit_market_value null for ~every row).
    with_price = utils._asof_carry_forward(
        df_fundamentals.select('fsym', 'fsym_id', 'date', 'ff_com_shs_out'), 'fsym_id', 'date',
        "SELECT fsym_id, p_date, p_price FROM fp_basic_prices",
        'fsym_id', 'p_date', 'p_price', '_asof_price',
        max_staleness_days=14,  # same daily price leg/bound as get_fundamental_data's own call
    )
    with_mktcap = with_price.withColumn(
        'pit_market_value', F.col('ff_com_shs_out') * F.col('_asof_price')
    ).select('fsym', 'date', 'pit_market_value').filter(F.col('pit_market_value').isNotNull()).cache()

    stats_row = with_mktcap.select(
        F.count('pit_market_value').alias('count'),
        F.mean('pit_market_value').alias('mean'),
        F.stddev('pit_market_value').alias('std'),
        F.min('pit_market_value').alias('min'),
        F.max('pit_market_value').alias('max'),
        F.expr(f"percentile_approx(pit_market_value, array({','.join(str(p) for p in PERCENTILES)}), 10000)").alias('pcts'),
    ).collect()[0]
    pcts = stats_row['pcts']
    median_val = pcts[3]

    summary_row = {
        'count': stats_row['count'], 'mean': stats_row['mean'], 'median': median_val,
        'std': stats_row['std'],
        'p1': pcts[0], 'p5': pcts[1], 'p25': pcts[2], 'p75': pcts[4], 'p95': pcts[5], 'p99': pcts[6],
        'min': stats_row['min'], 'max': stats_row['max'],
        'note': 'population already restricted to pit_market_value > market_value_threshold by '
                'Utils.get_fundamental_data() at extraction time - see assumption 3',
    }
    save_df(pd.DataFrame([summary_row]), f"{DIR_MARKET_CAP}/market_cap_summary.csv")

    hist = spark_histogram(
        with_mktcap.withColumn('log_mktcap', F.log10('pit_market_value')), 'log_mktcap', num_bins=60
    )
    save_df(hist, f"{DIR_MARKET_CAP}/market_cap_distribution.csv")

    monthly_coverage = (
        with_mktcap.groupBy(F.year('date').alias('year'), F.month('date').alias('month'))
        .agg(F.countDistinct('fsym').alias('n_eligible_companies'))
        .orderBy('year', 'month')
        .toPandas()
    )
    save_df(monthly_coverage, f"{DIR_MARKET_CAP}/market_cap_coverage_monthly.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_hist = hist[(hist['bin_index'] >= 1) & (hist['bin_index'] <= 60)]
    ax.bar(plot_hist['bin_left'], plot_hist['count'], width=(plot_hist['bin_right'] - plot_hist['bin_left']), align='edge')
    ax.set_xlabel('log10(point-in-time market value), $ thousands')
    ax.set_ylabel('Number of fundamental-report observations')
    ax.set_title('Finance-sector point-in-time market cap distribution (log scale)\n(raw statistics saved untransformed)')
    save_fig(f"{DIR_MARKET_CAP}/market_cap_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(monthly_coverage['year'].astype(str) + '-' + monthly_coverage['month'].astype(str) + '-01')
    ax.plot(x, monthly_coverage['n_eligible_companies'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of companies above market-cap threshold')
    ax.set_title('Finance-sector market-cap-eligible companies through time')
    save_fig(f"{DIR_MARKET_CAP}/market_cap_coverage_over_time.png")

    with_mktcap.unpersist()
    return summary_row


def section_feature_missingness(cleaned_fundamentals, candidate_feature_columns):
    """5. FEATURE MISSINGNESS - raw candidate feature set (pre any filter). Motivates the
    per-fold 30%-missingness threshold `Utils.drop_na_features` applies downstream."""
    print("[5] feature missingness...")
    missing_counts, total_rows = chunked_missing_counts(cleaned_fundamentals, candidate_feature_columns)

    rows = []
    for c in candidate_feature_columns:
        n_missing = missing_counts[c]
        rows.append({
            'feature': c,
            'total_observations': total_rows,
            'n_missing': n_missing,
            'n_non_missing': total_rows - n_missing,
            'pct_missing': n_missing / total_rows if total_rows else None,
        })
    missingness_df = pd.DataFrame(rows).sort_values('pct_missing', ascending=False).reset_index(drop=True)
    save_df(missingness_df, f"{DIR_MISSINGNESS}/feature_missingness.csv")

    bands = [
        ('0%', lambda p: p == 0),
        ('>0-10%', lambda p: 0 < p <= 0.10),
        ('>10-20%', lambda p: 0.10 < p <= 0.20),
        ('>20-30%', lambda p: 0.20 < p <= 0.30),
        ('>30-50%', lambda p: 0.30 < p <= 0.50),
        ('>50-75%', lambda p: 0.50 < p <= 0.75),
        ('>75-<100%', lambda p: 0.75 < p < 1.0),
        ('100%', lambda p: p >= 1.0),
    ]
    band_rows = [{'band': name, 'n_features': int(missingness_df['pct_missing'].apply(f).sum())} for name, f in bands]
    bands_df = pd.DataFrame(band_rows)
    save_df(bands_df, f"{DIR_MISSINGNESS}/missingness_bands.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bands_df['band'], bands_df['n_features'])
    ax.set_xlabel('Missingness band')
    ax.set_ylabel('Number of raw candidate features')
    ax.set_title('Finance-sector raw candidate feature missingness distribution')
    plt.xticks(rotation=30, ha='right')
    save_fig(f"{DIR_MISSINGNESS}/feature_missingness_distribution.png")

    return missingness_df, bands_df


def section_feature_variation(utils, cleaned_fundamentals, candidate_feature_columns):
    """6. STATIC / LOW-VARIANCE FEATURES - describes variation without removing anything; flags
    which features WOULD be dropped by `Utils.remove_static_features`'s own rule (majority of a
    50-company sample having per-company variance < 0.001) so a reader can see how big that cut
    would be, without this script performing the cut."""
    print("[6] feature variation / static-feature flagging...")

    unique_counts_rows = []
    for i in range(0, len(candidate_feature_columns), CHUNK_SIZE):
        chunk = candidate_feature_columns[i:i + CHUNK_SIZE]
        row = cleaned_fundamentals.select(
            [F.countDistinct(F.col(c)).alias(c) for c in chunk]
        ).collect()[0].asDict()
        unique_counts_rows.append(row)
    unique_counts = {}
    for r in unique_counts_rows:
        unique_counts.update(r)

    variance_rows = []
    for i in range(0, len(candidate_feature_columns), CHUNK_SIZE):
        chunk = candidate_feature_columns[i:i + CHUNK_SIZE]
        row = cleaned_fundamentals.select(
            [F.variance(F.col(c)).alias(c) for c in chunk]
        ).collect()[0].asDict()
        for c in chunk:
            variance_rows.append({'feature': c, 'variance': row[c]})
    variance_df = pd.DataFrame(variance_rows)

    # Reuses Utils.remove_static_features verbatim (same 50-company sample / 0.001 threshold /
    # ">half of sampled companies" majority rule the real pipeline applies per-fold) purely to
    # report which features WOULD be flagged - nothing is dropped from any output here.
    _, features_flagged_static = utils.remove_static_features(
        cleaned_fundamentals.select('fsym', *candidate_feature_columns), candidate_feature_columns
    )
    flagged_set = set(features_flagged_static)

    variance_df['unique_non_null_count'] = variance_df['feature'].map(unique_counts)
    variance_df['std'] = np.sqrt(variance_df['variance'].astype(float))
    variance_df['flagged_static_by_pipeline_rule'] = variance_df['feature'].isin(flagged_set)
    variance_df = variance_df[['feature', 'unique_non_null_count', 'variance', 'std', 'flagged_static_by_pipeline_rule']]
    variance_df = variance_df.sort_values('variance', na_position='first').reset_index(drop=True)
    save_df(variance_df, f"{DIR_VARIATION}/feature_variation.csv")

    return variance_df, len(flagged_set)


def section_feature_distributions(cleaned_fundamentals, candidate_feature_columns):
    """7. FEATURE DISTRIBUTIONS - full descriptive-statistics table for every raw candidate
    feature, plus a small set of representative histograms (never one plot per feature)."""
    print("[7] feature distributions (this is the slowest section - percentile_approx per "
          f"feature, {len(candidate_feature_columns)} features)...")
    stats_df = chunked_descriptive_stats(cleaned_fundamentals, candidate_feature_columns)
    save_df(stats_df, f"{DIR_DISTRIBUTIONS}/feature_descriptive_statistics.csv")

    total_rows = cleaned_fundamentals.count()
    coverage_floor = max(200, 0.05 * total_rows)
    eligible = stats_df[
        (stats_df['count'].fillna(0) >= coverage_floor)
        & stats_df['std'].notna() & (stats_df['std'] > 0)
        & (stats_df['p75'] > stats_df['p25'])
    ].copy()

    representatives = {}
    reasons = {}
    if not eligible.empty:
        eligible['abs_skew'] = eligible['skewness'].abs()
        symmetric = eligible.loc[eligible['abs_skew'].idxmin(), 'feature']
        representatives['symmetric'] = symmetric
        reasons['symmetric'] = 'smallest |skewness| among features with >=5% row coverage'

        pos_skew = eligible.loc[eligible['skewness'].idxmax(), 'feature']
        representatives['positively_skewed'] = pos_skew
        reasons['positively_skewed'] = 'largest (most positive) skewness'

        neg_skew = eligible.loc[eligible['skewness'].idxmin(), 'feature']
        representatives['negatively_skewed'] = neg_skew
        reasons['negatively_skewed'] = 'smallest (most negative) skewness'

        eligible['outlier_ratio'] = (eligible['p99'] - eligible['p1']) / (eligible['p75'] - eligible['p25'])
        outliers = eligible.loc[eligible['outlier_ratio'].idxmax(), 'feature']
        representatives['substantial_outliers'] = outliers
        reasons['substantial_outliers'] = 'largest (p99-p1)/(p75-p25) ratio - heaviest tails relative to the interquartile range'

    for label, feature in representatives.items():
        row = stats_df[stats_df['feature'] == feature].iloc[0]
        hist = spark_histogram(cleaned_fundamentals, feature, num_bins=50, lo=row['p1'], hi=row['p99'])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_hist = hist[(hist['bin_index'] >= 1) & (hist['bin_index'] <= 50)]
        ax.bar(plot_hist['bin_left'], plot_hist['count'],
               width=(plot_hist['bin_right'] - plot_hist['bin_left']), align='edge')
        ax.set_title(f"{feature}\n({label.replace('_', ' ')}, skew={row['skewness']:.2f})")
        ax.set_xlabel(f"{feature} (clipped to [p1, p99])")
        ax.set_ylabel('Count')
        save_fig(f"{DIR_DISTRIBUTIONS}/hist_{label}_{feature}.png")

    save_json({'selected_features': representatives, 'selection_reasons': reasons,
               'coverage_floor_rows': coverage_floor}, f"{DIR_DISTRIBUTIONS}/representative_features.json")
    return stats_df, representatives


def section_feature_correlation(cleaned_fundamentals, candidate_feature_columns, seed):
    """8. FEATURE REDUNDANCY / CORRELATION - descriptive only (see module docstring's leakage
    constraint: never used here to select features for modelling). Pairwise Spearman on a
    seeded company sample, pandas' pairwise-complete-observations handling (NOT a 0-fill), so
    each pair's correlation only uses rows where both features are actually observed."""
    print("[8] feature correlation...")
    sampled = sample_companies(cleaned_fundamentals, 'fsym', seed=seed, fraction=0.2, max_companies=400)
    pdf = (
        cleaned_fundamentals.filter(F.col('fsym').isin(sampled))
        .select(*candidate_feature_columns).toPandas()
    )
    corr = pdf.corr(method='spearman', min_periods=10)
    notna_mask = pdf.notna().astype(int)
    pair_counts = notna_mask.T.dot(notna_mask)

    pairs = []
    cols = candidate_feature_columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            rho = corr.loc[f1, f2]
            if pd.isna(rho):
                continue
            pairs.append({
                'feature_1': f1, 'feature_2': f2, 'spearman_rho': rho,
                'abs_spearman_rho': abs(rho), 'n_pairwise_observations': int(pair_counts.loc[f1, f2]),
            })
    pairs_df = pd.DataFrame(pairs).sort_values('abs_spearman_rho', ascending=False).reset_index(drop=True)
    save_df(pairs_df, f"{DIR_CORRELATION}/feature_spearman_correlations.csv")

    total_pairs = len(pairs_df)
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    threshold_counts = {}
    for t in thresholds:
        n = int((pairs_df['abs_spearman_rho'] >= t).sum())
        threshold_counts[str(t)] = {'n_pairs': n, 'pct_pairs': n / total_pairs if total_pairs else None}
    corr_summary = {
        'n_companies_sampled': len(sampled),
        'n_features': len(candidate_feature_columns),
        'total_feature_pairs_evaluated': total_pairs,
        'thresholds': threshold_counts,
    }
    save_json(corr_summary, f"{DIR_CORRELATION}/correlation_summary.json")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs_df['abs_spearman_rho'], bins=50)
    ax.set_xlabel('|Spearman rho|')
    ax.set_ylabel('Number of feature pairs')
    ax.set_title('Finance-sector raw candidate feature pairwise |Spearman correlation|')
    save_fig(f"{DIR_CORRELATION}/absolute_correlation_distribution.png")

    top_features = pd.unique(pairs_df.head(25)[['feature_1', 'feature_2']].values.ravel())[:20]
    if len(top_features) >= 2:
        sub_corr = corr.loc[top_features, top_features]
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(sub_corr.values, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=90, fontsize=7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=7)
        ax.set_title('Spearman correlation heatmap - most strongly correlated features')
        fig.colorbar(im, ax=ax, fraction=0.046)
        save_fig(f"{DIR_CORRELATION}/correlation_heatmap_top_features.png")

    return pairs_df, corr_summary


def section_monthly_return_target(monthly_returns):
    """9. MONTHLY RETURN TARGET - the GA's prediction target. Distribution + cross-sectional
    dispersion through time (a temporal-operator-relevant question: how much does the
    cross-section actually vary month to month)."""
    print("[9] monthly return target...")
    stats_row = monthly_returns.select(
        F.count('monthly_return').alias('count'),
        F.mean('monthly_return').alias('mean'),
        F.stddev('monthly_return').alias('std'),
        F.min('monthly_return').alias('min'),
        F.max('monthly_return').alias('max'),
        F.skewness('monthly_return').alias('skewness'),
        F.kurtosis('monthly_return').alias('kurtosis'),
        F.expr(f"percentile_approx(monthly_return, array({','.join(str(p) for p in PERCENTILES)}), 10000)").alias('pcts'),
    ).collect()[0]
    pcts = stats_row['pcts']
    summary = {
        'count': stats_row['count'], 'mean': stats_row['mean'], 'median': pcts[3],
        'std': stats_row['std'], 'min': stats_row['min'], 'max': stats_row['max'],
        'p1': pcts[0], 'p5': pcts[1], 'p25': pcts[2], 'p75': pcts[4], 'p95': pcts[5], 'p99': pcts[6],
        'skewness': stats_row['skewness'], 'kurtosis': stats_row['kurtosis'],
    }
    save_json(summary, f"{DIR_RETURNS}/monthly_return_summary.json")

    hist = spark_histogram(monthly_returns, 'monthly_return', num_bins=60, lo=pcts[0], hi=pcts[6])
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_hist = hist[(hist['bin_index'] >= 1) & (hist['bin_index'] <= 60)]
    ax.bar(plot_hist['bin_left'], plot_hist['count'], width=(plot_hist['bin_right'] - plot_hist['bin_left']), align='edge')
    ax.set_xlabel('Monthly return (clipped to [p1, p99])')
    ax.set_ylabel('Number of security-months')
    ax.set_title('Finance-sector monthly return distribution')
    save_fig(f"{DIR_RETURNS}/monthly_return_distribution.png")

    ts = (
        monthly_returns.groupBy('year', 'month').agg(
            F.mean('monthly_return').alias('mean_return'),
            F.expr('percentile_approx(monthly_return, 0.5, 10000)').alias('median_return'),
            F.stddev('monthly_return').alias('std_return'),
            F.expr('percentile_approx(monthly_return, 0.75, 10000)').alias('p75'),
            F.expr('percentile_approx(monthly_return, 0.25, 10000)').alias('p25'),
            F.count('*').alias('n_securities'),
        )
        .withColumn('iqr_return', F.col('p75') - F.col('p25'))
        .orderBy('year', 'month')
        .toPandas()
    )
    save_df(ts, f"{DIR_RETURNS}/monthly_return_time_series.csv")

    x = pd.to_datetime(ts['year'].astype(str) + '-' + ts['month'].astype(str) + '-01')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, ts['mean_return'], label='Mean')
    ax.plot(x, ts['median_return'], label='Median', alpha=0.7)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cross-sectional monthly return')
    ax.set_title('Finance-sector mean/median monthly return through time')
    ax.legend()
    save_fig(f"{DIR_RETURNS}/finance_mean_monthly_return_over_time.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, ts['std_return'], label='Std dev')
    ax.plot(x, ts['iqr_return'], label='IQR', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cross-sectional dispersion')
    ax.set_title('Finance-sector monthly return dispersion through time')
    ax.legend()
    save_fig(f"{DIR_RETURNS}/finance_return_dispersion_over_time.png")

    return summary, ts


def section_reporting_frequency(df_fundamentals):
    """10. FUNDAMENTAL REPORTING FREQUENCY - how often fundamentals actually update per
    company/year, and the gap between successive reports. Directly relevant to the temporal
    operators (lag/delta/growth/mean/std), which operate on the report grain, not the monthly
    as-of-expanded grain."""
    print("[10] fundamental reporting frequency...")
    from pyspark.sql.window import Window
    w = Window.partitionBy('fsym').orderBy('date')
    with_gaps = df_fundamentals.select('fsym', 'date').withColumn(
        'interval_days', F.datediff('date', F.lag('date', 1).over(w))
    )

    by_company_year = (
        df_fundamentals.groupBy('fsym', F.year('date').alias('year')).agg(F.count('*').alias('n_reports'))
        .orderBy('fsym', 'year')
        .toPandas()
    )
    save_df(by_company_year, f"{DIR_REPORTING_FREQ}/report_frequency_by_company_year.csv")

    interval_stats = with_gaps.select(
        F.count('interval_days').alias('count'),
        F.mean('interval_days').alias('mean'),
        F.stddev('interval_days').alias('std'),
        F.min('interval_days').alias('min'),
        F.max('interval_days').alias('max'),
        F.expr(f"percentile_approx(interval_days, array({','.join(str(p) for p in PERCENTILES)}), 10000)").alias('pcts'),
    ).collect()[0]
    pcts = interval_stats['pcts']
    interval_summary = {
        'count': interval_stats['count'], 'mean': interval_stats['mean'], 'median': pcts[3],
        'std': interval_stats['std'], 'min': interval_stats['min'], 'max': interval_stats['max'],
        'p1': pcts[0], 'p5': pcts[1], 'p25': pcts[2], 'p75': pcts[4], 'p95': pcts[5], 'p99': pcts[6],
    }
    save_json(interval_summary, f"{DIR_REPORTING_FREQ}/report_interval_summary.json")

    hist = spark_histogram(with_gaps, 'interval_days', num_bins=60, lo=0, hi=max(400, pcts[6] or 400))
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_hist = hist[(hist['bin_index'] >= 1) & (hist['bin_index'] <= 60)]
    ax.bar(plot_hist['bin_left'], plot_hist['count'], width=(plot_hist['bin_right'] - plot_hist['bin_left']), align='edge')
    ax.set_xlabel('Days between consecutive fundamental reports')
    ax.set_ylabel('Number of report-to-report intervals')
    ax.set_title('Finance-sector fundamental reporting interval distribution')
    save_fig(f"{DIR_REPORTING_FREQ}/report_interval_distribution.png")

    return by_company_year, interval_summary


def section_filing_delay(df_fundamentals):
    """11. FILING DELAY / POINT-IN-TIME AVAILABILITY (high priority) - how long after a fiscal
    period end a report actually becomes known (`fund_report_date - date`), which is exactly
    what the walk-forward pipeline's point-in-time join (`feature_selection_dataset`) depends
    on being right. See assumptions 5-6 for how valid/fallback/invalid rows are distinguished -
    `Utils.get_fundamental_data()` doesn't expose that flag itself, so it's recomputed here from
    the same `ff_source_is_date`/`ff_source_bs_date`/`ff_source_cf_date` columns it uses
    internally (not new logic - the identical condition it already evaluates)."""
    print("[11] filing delay / point-in-time availability...")

    base = df_fundamentals.select(
        'fsym', 'date', 'fund_report_date', 'ff_source_is_date', 'ff_source_bs_date', 'ff_source_cf_date'
    ).withColumn(
        '_raw_source_date', F.greatest('ff_source_is_date', 'ff_source_bs_date', 'ff_source_cf_date')
    ).withColumn(
        '_raw_source_lag_days', F.datediff('_raw_source_date', 'date')
    ).withColumn(
        'is_fallback', F.col('_raw_source_date').isNull() | (F.col('_raw_source_lag_days') < 0)
    ).withColumn(
        'is_negative_raw_lag', F.col('_raw_source_date').isNotNull() & (F.col('_raw_source_lag_days') < 0)
    ).withColumn(
        'filing_delay_days', F.datediff('fund_report_date', 'date')
    ).cache()

    total_reports = base.count()
    invalid = base.filter(F.col('is_negative_raw_lag')).select(
        'fsym', 'date', 'fund_report_date', '_raw_source_date', '_raw_source_lag_days'
    ).withColumnRenamed('_raw_source_date', 'raw_source_implied_filing_date') \
     .withColumnRenamed('_raw_source_lag_days', 'raw_source_implied_delay_days') \
     .toPandas()
    save_df(invalid, f"{DIR_FILING_DELAY}/invalid_filing_delays.csv")

    all_delays = base.select(
        'fsym', 'date', 'fund_report_date', 'filing_delay_days', 'is_fallback', 'is_negative_raw_lag'
    ).toPandas()
    save_df(all_delays, f"{DIR_FILING_DELAY}/filing_delays.csv")

    valid = base.filter(~F.col('is_fallback'))
    n_valid = valid.count()

    stats_row = valid.select(
        F.mean('filing_delay_days').alias('mean'),
        F.min('filing_delay_days').alias('min'),
        F.max('filing_delay_days').alias('max'),
        F.expr('percentile_approx(filing_delay_days, array(0.25,0.5,0.75,0.9,0.95,0.99), 10000)').alias('pcts'),
    ).collect()[0]
    pcts = stats_row['pcts']

    thresholds = [31, 45, 60, 90, 120]
    threshold_rows = []
    for t in thresholds:
        n_within = valid.filter(F.col('filing_delay_days') <= t).count()
        threshold_rows.append({
            'threshold_days': t, 'n_within': n_within,
            'pct_within': n_within / n_valid if n_valid else None,
        })
    threshold_df = pd.DataFrame(threshold_rows)
    save_df(threshold_df, f"{DIR_FILING_DELAY}/filing_availability_thresholds.csv")

    pct_unavailable_after_31 = 1 - (threshold_df.loc[threshold_df['threshold_days'] == 31, 'pct_within'].iloc[0] or 0)

    summary = {
        'total_reports': total_reports,
        'reports_with_valid_filing_dates': n_valid,
        'pct_with_valid_filing_dates': n_valid / total_reports if total_reports else None,
        'reports_using_fallback_lag': total_reports - n_valid,
        'reports_with_negative_raw_source_lag': int(invalid.shape[0]),
        'mean_delay_days': stats_row['mean'],
        'median_delay_days': pcts[1],
        'p25_delay_days': pcts[0],
        'p75_delay_days': pcts[2],
        'p90_delay_days': pcts[3],
        'p95_delay_days': pcts[4],
        'p99_delay_days': pcts[5],
        'min_delay_days': stats_row['min'],
        'max_delay_days': stats_row['max'],
        'availability_thresholds': {str(r['threshold_days']): r['pct_within'] for r in threshold_rows},
        'pct_unavailable_after_31_days': pct_unavailable_after_31,
    }
    save_json(summary, f"{DIR_FILING_DELAY}/filing_delay_summary.json")

    hist = spark_histogram(valid, 'filing_delay_days', num_bins=60, lo=0, hi=max(180, pcts[5] or 180))
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_hist = hist[(hist['bin_index'] >= 1) & (hist['bin_index'] <= 60)]
    ax.bar(plot_hist['bin_left'], plot_hist['count'], width=(plot_hist['bin_right'] - plot_hist['bin_left']), align='edge')
    ax.axvline(31, color='red', linestyle='--', label='31-day assumed lag (previous pipeline)')
    ax.set_xlabel('Filing delay (days)')
    ax.set_ylabel('Number of reports')
    ax.set_title('Finance-sector filing delay distribution (valid observed filing dates only)')
    ax.legend()
    save_fig(f"{DIR_FILING_DELAY}/filing_delay_distribution.png")

    delay_values = all_delays.loc[~all_delays['is_fallback'], 'filing_delay_days'].dropna().sort_values()
    cdf_y = np.arange(1, len(delay_values) + 1) / len(delay_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delay_values.values, cdf_y)
    median_delay = float(delay_values.median())
    ax.axvline(median_delay, color='gray', linestyle='--', label=f'Median ({median_delay:.0f} days)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
    ax.set_xlim(0, max(180, float(delay_values.quantile(0.99)) if len(delay_values) else 180))
    ax.set_xlabel('Filing delay (days)')
    ax.set_ylabel('Cumulative fraction of reports available')
    ax.set_title('Finance-sector filing delay CDF')
    ax.legend()
    save_fig(f"{DIR_FILING_DELAY}/filing_delay_cdf.png")

    base.unpersist()
    return summary


def section_pipeline_audit(utils, df_fundamentals, aligned_fundamentals, cleaned_fundamentals,
                            candidate_feature_columns):
    """12. RAW-TO-USABLE DATASET AUDIT - dataset dimensions at each pipeline stage obtainable
    without changing the main pipeline. Calls `Utils.drop_na_features`/`remove_static_features`
    directly (see assumption 7: whole-history here, purely descriptive - the real pipeline fits
    both per-fold, train-only, inside `run_fold()`)."""
    print("[12] raw-to-usable dataset audit...")
    rows = []

    n_companies_raw = df_fundamentals.select('fsym').distinct().count()
    n_rows_raw = df_fundamentals.count()
    rows.append({'stage': 'raw_extracted_finance_dataset', 'n_companies': n_companies_raw,
                 'n_rows': n_rows_raw, 'n_candidate_features': None,
                 'notes': 'Utils.get_fundamental_data() output - already sector/market-cap/'
                          'listing/currency filtered at SQL level'})

    n_companies_aligned = aligned_fundamentals.select('fsym').distinct().count()
    n_rows_aligned = aligned_fundamentals.count()
    rows.append({'stage': 'after_universe_restriction_and_pit_alignment', 'n_companies': n_companies_aligned,
                 'n_rows': n_rows_aligned, 'n_candidate_features': None,
                 'notes': 'Utils.align_data() - restricts to fsyms with overlapping fundamentals/price date ranges'})

    n_companies_cleaned = cleaned_fundamentals.select('fsym').distinct().count()
    n_rows_cleaned = cleaned_fundamentals.count()
    rows.append({'stage': 'after_dropping_invalid_rows', 'n_companies': n_companies_cleaned,
                 'n_rows': n_rows_cleaned, 'n_candidate_features': len(candidate_feature_columns),
                 'notes': 'Utils.drop_non_numeric_columns() - this pipeline drops no fundamentals '
                          'rows here (invalid-row exclusion is column-level via non-numeric drop; '
                          'see monthly_return_target section for the return-side sanity-filter '
                          'row exclusions)'})

    train_view = cleaned_fundamentals.select('fsym', 'date', *candidate_feature_columns)
    train_view, dropped_na, _ = utils.drop_na_features(train_view, drop_threshold=0.3)
    surviving_after_na = [c for c in candidate_feature_columns if c not in dropped_na]
    rows.append({'stage': 'after_missingness_filtering_30pct', 'n_companies': n_companies_cleaned,
                 'n_rows': n_rows_cleaned, 'n_candidate_features': len(surviving_after_na),
                 'notes': 'Utils.drop_na_features(drop_threshold=0.3), whole-history (descriptive '
                          'only - real pipeline fits this per-fold, train-only)'})

    train_view, dropped_static = utils.remove_static_features(train_view, surviving_after_na)
    surviving_after_static = [c for c in surviving_after_na if c not in dropped_static]
    rows.append({'stage': 'after_static_feature_removal', 'n_companies': n_companies_cleaned,
                 'n_rows': n_rows_cleaned, 'n_candidate_features': len(surviving_after_static),
                 'notes': 'Utils.remove_static_features() default params (50-company sample, '
                          'var_threshold=0.001), whole-history (descriptive only)'})

    rows.append({'stage': 'final_candidate_set_before_consensus_feature_selection',
                 'n_companies': n_companies_cleaned, 'n_rows': n_rows_cleaned,
                 'n_candidate_features': len(surviving_after_static),
                 'notes': 'Per-fold ConsensusFeatureSelector (Spearman/MI/RF-permutation '
                          'consensus + correlation clustering, capped at 40) is fit train-only, '
                          'per fold, inside run_fold() - not reproduced here since it is '
                          'genuinely fold-specific supervised feature selection, out of scope '
                          'for descriptive EDA (see module docstring leakage constraint)'})

    audit_df = pd.DataFrame(rows)
    save_df(audit_df, f"{DIR_AUDIT}/dataset_pipeline_audit.csv")
    return audit_df


def section_master_summary(dataset_scope, missingness_df, variation_df, n_static_flagged,
                            corr_summary, return_summary, filing_delay_summary,
                            candidate_feature_columns, per_company_obs):
    """13. MASTER SUMMARY - the headline numbers a thesis results/methodology section needs,
    gathered in one place from every section above."""
    print("[13] master summary...")
    thr_08 = corr_summary['thresholds'].get('0.8', {})
    summary = {
        'sector': 'Finance',
        'sample_period': {
            'earliest_fundamental_date': dataset_scope['earliest_fundamental_date'],
            'latest_fundamental_date': dataset_scope['latest_fundamental_date'],
            'earliest_price_date': dataset_scope['earliest_price_date'],
            'latest_price_date': dataset_scope['latest_price_date'],
        },
        'n_finance_companies': dataset_scope['n_unique_finance_companies'],
        'n_finance_securities': dataset_scope['n_unique_securities'],
        'n_fundamental_reports': dataset_scope['n_raw_fundamental_observations'],
        'n_monthly_return_observations': dataset_scope['n_monthly_return_observations'],
        'n_raw_candidate_features': len(candidate_feature_columns),
        'median_companies_per_month': dataset_scope['finance_companies_per_month_median'],
        'median_fundamental_observations_per_company': dataset_scope['fundamental_observations_per_company_median'],
        'median_feature_missingness_pct': float(missingness_df['pct_missing'].median()),
        'n_features_above_30pct_missing': int((missingness_df['pct_missing'] > 0.30).sum()),
        'n_static_or_near_static_features': int(n_static_flagged),
        'pct_feature_pairs_abs_spearman_ge_0_8': thr_08.get('pct_pairs'),
        'monthly_return_mean': return_summary['mean'],
        'monthly_return_std': return_summary['std'],
        'median_filing_delay_days': filing_delay_summary['median_delay_days'],
        'p90_filing_delay_days': filing_delay_summary['p90_delay_days'],
        'pct_reports_available_within_31_days': filing_delay_summary['availability_thresholds'].get('31'),
        'pct_reports_unavailable_after_31_days': filing_delay_summary['pct_unavailable_after_31_days'],
    }
    save_json(summary, f"{OUTPUT_ROOT}/eda_summary.json")
    return summary


# ================================================================================= main ====

def main():
    random.seed(FULL_CONFIG.random_seed)
    np.random.seed(FULL_CONFIG.random_seed)
    ensure_output_dirs()

    spark = _build_spark_session()
    config = FULL_CONFIG  # full scale, sector_filter='Finance', full 2001-> history - see module docstring
    written_files = []

    try:
        utils = Utils(
            num_partitions=config.num_partitions,
            start_date=config.start_date,
            market_value_threshold=config.market_value_threshold,
            sector_filter=config.sector_filter,
        )
        print(f"finance_eda: sector_filter={utils.sector_filter!r}, start_date={utils.start_date!r}, "
              f"market_value_threshold={utils.market_value_threshold}")

        print("Loading raw fundamentals / price data (Utils.get_fundamental_data/get_price_data)...")
        df_fundamentals = utils.get_fundamental_data().cache()
        df_prices = utils.get_price_data().cache()
        print(f"  raw fundamentals rows={df_fundamentals.count()}, raw price rows={df_prices.count()}")

        print("Aligning fundamentals/price on overlapping date ranges (Utils.align_data)...")
        aligned_fundamentals, aligned_prices = utils.align_data(df_fundamentals, df_prices)

        print("Dropping non-numeric columns to get the raw candidate feature set...")
        cleaned_fundamentals, _ = utils.drop_non_numeric_columns(
            aligned_fundamentals, keep_columns=['date', 'fsym', 'factset_sector_desc', 'fsym_id', 'fund_report_date']
        )
        cleaned_fundamentals = cleaned_fundamentals.cache()
        candidate_feature_columns = [
            c for c in cleaned_fundamentals.columns
            if c not in ['date', 'fsym', 'factset_sector_desc', 'fsym_id', 'fund_report_date']
        ]
        print(f"  raw candidate fundamental features: {len(candidate_feature_columns)}")

        print("Building the monthly return target (Utils.replace_price_outliers -> calculate_monthly_returns)...")
        smoothed_prices = utils.replace_price_outliers(aligned_prices, "adj_price", std=10)
        monthly_returns = utils.calculate_monthly_returns(smoothed_prices).cache()
        print(f"  monthly_returns rows={monthly_returns.count()}")

        print(f"Loading the point-in-time eligible monthly panel (checkpoint at {config.walk_forward_base_path})...")
        try:
            eligible_monthly_df = spark.read.parquet(config.walk_forward_base_path).select('fsym', 'target_date').cache()
            eligible_monthly_df.count()
            print("  read existing walk-forward base checkpoint")
        except Exception as e:
            print(f"  checkpoint not found/readable ({e}); building fresh via Utils.feature_selection_dataset "
                  "(same point-in-time definition, just not the persisted copy)")
            base_fundamentals_df = utils.interpolate_missing_values(cleaned_fundamentals, candidate_feature_columns)
            monthly_returns_with_lag = utils.add_return_lag_features(monthly_returns)
            eligible_monthly_df = utils.feature_selection_dataset(
                base_fundamentals_df, monthly_returns_with_lag
            ).select('fsym', 'target_date').cache()
            eligible_monthly_df.count()

        # ---- sections ----
        dataset_scope = section_dataset_scope(df_fundamentals, df_prices, monthly_returns, candidate_feature_columns)
        written_files += [f"{DIR_SCOPE}/dataset_summary.json", f"{DIR_SCOPE}/dataset_summary.csv"]

        section_company_coverage(df_fundamentals, monthly_returns, eligible_monthly_df)
        written_files += [f"{DIR_COVERAGE}/company_coverage_monthly.csv",
                           f"{DIR_COVERAGE}/company_coverage_yearly.csv",
                           f"{DIR_COVERAGE}/finance_companies_over_time.png"]

        per_company_obs, obs_summary = section_observation_coverage(df_fundamentals)
        written_files += [f"{DIR_OBS_COVERAGE}/company_observation_coverage.csv",
                           f"{DIR_OBS_COVERAGE}/observation_coverage_summary.json"]

        section_market_cap(utils, df_fundamentals)
        written_files += [f"{DIR_MARKET_CAP}/market_cap_summary.csv",
                           f"{DIR_MARKET_CAP}/market_cap_distribution.csv",
                           f"{DIR_MARKET_CAP}/market_cap_coverage_monthly.csv",
                           f"{DIR_MARKET_CAP}/market_cap_distribution.png",
                           f"{DIR_MARKET_CAP}/market_cap_coverage_over_time.png"]

        missingness_df, bands_df = section_feature_missingness(cleaned_fundamentals, candidate_feature_columns)
        written_files += [f"{DIR_MISSINGNESS}/feature_missingness.csv",
                           f"{DIR_MISSINGNESS}/missingness_bands.csv",
                           f"{DIR_MISSINGNESS}/feature_missingness_distribution.png"]

        variation_df, n_static_flagged = section_feature_variation(utils, cleaned_fundamentals, candidate_feature_columns)
        written_files += [f"{DIR_VARIATION}/feature_variation.csv"]

        section_feature_distributions(cleaned_fundamentals, candidate_feature_columns)
        written_files += [f"{DIR_DISTRIBUTIONS}/feature_descriptive_statistics.csv",
                           f"{DIR_DISTRIBUTIONS}/representative_features.json"]

        _, corr_summary = section_feature_correlation(cleaned_fundamentals, candidate_feature_columns, seed=config.random_seed)
        written_files += [f"{DIR_CORRELATION}/feature_spearman_correlations.csv",
                           f"{DIR_CORRELATION}/correlation_summary.json",
                           f"{DIR_CORRELATION}/absolute_correlation_distribution.png"]

        return_summary, _ = section_monthly_return_target(monthly_returns)
        written_files += [f"{DIR_RETURNS}/monthly_return_summary.json",
                           f"{DIR_RETURNS}/monthly_return_distribution.png",
                           f"{DIR_RETURNS}/monthly_return_time_series.csv",
                           f"{DIR_RETURNS}/finance_mean_monthly_return_over_time.png"]

        section_reporting_frequency(df_fundamentals)
        written_files += [f"{DIR_REPORTING_FREQ}/report_frequency_by_company_year.csv",
                           f"{DIR_REPORTING_FREQ}/report_interval_summary.json",
                           f"{DIR_REPORTING_FREQ}/report_interval_distribution.png"]

        filing_delay_summary = section_filing_delay(df_fundamentals)
        written_files += [f"{DIR_FILING_DELAY}/filing_delays.csv",
                           f"{DIR_FILING_DELAY}/filing_delay_summary.json",
                           f"{DIR_FILING_DELAY}/filing_availability_thresholds.csv",
                           f"{DIR_FILING_DELAY}/filing_delay_distribution.png",
                           f"{DIR_FILING_DELAY}/filing_delay_cdf.png",
                           f"{DIR_FILING_DELAY}/invalid_filing_delays.csv"]

        section_pipeline_audit(utils, df_fundamentals, aligned_fundamentals, cleaned_fundamentals,
                                candidate_feature_columns)
        written_files += [f"{DIR_AUDIT}/dataset_pipeline_audit.csv"]

        section_master_summary(dataset_scope, missingness_df, variation_df, n_static_flagged,
                                corr_summary, return_summary, filing_delay_summary,
                                candidate_feature_columns, per_company_obs)
        written_files += [f"{OUTPUT_ROOT}/eda_summary.json"]

        df_fundamentals.unpersist()
        df_prices.unpersist()
        cleaned_fundamentals.unpersist()
        monthly_returns.unpersist()
        eligible_monthly_df.unpersist()

    finally:
        print("Stopping SparkSession...")
        spark.stop()

    print(f"\n=== finance_eda: done. {len(written_files)} files written under {OUTPUT_ROOT}/ ===")
    for f in written_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
