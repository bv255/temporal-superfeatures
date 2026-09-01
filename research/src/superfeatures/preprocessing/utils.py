"""
Ported from `research/PreProcessing_test.ipynb` cell 2 — see `docs/RESTRUCTURING_TODO.md`
and the port plan in `docs/RESEARCH_STRUCTURE.md`. Extracted mechanically from the notebook
cell source (not retyped) to avoid transcription drift; the notebook remains the frozen parity
reference. Only the import block was touched: names the cell relied on getting from the shared
notebook kernel namespace (star-imports in cell 0, e.g. `from pyspark.sql.types import *`) are
now imported explicitly here, since a standalone module has no such shared kernel state.
"""

# Utils class to preprocess data (price for return computing + fundamentals)
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import when, lit, col, greatest, least, log, exp, sum, coalesce, datediff, lag, variance, first, last, year, month, add_months, broadcast
from typing import Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
from functools import reduce
import os
from pathlib import Path
from pyspark.sql.types import DecimalType, DoubleType, FloatType, IntegerType, LongType

class Utils():
    
    def __init__(self, num_partitions: int = 20, start_date: str = '2001-01-01', market_value_threshold: int = 500, sector_filter: str = 'Finance'):
        print("Utils.__init__: starting SparkSession...")
        self.spark = SparkSession.builder.master("yarn").appName("SuperFeatures_2024").getOrCreate()
        print("Utils.__init__: SparkSession ready.")
        BASE = "hdfs://bialobog:8020/user/hive/warehouse/2026_07_06_us"
        tables = ["ff_basic_qf", "ff_basic_der_qf", "ff_advanced_qf", "ff_advanced_der_qf", "sym_entity_sector", "fp_sec_entity", "sym_coverage", "factset_sector_map", "ff_basic_cf", "fp_basic_prices", "fp_basic_dividends", "fp_basic_splits"]
        for i, t in enumerate(tables, start=1):
            print(f"Utils.__init__: [{i}/{len(tables)}] registering temp view {t}")
            self.spark.read.parquet(f"{BASE}/{t}").createOrReplaceTempView(t)
        print("Utils.__init__: all temp views registered.")
        self.num_partitions = num_partitions
        self.start_date = start_date
        self.market_value_threshold = market_value_threshold
        # Restricts get_fundamental_data/get_price_data to a single sector at the SQL level
        # (pushed into the WHERE clause, not filtered after the fact). Set to None to restore
        # the full multi-sector universe.
        self.sector_filter = sector_filter
        if self.sector_filter:
            print(f"Utils.__init__: sector_filter='{self.sector_filter}' - fundamentals/price queries "
                  f"will be restricted to this sector only")

    def _asof_carry_forward(self, keys_df: DataFrame, key_fsym_col: str, key_date_col: str,
                             source_sql: str, source_fsym_col: str, source_date_col: str,
                             source_val_col: str, out_col: str, max_staleness_days: int) -> DataFrame:
        """
        Causal as-of join: for each (fsym, date) row in keys_df, attach the most recent non-null
        value from a source series (source_sql, keyed by source_fsym_col/source_date_col) at or
        before that date - never a later one. Same forward-fill idiom as
        interpolate_missing_values/replace_price_outliers: union the two date axes per fsym,
        then F.last(ignorenulls=True) over an unbounded-preceding window.

        Used to reconstruct a genuine point-in-time market value (shares outstanding x price) in
        place of the old ff_basic_cf-based filter, which joined a CURRENT-only snapshot table
        (one row per fsym_id, no historical date at all - confirmed by inspecting the parquet
        directly) unconditioned on date, so every historical row for a company was filtered by
        its PRESENT-day market value - a look-ahead/survivorship leak in which companies were
        even in the universe at each historical date.

        max_staleness_days bounds how old the carried-forward value's own SOURCE row may be
        relative to the key row's date: if the most recent available source observation is older
        than that, out_col is null rather than silently carrying an arbitrarily stale value
        forward indefinitely. No shared default - required at each call site, since the two
        source series here have very different cadence (a daily price series vs. a quarterly
        shares-outstanding series need very different bounds to mean anything).
        """
        keys = keys_df.select(
            col(key_fsym_col).alias('_fsym'), col(key_date_col).alias('_date')
        ).distinct().withColumn('_val', lit(None).cast('double')).withColumn('_is_key', lit(True))

        source = self.spark.sql(source_sql).select(
            col(source_fsym_col).alias('_fsym'),
            col(source_date_col).alias('_date'),
            col(source_val_col).cast('double').alias('_val')
        ).filter(col('_val').isNotNull()).withColumn('_is_key', lit(False))

        # Restrict the source series to companies actually present in keys_df before the union -
        # source_sql pulls from an unfiltered whole-history table (fp_basic_prices/ff_basic_qf),
        # while keys_df is already scoped to this pipeline's sector/date/listing universe.
        relevant_fsyms = keys.select('_fsym').distinct()
        source = source.join(relevant_fsyms, on='_fsym', how='inner')

        win = Window.partitionBy('_fsym').orderBy('_date', '_is_key').rowsBetween(Window.unboundedPreceding, Window.currentRow)
        combined = (
            keys.unionByName(source)
            .withColumn(out_col, last('_val', ignorenulls=True).over(win))
            # Date of the SOURCE row that produced the carried-forward value above (not the
            # current row's own _date) - a companion carry-forward over the same
            # partition/order, restricted to non-key (source) rows' _date only.
            .withColumn('_source_date', last(when(~col('_is_key'), col('_date')), ignorenulls=True).over(win))
        )

        asof = combined.filter(col('_is_key')).select('_fsym', '_date', out_col, '_source_date')
        asof = asof.withColumn(
            out_col,
            when(datediff(col('_date'), col('_source_date')) > max_staleness_days, lit(None)).otherwise(col(out_col))
        ).drop('_source_date')

        return keys_df.join(
            asof,
            (col(key_fsym_col) == asof['_fsym']) & (col(key_date_col) == asof['_date']),
            'left'
        ).drop('_fsym', '_date')

    def get_fundamental_data(self) -> DataFrame:
        """Select all fundamental data from ff_basic_qf, ff_basic_cf, ff_basic_der_qf, ff_advanced_der_qf, ff_advanced_qf"""
        
        print("get_fundamental_data: resolving column overlaps across the 4 fundamentals tables...")
        fun_columns = self.spark.sql("SELECT * FROM ff_basic_qf LIMIT 1").columns
        funder_columns = self.spark.sql("SELECT * FROM ff_basic_der_qf LIMIT 1").columns
        funder_columns = [col for col in funder_columns if col not in fun_columns]
        fun_adv_columns = self.spark.sql("SELECT * FROM ff_advanced_qf LIMIT 1").columns
        fun_adv_columns = [col for col in fun_adv_columns if col not in fun_columns and col not in funder_columns]
        funder2_columns = self.spark.sql("SELECT * FROM ff_advanced_der_qf LIMIT 1").columns
        funder2_columns = [col for col in funder2_columns if col not in fun_columns and col not in funder_columns and col not in fun_adv_columns]

        # SQL query conversion
        funder2_columns_str = ', '.join([f'funder2.{col}' for col in funder2_columns])
        fun_adv_columns_str = ', '.join([f'fun_adv.{col}' for col in fun_adv_columns])
        funder_columns_str = ', '.join([f'funder.{col}' for col in funder_columns])

        # SQL query
        sector_clause = f"AND sec.factset_sector_desc = '{self.sector_filter}'" if self.sector_filter else ""
        query = f"""
        SELECT DISTINCT sc.fsym_id AS fsym, e.factset_entity_id, sec.factset_sector_desc,
                        fun.*, {funder2_columns_str}, {fun_adv_columns_str}, {funder_columns_str}
        FROM sym_entity_sector AS e

        INNER JOIN fp_sec_entity AS sc ON e.factset_entity_id = sc.factset_entity_id
        INNER JOIN sym_coverage AS sm ON sc.fsym_id = sm.fsym_security_id
        INNER JOIN factset_sector_map AS sec ON e.sector_code = sec.factset_sector_code
        INNER JOIN ff_basic_qf AS fun ON sm.fsym_regional_id = fun.fsym_id
        INNER JOIN ff_basic_der_qf AS funder ON sm.fsym_regional_id = funder.fsym_id and funder.date = fun.date
        INNER JOIN ff_advanced_der_qf AS funder2 ON sm.fsym_regional_id = funder2.fsym_id and funder2.date = fun.date
        INNER JOIN ff_advanced_qf AS fun_adv ON sm.fsym_regional_id = fun_adv.fsym_id and fun_adv.date = fun.date

        WHERE sm.fsym_id IS NOT NULL
        AND (sec.factset_sector_desc != 'Miscellaneous' AND sec.factset_sector_desc != 'Government')
        AND sm.fref_security_type IS NOT NULL AND sm.fref_security_type = 'SHARE'
        AND sm.currency IS NOT NULL AND sm.currency = 'USD'
        AND e.factset_entity_id IS NOT NULL
        AND e.sector_code IS NOT NULL
        AND (sm.fref_listing_exchange = 'NAS' OR sm.fref_listing_exchange = 'NYS')
        AND funder.date >= '{self.start_date}'
        {sector_clause}

        ORDER BY fsym ASC, date DESC
        """

        print("get_fundamental_data: running the joined SQL query...")
        df_fundamentals = self.spark.sql(query)

        # repartition and caching for efficiency
        df_fundamentals = df_fundamentals.repartition(self.num_partitions)

        # --- Point-in-time market value filter: replaces the old ff_basic_cf-based filter (see
        # _asof_carry_forward docstring for why that table can't give a point-in-time value at
        # all). Reconstructed as shares outstanding (ff_com_shs_out, already present via fun.*
        # above, exact-dated to this row's own `date`) x price (fp_basic_prices.p_price -
        # UNADJUSTED, the actual historical price level, not split-back-adjusted - carried
        # forward causally from the most recent trading day at or before this row's `date`).
        # Keyed on 'fsym_id' (not the 'fsym' alias above) - fsym_id here is ff_basic_qf's own
        # fsym_id column, pulled in unaliased via fun.* above, and per the join condition on line
        # ~110 (sm.fsym_regional_id = fun.fsym_id) it's the REGIONAL id, not the security id
        # 'fsym' is. fp_basic_prices.fsym_id is also regional (see get_price_data's own join:
        # sc.fsym_regional_id = p.fsym_id) - so the join keys must agree on 'fsym_id', not 'fsym'.
        # Using 'fsym' here previously joined a security id against a regional-keyed table, which
        # almost never matches, leaving pit_market_value null for nearly every row and the
        # `> market_value_threshold` filter below dropping the entire dataset.
        print("get_fundamental_data: reconstructing point-in-time market value (shares x price)...")
        # fp_basic_prices is a daily series - 14 days of staleness tolerance covers a long
        # weekend plus a holiday or an isolated missing print; anything staler than that is a
        # genuine signal (halt, delisting, data gap), not routine noise.
        df_fundamentals = self._asof_carry_forward(
            df_fundamentals, 'fsym_id', 'date',
            "SELECT fsym_id, p_date, p_price FROM fp_basic_prices",
            'fsym_id', 'p_date', 'p_price', '_asof_price', max_staleness_days=14
        )
        df_fundamentals = df_fundamentals.withColumn(
            'pit_market_value', col('ff_com_shs_out') * col('_asof_price')
        ).filter(col('pit_market_value') > self.market_value_threshold).drop('_asof_price', 'pit_market_value')

        # --- Point-in-time fix: `date` is a report's fiscal PERIOD-END, not when it was
        # actually filed/known. ff_source_is_date/ff_source_bs_date/ff_source_cf_date (already
        # present via fun.* above) are the real filing dates for the income statement/balance
        # sheet/cash flow, so their max is the true date the full report became available.
        # Falls back, for rows missing all three (or with an implausible/negative implied lag -
        # a data error, not a real lag) to `date` + a single global 90th-percentile lag constant,
        # computed once here - not re-fit per fold, since this runs once before any walk-forward
        # split exists.
        df_fundamentals = df_fundamentals.withColumn(
            '_report_date_raw', greatest('ff_source_is_date', 'ff_source_bs_date', 'ff_source_cf_date')
        ).withColumn('_report_lag_days', datediff('_report_date_raw', 'date'))

        df_fundamentals.cache()

        fallback_lag_row = df_fundamentals.filter(col('_report_lag_days') >= 0).select(
            F.expr('percentile_approx(_report_lag_days, 0.9, 10000)').alias('p90')
        ).collect()[0]
        self.fund_report_fallback_lag_days = int(fallback_lag_row['p90']) if fallback_lag_row['p90'] is not None else 45
        print(f"get_fundamental_data: point-in-time fund_report_date fallback lag (90th percentile "
              f"of the observed filing lag, computed once over the whole dataset) = "
              f"{self.fund_report_fallback_lag_days} days")

        df_fundamentals = df_fundamentals.withColumn(
            'fund_report_date',
            when(
                col('_report_date_raw').isNotNull() & (col('_report_lag_days') >= 0),
                col('_report_date_raw')
            ).otherwise(F.date_add(col('date'), self.fund_report_fallback_lag_days))
        ).drop('_report_date_raw', '_report_lag_days')

        print("get_fundamental_data: done (query is lazy - it materializes on the first action).")
        
        return df_fundamentals
    
    def get_price_data(self) -> DataFrame:
        """Select all price data"""
        
        print("get_price_data: building query...")
        sector_clause = f"AND sm.factset_sector_desc = '{self.sector_filter}'" if self.sector_filter else ""
        query = f"""
        SELECT DISTINCT s.fsym_id AS fsym_id, sc.fsym_regional_id, sm.factset_sector_desc, sc.proper_name, p.p_date, p.p_price,
                        divs.p_divs_exdate, divs.p_divs_pd, divs.p_divs_s_spinoff, splits.p_split_date, splits.p_split_factor

        FROM sym_entity_sector AS e

        INNER JOIN fp_sec_entity AS s ON e.factset_entity_id = s.factset_entity_id
        INNER JOIN sym_coverage AS sc ON s.fsym_id = sc.fsym_security_id
        INNER JOIN fp_basic_prices AS p ON sc.fsym_regional_id = p.fsym_id
        INNER JOIN factset_sector_map AS sm ON e.sector_code = sm.factset_sector_code
        LEFT JOIN fp_basic_dividends AS divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
        LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id

        WHERE sc.fsym_security_id IS NOT NULL
        AND (sm.factset_sector_desc != 'Miscellaneous' AND sm.factset_sector_desc != 'Government')
        AND sc.fref_security_type IS NOT NULL AND sc.fref_security_type = 'SHARE'
        AND sc.currency IS NOT NULL AND sc.currency = 'USD'
        AND s.factset_entity_id IS NOT NULL
        AND e.sector_code IS NOT NULL
        AND p.p_date IS NOT NULL
        AND (sc.fref_listing_exchange = 'NAS' OR sc.fref_listing_exchange = 'NYS')
        AND p.p_date >= '{self.start_date}'
        {sector_clause}

        ORDER BY fsym_id ASC, p_date DESC
        """

        print("get_price_data: running query...")
        adj = self.spark.sql(query)
        adj = adj.repartition(self.num_partitions)

        # --- Point-in-time market value filter: same replacement as get_fundamental_data() (see
        # _asof_carry_forward docstring) - ff_basic_cf is a current-only snapshot with no
        # historical date, so market value is reconstructed as price x shares outstanding, with
        # shares outstanding (ff_basic_qf.ff_com_shs_out, quarterly) carried forward causally to
        # each daily price row.
        # Keyed on 'fsym_regional_id' (added to the SELECT above), not the output 'fsym_id'
        # alias (s.fsym_id, security-level) - ff_basic_qf.fsym_id is the REGIONAL id (see
        # get_fundamental_data's own join: sm.fsym_regional_id = fun.fsym_id), so the join keys
        # must agree on the regional id, not the security id 'fsym_id' here is. Using 'fsym_id'
        # previously joined a security id against a regional-keyed table, which almost never
        # matches, leaving pit_market_value null for nearly every row and the
        # `> market_value_threshold` filter below dropping the entire dataset.
        print("get_price_data: reconstructing point-in-time market value (price x shares)...")
        # ff_basic_qf.ff_com_shs_out is quarterly-cadence, same as fundamentals reporting - using
        # the same 183-day (~2 reporting cycles) bound as interpolate_missing_values's fundamentals
        # forward-fill, by analogy (not an explicitly separately-specified number - override if a
        # tighter bound is wanted here specifically).
        adj = self._asof_carry_forward(
            adj, 'fsym_regional_id', 'p_date',
            "SELECT fsym_id, date, ff_com_shs_out FROM ff_basic_qf",
            'fsym_id', 'date', 'ff_com_shs_out', '_asof_shares', max_staleness_days=183
        )
        adj = adj.withColumn(
            'pit_market_value', col('p_price') * col('_asof_shares')
        ).filter(col('pit_market_value') > self.market_value_threshold).drop('fsym_regional_id', '_asof_shares', 'pit_market_value')

        # Adjust for splits
        print("Adjusting prices for splits...")
        adj = adj.withColumn("temp_split_factor", when(adj.p_date == adj.p_split_date, lit(adj.p_split_factor)).otherwise(lit(1.0)))
        adj = adj.withColumn("cum_split_factor", lit(0.0))  # Placeholder for cumulative split factor
        window_spec = Window.partitionBy('fsym_id').orderBy('p_date')
        adj = adj.withColumn("cum_split_factor", F.exp(F.sum(F.log("temp_split_factor")).over(window_spec)))
        
        # Split-adjust the price: P*_t, raw price rebased onto a consistent,
        # current-share-count basis. Unaffected by dividends.
        adj = adj.withColumn("adj_price", adj.p_price / adj.cum_split_factor)

        # Incorporate cash dividends as a total-return adjustment. A dividend is cash
        # paid to the shareholder on top of the (post-ex-date, lower) price - it must
        # never be subtracted from price, since that double-counts the value the
        # market already removed from the quote on the ex-date. The correct one-period
        # total return on the ex-dividend date is (P*_t + D*_t) / P*_{t-1}, i.e. a
        # MULTIPLICATIVE step of (1 + D*_t / P*_t) folded into the cumulative factor
        # exactly once, on the ex-date, and left unchanged on every later date (no
        # reversal/rebound the next day). D*_t/P*_t is scale-invariant to
        # cum_split_factor (both numerator and denominator carry the same factor), so
        # it's computed directly off the raw price/dividend rather than the
        # split-adjusted ones. Genuine spinoffs (p_divs_s_spinoff = 1) are excluded -
        # a spinoff distributes shares of another company, not cash, so this
        # cash-total-return treatment doesn't apply to it; only ordinary cash
        # dividends (p_divs_s_spinoff = 0) feed this step. Confirmed correct against
        # real Apple Inc. (fsym MH33D6-R) price/split/dividend history, including the
        # 2020-08-31 4-for-1 split and the 2020-05-08 $0.82 ex-dividend.
        print("Adjusting prices for cash dividends (total-return step)...")
        is_cash_dividend = (
            (adj.p_date == adj.p_divs_exdate) & (adj.p_divs_pd > 0)
            & (F.coalesce(adj.p_divs_s_spinoff, lit(0)) == 0)
        )
        adj = adj.withColumn(
            "temp_dividend_factor",
            when(is_cash_dividend, lit(1.0) + (adj.p_divs_pd / adj.p_price)).otherwise(lit(1.0))
        )
        window_spec_div = Window.partitionBy('fsym_id').orderBy('p_date').rowsBetween(Window.unboundedPreceding, Window.currentRow)
        adj = adj.withColumn("cum_dividend_factor", F.exp(F.sum(F.log("temp_dividend_factor")).over(window_spec_div)))

        # Fully Adjusted Price
        print('Calculating fully adjusted price..')
        adj = adj.withColumn("adj_price", adj.adj_price * adj.cum_dividend_factor)

        adj.cache()
        print("get_price_data: done (lazy - materializes on the first action).")
        
        return adj    
    
    def align_data(self, fundamental_df: DataFrame, price_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Check alignment between prices and fundamental features"""
        print("align_data: starting...")
        # Cache to avoid recomputation
        fundamental_df.cache()
        price_df.cache()

        # Get the minimum and maximum dates for fundamentals and price data
        min_max_fundamentals = fundamental_df.groupBy("fsym").agg(F.min("date").alias("min_fun"), F.max("date").alias("max_fun"))
        min_max_prices = price_df.groupBy("fsym_id").agg(F.min("p_date").alias("min_adj"), F.max("p_date").alias("max_adj"))

        # Align data based on fsym (renamed fsym from fsym_id) and dates
        min_max_dates = min_max_fundamentals.join(min_max_prices.withColumnRenamed('fsym_id', 'fsym'), 'fsym', 'inner')

        # Get the maximum of the minimum dates and the minimum of the maximum dates
        min_max_dates = min_max_dates.withColumn('abs_min', F.greatest(col('min_fun'), col('min_adj')))
        min_max_dates = min_max_dates.withColumn('abs_max', F.least(col('max_fun'), col('max_adj')))

        # Keep entries where abs_max is greater than or equal to abs_min
        min_max_dates = min_max_dates.withColumn('keep', when(col('abs_max') >= col('abs_min'), lit(1)).otherwise(lit(0)))
        min_max_dates = min_max_dates.filter(col('keep') == 1).drop('keep', 'min_fun', 'max_fun', 'min_adj', 'max_adj')

        # Filter the original dataframes to include only the existing data (based on fsym ids
        aligned_fundamental_df = fundamental_df.join(min_max_dates.select('fsym'), 'fsym', 'inner')
        aligned_price_df = price_df.join(min_max_dates.select('fsym').withColumnRenamed('fsym', 'fsym_id'), 'fsym_id', 'inner')

        # Repartition and cache for efficiency
        aligned_fundamental_df = aligned_fundamental_df.repartition(self.num_partitions)
        aligned_fundamental_df.cache()

        aligned_price_df = aligned_price_df.repartition(self.num_partitions)
        aligned_price_df.cache()
        
        del min_max_dates, fundamental_df, price_df

        print("align_data: done.")
        return aligned_fundamental_df, aligned_price_df
    
    def count_columns(self, df: DataFrame) -> int:
        """Count the number of columns in any DataFrame, useful to check feature progress"""
        return len(df.columns)
    
    def count_companies(self, df: DataFrame) -> int:
        """Count the number of unique fsym_ids (companies) in a DataFrame"""
        return df.select("fsym_id").distinct().count()

    @staticmethod
    def eda_summary(df: DataFrame, label: str = "", target_col: str = None, top_n_missing: int = 10) -> dict:
        """
        Print a compact EDA snapshot for a DataFrame: row count, feature count,
        dtype breakdown, missing-value rates for numeric features, and target
        variable stats (if a target column is given).

        Args:
        - df: Input DataFrame.
        - label: Short name for this pipeline step, printed in the summary header.
        - target_col: Optional numeric column to summarize (mean/std/min/max), e.g. 'monthly_return'.
        - top_n_missing: How many of the highest-missing-rate features to list.

        Returns:
        - dict with 'rows' and 'features' counts, so callers can build before/after comparisons.
        """
        num_rows = df.count()
        num_cols = len(df.columns)

        dtype_counts = {}
        for _, dtype in df.dtypes:
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        print(f"--- EDA summary: {label} ---")
        print(f"Rows: {num_rows} | Features (columns): {num_cols}")
        print(f"Dtype breakdown: {dtype_counts}")

        numeric_cols = [c for c, dtype in df.dtypes if dtype in ('int', 'bigint', 'float', 'double')]
        if numeric_cols and num_rows > 0:
            # Chunked into batches of 30 columns per select - a single select spanning 200+
            # columns can stall Spark's codegen (same mitigation as
            # ConsensusFeatureSelector._compute_diagnostics / Utils.add_temporal_features).
            # Matters now that add_temporal_features can push numeric_cols well past 100.
            missing_counts = {}
            EDA_CHUNK_SIZE = 30
            for _i in range(0, len(numeric_cols), EDA_CHUNK_SIZE):
                _chunk = numeric_cols[_i:_i + EDA_CHUNK_SIZE]
                _chunk_counts = df.select(
                    [F.sum(F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1).otherwise(0)).alias(c) for c in _chunk]
                ).collect()[0].asDict()
                missing_counts.update(_chunk_counts)
            missing_pct = {c: missing_counts[c] / num_rows for c in numeric_cols}
            # NOTE: `sum` is shadowed by pyspark.sql.functions.sum (imported above as `sum`),
            # so the plain builtin is used explicitly here instead of calling sum(...) directly.
            total_missing = 0.0
            for v in missing_pct.values():
                total_missing += v
            avg_missing = total_missing / len(missing_pct)
            top_missing = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:top_n_missing]
            print(f"Average missing-value rate across {len(numeric_cols)} numeric features: {avg_missing:.2%}")
            print(f"Top {top_n_missing} features by missing rate: {[(c, f'{p:.1%}') for c, p in top_missing]}")

        if target_col and target_col in df.columns:
            stats = df.select(
                F.mean(target_col).alias('mean'), F.stddev(target_col).alias('std'),
                F.min(target_col).alias('min'), F.max(target_col).alias('max')
            ).collect()[0].asDict()
            print(f"Target '{target_col}' stats: {stats}")

        print(f"--- end EDA summary: {label} ---\n")
        return {'rows': num_rows, 'features': num_cols}

    @staticmethod
    def replace_price_outliers(df: DataFrame, column: str, std: int = 5, window_size: int = 30) -> DataFrame:
        """
        Replace price anomalies with the causal trailing moving average.

        Causal by construction: both anomaly detection and the replacement value only ever
        depend on the current row and earlier rows (same or lower p_date) within each fsym_id's
        own series - no future observation is used. This used to replace a flagged anomaly with
        the average of the previous AND NEXT value (F.lead), which leaked a future price into a
        past/current row; that average is now the already-causal trailing `moving_avg` computed
        below, which was already being computed for anomaly detection but wasn't being reused
        for the replacement itself.

        Parameters:
        df (DataFrame): The input DataFrame containing the price data.
        column (str): The name of the column targeted
        std (int, optional): The number of standard deviations to use for identifying anomalies. Default is 4.
        window_size (int, optional): The number of look back days used in the moving average and standard deviation calculation. Default is 30.

        Returns:
        DataFrame: The smoothed DataFrame with flags for anomalies.
        """

        print(f"replace_price_outliers: starting on column '{column}'...")
        # Define a rolling window (trailing, causal) for calculating moving average and standard deviation
        roll_win = Window.partitionBy('fsym_id').orderBy('p_date').rowsBetween(-window_size, 0)
        # Define a window for accessing the previous value (used only for the negative-price fallback)
        win = Window.partitionBy('fsym_id').orderBy('p_date')

        # Calculate the moving average and moving standard deviation (add them as columns)
        df = df.withColumn("moving_avg", F.avg(F.col(column)).over(roll_win))
        df = df.withColumn("moving_stddev", F.stddev(F.col(column)).over(roll_win))

        # Calculate the absolute deviation from the moving average
        df = df.withColumn("deviation", F.abs(F.col(column) - F.col("moving_avg")))

        # Flag anomalies based on the deviation exceeding the specified number of standard deviations
        df = df.withColumn(f"{column}_is_anomaly", F.when(F.col("deviation") > std * F.col("moving_stddev"), True).otherwise(False))

        # Get previous value of the column (causal fallback for the negative-price case below)
        df = df.withColumn(f"{column}_prev", F.lag(F.col(column), 1).over(win))

        # Replace anomalies with the causal trailing moving average (never a future value)
        df = df.withColumn("smoothed_" + column, F.when(F.col(f"{column}_is_anomaly"), F.col("moving_avg"))
                                        .otherwise(F.col(column)))

        # Replace negative prices with the previous non-negative value
        df = df.withColumn("smoothed_" + column, F.when(F.col("smoothed_" + column) < 0, F.col(f"{column}_prev")).otherwise(F.col("smoothed_" + column)))

        print("replace_price_outliers: done.")
        return df
    
    def count_dates_per_stock(self, df: DataFrame) -> DataFrame:
        """
        Count the number of dates recorded for each stock (fsym).

        Args:
        - df: Input DataFrame.

        Returns:
        - DataFrame: DataFrame with the count of dates recorded for each stock.
        """
        count_dates = df.groupBy("fsym").agg(F.countDistinct("date").alias("date_count"))
        return count_dates
    
    @staticmethod
    def drop_non_numeric_columns(df: DataFrame, keep_columns: list[str] = ['factset_sector_desc', 'fsym_id', 'fsym', 'date']) -> DataFrame:
        """
        Drop non-numeric columns from the DataFrame except for specified columns.

        Args:
        - df: Input DataFrame.
        - keep_columns: List of non-numeric columns to keep.

        Returns:
        - DataFrame: DataFrame with non-numeric columns dropped.
        """
        print("drop_non_numeric_columns: starting...")
        features_before = len(df.columns)
        print(f"drop_non_numeric_columns params: keep_columns={keep_columns}")
        # Identify non-numeric columns
        non_numeric_cols = [col for col, dtype in df.dtypes if dtype not in ('int', 'float', 'double') and col not in keep_columns]

        # Drop non-numeric columns except the specified ones
        df = df.drop(*non_numeric_cols)
        features_after = len(df.columns)
        print(f"drop_non_numeric_columns: features before={features_before}, after={features_after} (rows unaffected - column drop only)")
        print(f"drop_non_numeric_columns: done, dropped {len(non_numeric_cols)} columns: {non_numeric_cols}")

        return df, non_numeric_cols

    @staticmethod
    def drop_na_features(df: DataFrame, drop_threshold: float = 0.3) -> tuple[DataFrame, list[str], list[str]]:
        """
        Drop numeric features (columns) with a high percentage of missing values and output non-numeric features.

        Args:
        - df: Input DataFrame.
        - drop_threshold: Percentage of missing values above which a column should be dropped.

        Returns:
        - Cleaned DataFrame with numeric features.
        - List of dropped numeric features.
        - List of non-numeric features.
        """

        print("drop_na_features: starting...")
        print(f"drop_na_features params: drop_threshold={drop_threshold} (features with > {drop_threshold:.0%} missing values are dropped)")
        print(f"Number of columns pre-drop: {len(df.columns)}")
        cols_to_clean = [col for col in df.columns if col not in ['date', 'fsym']]

        # Separate numeric and non-numeric columns
        numeric_cols = [col for col in cols_to_clean if isinstance(df.schema[col].dataType, (FloatType, DoubleType, IntegerType, LongType, DecimalType))]
        non_numeric_cols = [col for col in cols_to_clean if col not in numeric_cols]

        # Count the null and NaN values and create a dictionary to associate for each feature
        counts = df.select(
            [F.sum(F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1).otherwise(0)).alias(c) for c in numeric_cols]
        )
        counts_dict = counts.collect()[0].asDict()

        num_rows = df.count()
        features_to_drop = []
        perc_missing = []

        for col in numeric_cols:
            null_count = counts_dict[col]
            perc_null = null_count / num_rows

            if perc_null > drop_threshold:
                features_to_drop.append(col)
            else:
                perc_missing.append(col)

        df = df.drop(*features_to_drop)
        print(f"Number of columns post-drop: {len(df.columns)} (rows unaffected at {num_rows} - column drop only)")
        print(f"drop_na_features: dropped {len(features_to_drop)} features: {features_to_drop}")
        print("drop_na_features: done.")

        return df, features_to_drop, non_numeric_cols

    def check_feature_frequency(self, df: DataFrame) -> dict:   
        # NOT RUNNING FOR NOW 
        # RESULTS: 15 features skipped and Feature Frequencies Count: {'monthly': 0, 'quarterly': 585, 'annually': 0}
        """
        Check the frequency of data points for each feature in the DataFrame.

        Args:
        - df: Input DataFrame.

        Returns:
        - frequency_map: Dictionary mapping each feature to its frequency.
        """
        features = [col for col in df.columns if col not in ['date', 'fsym']]
        frequency_map = {}
        frequency_count = {'monthly': 0, 'quarterly': 0, 'annually': 0}
        
        # Calculate the date difference for each 'fsym'
        window_spec = Window.partitionBy('fsym').orderBy('date')
        df_diff = df.withColumn('date_diff', datediff('date', lag('date', 1).over(window_spec)))
        
        total_features = len(features)
        
        for idx, feature in enumerate(features, start=1):
            
            print(f"Processing feature {idx}/{total_features}")

            # Calculate the most common interval for each feature
            feature_df = df_diff.select('fsym', 'date_diff').where(df_diff[feature].isNotNull())
            common_interval_row = feature_df.groupBy('date_diff').count().orderBy('count', ascending=False).first()
            
            if common_interval_row:
                common_interval = common_interval_row['date_diff']

                # Determine frequency based on the common interval
                if common_interval < 45:
                    frequency = 'monthly'
                elif common_interval < 135:
                    frequency = 'quarterly'
                else:
                    frequency = 'annually'

                frequency_map[feature] = frequency
                frequency_count[frequency] += 1
            
            else:
                print(f"No data found for feature {feature}, skipping it.")
                
        return frequency_count

    @staticmethod
    def interpolate_missing_values(df: DataFrame, columns: list[str], max_staleness_days: int = 183) -> DataFrame:
        """
        Fill infinite and missing values causally: carry forward the most recent non-null,
        non-infinite value seen so far for that fsym (ordered by date). A row's filled value
        never depends on any later date's value.

        This used to fill a null/inf with the AVERAGE of the previous and next values
        (F.lead(..., 1)), which leaked a future quarter's fundamentals into a past/current row.
        That lead-based pass has been removed; the causal last-known-value carry-forward that
        was already present as a second pass (for values still null at the start of a series)
        is now the sole fill mechanism.

        max_staleness_days bounds how long a carried-forward value may go without a fresh
        non-null source observation before it's treated as missing again - reporting cadence is
        irregular across companies (see add_temporal_features's own note on this), so the bound
        is in calendar days, not report counts. Default 183 (~2 quarterly reporting cycles):
        most companies in this universe (USD, NASDAQ/NYSE-listed) report roughly every ~91 days,
        so 183 gives one full cycle of slack for a late or skipped filing before a value a
        company stopped disclosing years ago keeps reading as present today.

        Args:
        - df: Input DataFrame.
        - columns: List of columns to interpolate.
        - max_staleness_days: staleness bound described above.

        Returns:
        - DataFrame: DataFrame with infinite and missing values filled causally.
        """
        print("Starting causal fill of infinite and missing values...")
        # Non-target columns (e.g. factset_sector_desc, fsym_id) must survive this function -
        # downstream code (GAPreprocessing) hard-requires factset_sector_desc on fundamentals_df.
        passthrough_cols = [c for c in df.columns if c not in ['fsym', 'date'] + columns]
        win = Window.partitionBy('fsym').orderBy('date')
        rolling_win = win.rowsBetween(Window.unboundedPreceding, 0)

        # Convert +/-inf to null so both are handled by the same causal fill below.
        inf_to_null_exprs = [
            when(
                (col(c) == float('inf')) | (col(c) == float('-inf')),
                lit(None)
            ).otherwise(col(c)).alias(c)
            for c in columns
        ]
        df = df.select('fsym', 'date', *passthrough_cols, *inf_to_null_exprs)

        # Carry forward the most recent non-null value seen so far for that fsym (causal:
        # rowsBetween(unboundedPreceding, 0) only ever looks at the current row and earlier
        # rows), plus a companion "date of that source observation" per column - a non-null row's
        # own source is itself (0 days stale), so the staleness check below is a no-op for rows
        # that were never actually missing. Values still null at the very start of a series (no
        # prior observation exists) remain null either way.
        fill_exprs = []
        source_date_exprs = []
        for c in columns:
            fill_exprs.append(F.last(col(c), ignorenulls=True).over(rolling_win).alias(c))
            source_date_exprs.append(
                F.last(when(col(c).isNotNull(), col('date')), ignorenulls=True).over(rolling_win).alias(f'_src_date_{c}')
            )

        df = df.select('fsym', 'date', *passthrough_cols, *fill_exprs, *source_date_exprs)

        # Null out any filled value whose own source observation is older than
        # max_staleness_days relative to the current row's date - a company that stopped
        # disclosing a feature long ago should stop reading as "present today."
        staleness_exprs = [
            when(
                datediff(col('date'), col(f'_src_date_{c}')) > max_staleness_days, lit(None)
            ).otherwise(col(c)).alias(c)
            for c in columns
        ]
        df = df.select('fsym', 'date', *passthrough_cols, *staleness_exprs)

        print("Causal fill completed!")

        return df

    @staticmethod
    def add_temporal_features(df: DataFrame, feature_cols: list[str], lag_periods: list[int] = [1],
                               window_sizes: list[int] = [3]) -> Tuple[DataFrame, list[str]]:
        """
        Add lag/delta/growth/mean/std variants of each feature in feature_cols as extra
        candidate columns, computed report-to-report (Window.partitionBy('fsym').orderBy('date')
        - the same per-report grain as _asof_carry_forward/interpolate_missing_values), NOT on
        the monthly as-of-expanded table feature_selection_dataset builds later. Doing it here
        avoids the degenerate case of lagging/differencing a value that's mostly forward-filled
        across months between reports - "1 report ago" is the previous report for that fsym,
        regardless of the actual calendar gap (reporting cadence is irregular across companies).

        Every variant is backward-looking only (lag() never looks ahead), so - like
        interpolate_missing_values - this is safe to run once, globally, before any walk-forward
        split exists: a row's lag/delta/growth/mean/std value never depends on a later report.

        For each feature `c` and each lag period `k` in lag_periods:
        - {c}_lag{k}:    value k reports ago                       (pandas: c.shift(k))
        - {c}_delta{k}:  c - lag_k(c)                               (pandas: c.diff(k))
        - {c}_growth{k}: (c - lag_k(c)) / lag_k(c), with the same near-zero-denominator floor
          used by the GA's '/' operator (cell 19's combine_dataframes - fundamentals ratios can
          sit at exactly 0)
        For each window size `w` in window_sizes (trailing w reports, inclusive of the current one):
        - {c}_mean{w}: trailing mean. {c}_std{w}: trailing sample std (needs >= 2 non-null values
          in the window, so it's null for at least one more leading report than the others).

        New columns are added in batches of ~30 per .select() call rather than one single wide
        select spanning feature_cols * 5 columns - same "wide Spark aggregations can hang"
        mitigation already used by ConsensusFeatureSelector._compute_diagnostics (ChunkSIZE=30) -
        a single select/agg spanning 200+ columns can stall Spark's codegen (Java's 64KB
        per-method bytecode cap), independent of row count.

        Args:
        - df: fundamentals DataFrame, one row per (fsym, date=report) - should be the causally
          cleaned frame (i.e. called after interpolate_missing_values), not raw.
        - feature_cols: base feature columns to generate temporal variants for.
        - lag_periods: which k's to generate lag/delta/growth for.
        - window_sizes: which trailing window sizes to generate mean/std for.

        Returns:
        - (DataFrame, list[str]): df with the new temporal columns appended (originals
          untouched), and the list of new column names (for the caller to fold into its own
          candidate_feature_columns list - see the driver cell).
        """
        print(f"add_temporal_features: generating lag/delta/growth (k={lag_periods}) and "
              f"mean/std (window={window_sizes}) for {len(feature_cols)} features...")
        win = Window.partitionBy('fsym').orderBy('date')

        exprs = []
        for c in feature_cols:
            for k in lag_periods:
                lag_c = lag(col(c), k).over(win)
                exprs.append((f'{c}_lag{k}', lag_c))
                exprs.append((f'{c}_delta{k}', col(c) - lag_c))
                exprs.append((
                    f'{c}_growth{k}',
                    (col(c) - lag_c) / when(lag_c == 0, lit(1e-6)).otherwise(lag_c)
                ))
            for w in window_sizes:
                rolling_win = win.rowsBetween(-(w - 1), 0)
                exprs.append((f'{c}_mean{w}', F.avg(col(c)).over(rolling_win)))
                exprs.append((f'{c}_std{w}', F.stddev(col(c)).over(rolling_win)))

        TEMPORAL_CHUNK_SIZE = 30
        for i in range(0, len(exprs), TEMPORAL_CHUNK_SIZE):
            batch = exprs[i:i + TEMPORAL_CHUNK_SIZE]
            df = df.select('*', *[expr.alias(name) for name, expr in batch])

        new_cols = [name for name, _ in exprs]
        print(f"add_temporal_features: added {len(new_cols)} temporal columns "
              f"({len(feature_cols)} features x ({len(lag_periods)} lag periods x 3 ops + "
              f"{len(window_sizes)} window sizes x 2 ops)).")
        return df, new_cols

    @staticmethod
    def remove_static_features(df: DataFrame, feature_columns: list[str], sd_rms_threshold: float = 0.10) -> list[str]:
        """
        Drop features (columns) that are near-constant within an individual company's own
        history, judged by that company's own scale - not by comparing to a variance pooled
        across every company.

        Args:
        - df: Input DataFrame.
        - feature_columns: List of feature columns to check.
        - sd_rms_threshold: a company is flagged static on a feature when
          SD_company(x) / RMS_company(x) < sd_rms_threshold, where RMS_company(x) =
          sqrt(mean(x_t^2)) over that company's own rows. Algebraically, RMS^2 = SD^2 + mean^2,
          so this ratio is a bounded [0, 1) reparameterization of the ordinary coefficient of
          variation SD/|mean| (0.10 here corresponds to CV of about 10%) - chosen over raw CV
          specifically because CV blows up/is undefined as a company's own mean approaches zero
          (common for something like net income, which crosses zero), while SD/RMS stays
          well-behaved there (RMS folds the mean and spread together, so it only vanishes if
          every value is exactly zero).

        Returns:
        - List of features to drop.

        **Per-company scale, not a pooled global one - this is the actual fix, not the threshold
        value.** The previous version compared each company's local variance to ONE variance
        pooled across every company's rows, which is dominated by CROSS-COMPANY scale
        differences, not genuine temporal informativeness: `variance()` scales with the square
        of a value's magnitude, so a handful of large companies' huge absolute swings can make
        the pooled variance enormous, and dividing a small company's (perfectly normal,
        proportionally-sized) absolute variance by that pooled figure makes it look "static"
        purely because it's a small company - confirmed directly on real data (`ff_sales` in a
        real fold: the largest companies scored ratios of 0.57-1.4, comfortably above the old
        0.01 cutoff, while the majority of smaller companies scored near zero on the exact same,
        genuinely-varying feature). SD/RMS is computed entirely from one company's own values, so
        it carries no cross-company scale confound at all.

        No company sampling: every company present in `df` is checked (unchanged from before -
        see git history for why the sampling this used to do was removed: a fixed random seed on
        `.sample().limit()` looked reproducible but Spark's `.limit()` isn't actually
        deterministic without an explicit `.orderBy()`, so two folds training on identical rows
        could reach different static-feature verdicts by chance).
        """
        print("Starting removal of static features...")
        features_before = len(feature_columns)
        print(f"remove_static_features params: sd_rms_threshold={sd_rms_threshold} (SD/RMS ratio, per company)")

        working_df = df.select('fsym', *feature_columns).fillna(0, subset=feature_columns).cache()

        # Per-company SD and RMS of every feature, one row per company. Chunked at CHUNK_SIZE
        # columns/aggregation (same convention as the rest of this file's wide Spark aggregations
        # - see the "Wide Spark aggregations can hang" pitfall): feature_columns can run into the
        # hundreds, and this aggregation spans every company in the dataset.
        CHUNK_SIZE = 30
        local_stats_by_company: dict = {}  # fsym -> {feature: (sd, rms)}
        for i in range(0, len(feature_columns), CHUNK_SIZE):
            chunk = feature_columns[i:i + CHUNK_SIZE]
            agg_exprs = []
            for c in chunk:
                agg_exprs.append(variance(col(c)).alias(f"{c}__var"))
                agg_exprs.append(F.avg(F.pow(col(c), 2)).alias(f"{c}__msq"))
            for row in working_df.groupBy('fsym').agg(*agg_exprs).collect():
                entry = local_stats_by_company.setdefault(row['fsym'], {})
                for c in chunk:
                    var = row[f"{c}__var"]
                    msq = row[f"{c}__msq"]
                    sd = var ** 0.5 if var is not None and var >= 0 else None
                    rms = msq ** 0.5 if msq is not None and msq >= 0 else None
                    entry[c] = (sd, rms)

        n_companies = len(local_stats_by_company)
        stock_cols = {key: 0 for key in feature_columns}
        for company_stats in local_stats_by_company.values():
            for c in feature_columns:
                sd, rms = company_stats[c]
                # A company with fewer than 2 rows has an undefined (null) sample variance, so SD
                # is None - matches pandas' .var() returning NaN in that case, which also never
                # satisfies the threshold check below.
                if sd is None:
                    continue
                # RMS == 0 means every one of this company's values is exactly zero - definitely
                # static, and avoids dividing by (near-)zero (SD is also exactly 0 in this case,
                # so the ratio is 0/0 without this guard).
                if rms is None or rms <= 0:
                    stock_cols[c] += 1
                elif (sd / rms) < sd_rms_threshold:
                    stock_cols[c] += 1

        features_to_drop = [key for key in stock_cols if stock_cols[key] > n_companies / 2]
        print(f"Features to drop: {features_to_drop}")
        working_df.unpersist()

        # Drop the static features from the DataFrame
        df = df.drop(*features_to_drop)
        features_after = len(df.columns)
        print(f"remove_static_features: feature_columns checked before={features_before}, features dropped={len(features_to_drop)}, "
              f"companies checked={n_companies}, columns remaining in df after={features_after} (rows unaffected - column drop only)")

        return df, features_to_drop
    
    @staticmethod
    def calculate_monthly_returns(df: DataFrame, max_abs_return: float = 5.0) -> DataFrame:
        """
        Calculate the monthly return for each stock as month-end-to-month-end: the change from
        the last traded price of the PREVIOUS calendar month to the last traded price of THIS
        calendar month - the standard convention (CRSP-style, Fama-French, most asset-pricing
        literature), which is what lets consecutive monthly returns compound cleanly into the
        true buy-and-hold cumulative return over multiple months (month t's close is exactly
        month t+1's starting price, so there's no gap between them). An earlier version computed
        a within-month return instead (first price observed in month t to last price observed in
        that SAME month t), which silently dropped whatever the stock did between the last
        trading day of month t-1 and the first trading day of month t from every month's return -
        not a standard definition, and not exact for compounding/backtesting purposes.

        Args:
        - df: Input DataFrame containing the price data with columns 'fsym_id', 'p_date', 'adj_price'.
        - max_abs_return: sanity bound on |monthly_return|. A real return can never be below -1
          (-100%, can't lose more than the whole investment) - anything below that is always
          corrupt data (e.g. a prior month-end price that's near-zero/negative from a bad tick or
          an unhandled delisting). Returns above max_abs_return (default 5.0, i.e. +500%) are
          overwhelmingly also data glitches (a near-zero starting price causing a huge "recovery"
          the following month) rather than genuine organic moves, and get dropped the same way.
          This only filters the small, already-aggregated monthly_returns table, so it's cheap.

        Gap guard: naively lagging by one row (ordered by year/month) would return whichever row
        happens to precede this one, not necessarily the immediately preceding CALENDAR month -
        if a company has a genuine gap (no price rows at all in some month), that would silently
        compute a two-(or more-)month return and label it as one month's return. Guarded
        explicitly below via an integer month_index (year*12+month, robust across a
        December->January rollover without special-casing it): a row's return is only computed
        when its lagged row's month_index is exactly one less than its own; otherwise (including
        a company's very first observed month, which has no prior row at all) the return is left
        null and the row is dropped, same as any other invalid return.

        Returns:
        - DataFrame with monthly returns.
        """
        print("calculate_monthly_returns: starting...")
        df = df.withColumn('year', F.year('p_date'))
        df = df.withColumn('month', F.month('p_date'))

        # One row per (fsym_id, year, month): that month's closing (last-traded) price.
        # ignorenulls=True: without it, F.last() defaults to ignorenulls=False, so a null
        # adj_price on the chronologically LAST row of the month (a data glitch, a halted-
        # trading day, etc.) would null out month_end_price for the WHOLE month, silently
        # dropping that month's (and the following month's, since this also feeds
        # prev_month_end_price) return - even though a perfectly good price existed earlier
        # in the same month. ignorenulls=True correctly falls back to the last non-null price.
        window_spec = Window.partitionBy('fsym_id', 'year', 'month').orderBy('p_date').rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        df = df.withColumn('month_end_price', F.last('adj_price', ignorenulls=True).over(window_spec))
        monthly_prices = df.select('fsym_id', 'factset_sector_desc', 'year', 'month', 'month_end_price') \
            .groupBy('fsym_id', 'factset_sector_desc', 'year', 'month') \
            .agg(F.max('month_end_price').alias('month_end_price'))

        monthly_prices = monthly_prices.withColumn('month_index', F.col('year') * 12 + F.col('month'))
        company_window = Window.partitionBy('fsym_id').orderBy('month_index')
        monthly_prices = monthly_prices.withColumn('prev_month_end_price', F.lag('month_end_price').over(company_window))
        monthly_prices = monthly_prices.withColumn('prev_month_index', F.lag('month_index').over(company_window))

        # Calculate monthly return - only where the lagged row is exactly the immediately
        # preceding calendar month (see the gap-guard docstring above); F.when() with no
        # .otherwise() already defaults to null when the condition is false.
        monthly_returns = monthly_prices.withColumn(
            'monthly_return',
            F.when(
                F.col('prev_month_index') == F.col('month_index') - 1,
                (F.col('month_end_price') - F.col('prev_month_end_price')) / F.col('prev_month_end_price')
            )
        ).select('fsym_id', 'factset_sector_desc', 'year', 'month', 'monthly_return')

        rows_before_gap_filter = monthly_returns.count()
        monthly_returns = monthly_returns.filter(F.col('monthly_return').isNotNull())
        print(f"calculate_monthly_returns: dropped {rows_before_gap_filter - monthly_returns.count()} row(s) with "
              f"no immediately-preceding calendar month (a company's first observed month, or a genuine data gap)")

        # Sanity-filter implausible returns before they reach anything downstream (feature
        # selection, GA fitness, backtests) - see max_abs_return docstring above for why these
        # bounds are always data errors rather than genuine returns.
        rows_before_sanity = monthly_returns.count()
        impossible = monthly_returns.filter(F.col('monthly_return') < -1).count()
        implausible = monthly_returns.filter(F.col('monthly_return') > max_abs_return).count()
        monthly_returns = monthly_returns.filter(
            (F.col('monthly_return') >= -1) & (F.col('monthly_return') <= max_abs_return)
        )
        print(f"calculate_monthly_returns: sanity filter (max_abs_return={max_abs_return}) - "
              f"rows {rows_before_sanity} -> {monthly_returns.count()} "
              f"(dropped {impossible} impossible [<-100%] + {implausible} implausible [>{max_abs_return:.0%}] rows)")

        print("calculate_monthly_returns: done.")
        return monthly_returns
    
    @staticmethod
    def add_return_lag_features(monthly_returns_df: DataFrame, sector_col: str = 'factset_sector_desc') -> DataFrame:
        """
        Add prev_month_return and prev_month_sector_return as extra candidate feature columns
        on the monthly returns table. Both are pure backward-looking lags (previous calendar
        month only, per fsym), so this is safe to compute once, globally, before any train/eval
        split exists - same causal-safety argument as interpolate_missing_values.

        Args:
        - monthly_returns_df: DataFrame with columns fsym_id, sector_col, year, month, monthly_return.

        Returns:
        - DataFrame: monthly_returns_df with two extra columns added: prev_month_return (that
          fsym's own previous month's return) and prev_month_sector_return (the sector-wide
          average return from the previous month - never the current/target month, so this
          never leaks the row's own target into its own feature).
        """
        print("add_return_lag_features: starting...")

        sector_month_avg = monthly_returns_df.groupBy(sector_col, 'year', 'month') \
            .agg(F.avg('monthly_return').alias('avg_sector_return'))

        monthly_returns_df = monthly_returns_df.join(
            sector_month_avg, on=[sector_col, 'year', 'month'], how='left'
        )

        window_spec = Window.partitionBy('fsym_id').orderBy('year', 'month')
        monthly_returns_df = monthly_returns_df.withColumn('prev_month_return', lag('monthly_return').over(window_spec))
        monthly_returns_df = monthly_returns_df.withColumn('prev_month_sector_return', lag('avg_sector_return').over(window_spec))

        # avg_sector_return itself (current/target month) is only an intermediate value used to
        # build the lag above - drop it so it's never mistakenly used as a same-month feature.
        monthly_returns_df = monthly_returns_df.drop('avg_sector_return')

        print("add_return_lag_features: done.")
        return monthly_returns_df

    @staticmethod
    def calculate_industry_returns(df: DataFrame) -> DataFrame: # For use only when constructing the dataframe for experiment 2
        """ 
        Calculate industry returns of a specific industry that the stock 
        belongs to based on the factset_sector_desc column and add it as a 
        column to the dataframe.

        Args:
        - df: Input DataFrame containing the price data with columns 
              'fsym_id', 'factset_sector_desc', 'year', 'month', 'monthly_return'.

        Returns:
        - DataFrame with industry returns added.
        """
        print("calculate_industry_returns: starting...")
        # Define window to calculate industry returns
        industry_window = Window.partitionBy('factset_sector_desc', 'year', 'month')

        # Calculate the average monthly return for each industry
        df = df.withColumn('industry_return', F.avg('monthly_return').over(industry_window))

        # Ensure distinct rows
        industry_returns = df.select('fsym_id', 'factset_sector_desc', 'year', 'month', 'monthly_return', 'industry_return').distinct()

        print("calculate_industry_returns: done.")
        return industry_returns
    
    @staticmethod
    def count_total_infinity_values(df: DataFrame) -> None:
        """
        Count the total number of infinity values in all columns of a DataFrame, excluding specified columns.

        Parameters:
        df (DataFrame): The DataFrame to check.

        Returns:
        None: Prints the total number of infinity values in the DataFrame.
        """
        print("count_total_infinity_values: starting...")
        # List of columns to check for infinity values, excluding non-double type columns
        columns_to_check = [c for c in df.columns if isinstance(df.schema[c].dataType, DoubleType)]

        # Chunked into batches of 30 columns - see eda_summary's missing_counts for why (a
        # single wide select spanning 200+ columns can stall Spark's codegen). Matters now that
        # add_temporal_features can push columns_to_check well past 100.
        INF_CHUNK_SIZE = 30
        total_inf_count = 0
        for _i in range(0, len(columns_to_check), INF_CHUNK_SIZE):
            _chunk = columns_to_check[_i:_i + INF_CHUNK_SIZE]
            _chunk_row = df.select(
                [F.sum((col(c) == float('inf')).cast('int') + (col(c) == float('-inf')).cast('int')).alias(c) for c in _chunk]
            ).collect()[0]
            for c in _chunk:
                total_inf_count += (_chunk_row[c] or 0)

        print(f"Total number of infinity values in the given df: {total_inf_count}")
    
    @staticmethod
    def get_fsym_list(monthly_returns_df: DataFrame) -> List[str]:
        fsym_list = [row['fsym_id'] for row in monthly_returns_df.select('fsym_id').distinct().collect()]
        return fsym_list
    
    @staticmethod
    def get_fsym_from_sector(monthly_returns_df: DataFrame, sector: str) -> List[str]:
        fsym_list = [row['fsym_id'] for row in monthly_returns_df.filter(F.col('factset_sector_desc') == sector).select('fsym_id').distinct().collect()]
        return fsym_list
    
    @staticmethod
    def prepare_fsym_data(fundamentals_df: DataFrame, monthly_returns_df: DataFrame, fsym: str, lever: int = 1) -> DataFrame:
        """
        Generate a dataset for the specified fsym id for the genetic algorithm.

        Args:
        - fundamentals_df: DataFrame with fundamental data.
        - monthly_returns_df: DataFrame with monthly returns.
        - fsym: specified stock id
        - lever: Number of past quarters to include.

        Returns:
        - Prepared DataFrame for the genetic algorithm for the specified fsym id. Which contains:
            - number of past quarters of features as specified by the lever
            - current month's fsym returns
            - current month's fsym sector returns
            - next month's fsym returns (target to predict)
        """
        # Ensure the 'date' column exists
        if 'date' not in fundamentals_df.columns:
            print("The 'date' column is missing in the fundamentals DataFrame.")
            return None
        
        # Extract year and month from the date
        fundamentals_df = fundamentals_df.withColumn('year', year('date')).withColumn('month', month('date'))

        # Check the presence of the fsym in both DataFrames
        fsym_fundamentals_count = fundamentals_df.filter(F.col('fsym') == fsym).count()
        fsym_monthly_returns_count = monthly_returns_df.filter(F.col('fsym_id') == fsym).count()

        print(f"fsym: {fsym} - Count in Fundamentals DF: {fsym_fundamentals_count}")
        print(f"fsym: {fsym} - Count in Monthly Returns DF: {fsym_monthly_returns_count}")

        # Print the dates on which returns are computed from the monthly returns df
        print("Dates with returns computed from the monthly returns df:")
        monthly_returns_dates = monthly_returns_df.select('year', 'month').distinct().orderBy('year', 'month')
        monthly_returns_dates.show(truncate=False)

        # Print the dates on which features are computed from the fundamental df
        print("Dates with features computed from the fundamental df:")
        fundamental_dates = fundamentals_df.select('date').distinct().orderBy('date')
        fundamental_dates.show(truncate=False)

        # Filter fundamentals and returns data for the specific fsym
        fsym_fundamentals_df = fundamentals_df.filter(F.col('fsym') == fsym)
        fsym_monthly_returns_df = monthly_returns_df.filter(F.col('fsym_id') == fsym)

        # Print the schema to verify the 'date' column
        print("Schema of fsym_fundamentals_df:")
        fsym_fundamentals_df.printSchema()

        # Renaming columns to match the expected format
        fsym_monthly_returns_df = fsym_monthly_returns_df.withColumnRenamed('year', 'return_year').withColumnRenamed('month', 'return_month')

        # List the number of features for each specified date
        dates_to_check = [
            '2001-01-31', '2001-02-28', '2001-03-31'
        ]

        for date in dates_to_check:
            feature_counts = fsym_fundamentals_df.filter(F.col('date') == date).select(
                [F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in fsym_fundamentals_df.columns if c not in ['fsym', 'date', 'year', 'month']]
            ).collect()[0].asDict()

            non_null_count = 0
        for v in feature_counts.values():
            non_null_count += v
            print(f"Date: {date}, Non-null Feature Count: {non_null_count}")

        return fsym_fundamentals_df, fsym_monthly_returns_df
    
    @staticmethod
    def summarize_feature_reporting(fundamentals_df: DataFrame, fsym: str) -> None:
        """
        Summarize the reporting frequency of specific features in the fundamentals DataFrame for the specified fsym.

        Args:
        - fundamentals_df: DataFrame with fundamental data.
        - fsym: specified stock id

        Returns:
        - None (prints the dates when specific features are reported)
        """
        # List the specific features to check
        features_to_check = ['ff_bk_invest_tot', 'ff_gross_inc']

        # Filter fundamentals data for the specific fsym
        fsym_fundamentals_df = fundamentals_df.filter(F.col('fsym') == fsym)

        # Print the dates for the specified features
        for feature in features_to_check:
            print(f"Dates when {feature} is available:")
            fsym_fundamentals_df.filter(F.col(feature).isNotNull()).select('date').orderBy('date').show(truncate=False)

    @staticmethod
    def write_to_csv(df: DataFrame, title: str):
        # Convert title to a Path object
        path = Path(title)
        
        # Check if the directory exists and contains the CSV file
        if path.exists() and any(path.glob('*.csv')):
            # If the CSV file exist, overwrite the directory
            print(f"CSV file(s) already exist in {title}. Overwriting...")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(title)
        else:
            # If no CSV file exists, write to the directory
            print(f"No CSV file found in {title}. Writing new CSV file...")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(title)
        
        print("Converted to CSV successfully!")
    
    @staticmethod
    def delete_feature_columns(df: DataFrame, feature_columns: list[str]) -> DataFrame:
        """
        Delete specified feature columns from the DataFrame.
        
        Args:
        - df: Input DataFrame.
        - feature_columns: List of columns to delete.

        Returns:
        - DataFrame: DataFrame with specified columns deleted.
        """
        # Check if the columns exist in the DataFrame
        existing_columns = set(df.columns)
        columns_to_delete = [col for col in feature_columns if col in existing_columns]

        # Drop the specified columns
        df = df.drop(*columns_to_delete)

        return df

    @staticmethod
    def apply_selected_features(df: DataFrame, candidate_columns: list[str], final_features: list[str]) -> DataFrame:
        """
        Apply an externally-decided feature list to a DataFrame: drop every candidate column
        that isn't in final_features. This is a pure column-selection ("transform") step with
        no fitting of its own - it's the walk-forward counterpart to
        ConsensusFeatureSelector.run(), used to apply a fold's train-fit feature list to that
        same fold's validation/test rows without ever re-deriving the list from them.

        Args:
        - df: Input DataFrame (train or validation/test slice of a fold).
        - candidate_columns: The full candidate feature-column list that final_features was
          chosen from (i.e. whatever was passed into drop_na_features/remove_static_features/
          ConsensusFeatureSelector.run() for that fold).
        - final_features: The retained feature columns for that fold.

        Returns:
        - DataFrame: df with every candidate column not in final_features dropped.
        """
        features_to_remove = [c for c in candidate_columns if c not in final_features]
        return Utils.delete_feature_columns(df, features_to_remove)

    @staticmethod
    def feature_selection_dataset(fundamentals_df: DataFrame, monthly_returns_df: DataFrame) -> DataFrame:
        """
        Prepare the DataFrame for feature selection experiment by associating each
        fundamentals report with every month it is the most-recently-known report for - an
        as-of join, not just the single following month. A report becomes usable starting the
        month after fund_report_date (the report's true filing/availability date - see
        Utils.get_fundamental_data - not its fiscal period-end `date`) and stays in effect until
        superseded by that fsym's next report - or, for a stock's final report, through the
        last month present in monthly_returns_df.

        Args:
        - fundamentals_df: DataFrame with fundamental data, including a `fund_report_date`
          column (see Utils.get_fundamental_data).
        - monthly_returns_df: DataFrame with monthly returns (including prev_month_return /
          prev_month_sector_return, added upstream by add_return_lag_features).

        Returns:
        - DataFrame: one row per (fsym, target_year, target_month) with that month's
          most-recently-known fundamentals features and that month's return.
        """
        from datetime import date

        # Extract year and month from the date (kept for parity with the rest of the pipeline;
        # target_date/target_year/target_month below - not this - now drive the join).
        fundamentals_df = fundamentals_df.withColumn('year', year('date'))
        fundamentals_df = fundamentals_df.withColumn('month', month('date'))

        print('Expanding each report to every month it is the as-of most recent report for...')

        # Rename columns for joining, and find the last month present in the return series - a
        # stock's last fundamentals report carries forward through this month (rather than an
        # arbitrary constant), since there is no return data beyond it to align to anyway.
        monthly_returns_df = monthly_returns_df.withColumnRenamed('year', 'return_year')
        monthly_returns_df = monthly_returns_df.withColumnRenamed('month', 'return_month')

        max_ym_row = monthly_returns_df.select(
            (F.col('return_year') * 100 + F.col('return_month')).alias('ym')
        ).agg(F.max('ym').alias('max_ym')).collect()[0]
        max_return_year, max_return_month = divmod(max_ym_row['max_ym'], 100)
        as_of_cutoff_date = (
            date(max_return_year + 1, 1, 1) if max_return_month == 12
            else date(max_return_year, max_return_month + 1, 1)
        ) - timedelta(days=1)
        print(f'feature_selection_dataset: carrying the last report of any fsym forward through '
              f'{as_of_cutoff_date} (the last month present in monthly_returns_df)')

        # Per fsym, find the next report's point-in-time availability date (null for a stock's
        # most recent report) - ordering by `date` (fiscal period-end), which is inherently
        # monotonic per fsym by construction, rather than by fund_report_date itself (a real
        # filing date, occasionally noisy).
        fsym_order = Window.partitionBy('fsym').orderBy('date')
        fundamentals_df = fundamentals_df.withColumn('next_report_date', F.lead('fund_report_date').over(fsym_order))

        # Coverage window per report: starts the month after fund_report_date (the report isn't
        # usable until the month after it's actually known - see Utils.get_fundamental_data for
        # how fund_report_date is derived from the true filing date, not the fiscal period-end).
        # Ends the day before the NEXT report's own coverage_start - or the as-of cutoff date
        # above, for a stock's last report. Both boundaries are now on the same point-in-time
        # basis (fund_report_date), rather than mixing period-end and filing-date bases.
        fundamentals_df = fundamentals_df.withColumn('coverage_start', add_months(F.trunc(col('fund_report_date'), 'MM'), 1))
        fundamentals_df = fundamentals_df.withColumn(
            'coverage_end',
            when(col('next_report_date').isNotNull(),
                 F.date_sub(add_months(F.trunc(col('next_report_date'), 'MM'), 1), 1))
            .otherwise(lit(as_of_cutoff_date))
        )

        # Drop reports whose coverage window is empty (a report superseded within the same
        # month it was released, before it was ever usable).
        fundamentals_df = fundamentals_df.filter(col('coverage_start') <= col('coverage_end'))

        print('Repartitioning for efficiency...')
        # Repartition fundamentals_df to optimize the join - done before the explode below so
        # the shuffle happens on the smaller, pre-expansion, one-row-per-report frame.
        fundamentals_df = fundamentals_df.repartition(20, 'fsym').cache()

        # Explode each report's coverage window into one row per month it covers.
        fundamentals_df = fundamentals_df.withColumn(
            'target_date', F.explode(F.sequence(col('coverage_start'), col('coverage_end'), F.expr('interval 1 month')))
        )
        fundamentals_df = fundamentals_df.withColumn('target_date', F.trunc(col('target_date'), 'MM'))
        fundamentals_df = fundamentals_df.withColumn('target_year', year('target_date'))
        fundamentals_df = fundamentals_df.withColumn('target_month', month('target_date'))

        # Safety net: truncating two different reports' coverage windows to month-start can very
        # rarely collide on the same (fsym, target_year, target_month) at a transition boundary -
        # keep the row sourced from the more recent report for that month.
        dedup_window = Window.partitionBy('fsym', 'target_year', 'target_month').orderBy(col('date').desc())
        fundamentals_df = fundamentals_df.withColumn('_rn', F.row_number().over(dedup_window)) \
            .filter(col('_rn') == 1).drop('_rn', 'next_report_date', 'coverage_start', 'coverage_end', 'fund_report_date')

        # Repartition monthly_returns_df to optimize the join
        monthly_returns_df = monthly_returns_df.repartition(20, 'fsym_id').cache()

        broadcast_monthly_returns_df = broadcast(monthly_returns_df)

        print('Creating the final DataFrame...')

        # Single join across the whole range instead of looping per target_year and
        # unioning: target_year == return_year is already one of the join's own equality
        # conditions, so filtering to one year at a time and unioning the results is
        # mathematically identical to just doing the join once - the loop only added N
        # sequential jobs plus a deep, linear .union() chain, and this is the heaviest job
        # in the pipeline (see the driver cell).
        final_merged_df = fundamentals_df.join(
            broadcast_monthly_returns_df,
            (fundamentals_df.fsym == broadcast_monthly_returns_df.fsym_id) &
            (fundamentals_df.target_year == broadcast_monthly_returns_df.return_year) &
            (fundamentals_df.target_month == broadcast_monthly_returns_df.return_month),
            'inner'
        )

        feature_columns = [c for c in fundamentals_df.columns if c not in ['date', 'fsym', 'year', 'month', 'target_date', 'target_year', 'target_month', 'factset_sector_desc', 'fsym_id']]
        # target_date is kept in the output because the walk-forward split filters on it
        # directly, not on the fundamentals 'date'. prev_month_return/prev_month_sector_return
        # come from the returns side and are pulled through as ordinary candidate features.
        final_merged_df = final_merged_df.select('fsym', 'date', fundamentals_df['factset_sector_desc'], 'target_date', 'return_year', 'return_month',
                                               'monthly_return', 'prev_month_return', 'prev_month_sector_return', *feature_columns)

        # Drop rows with NaN values in the 'monthly_return' column
        final_merged_df = final_merged_df.dropna(subset=['monthly_return'])

        # Check for any remaining NaN values
        sum_na = final_merged_df.filter(F.col('monthly_return').isNull()).count()
        if sum_na > 0:
            print(f'Some dates do not align: {sum_na} missing return values in the dataset')

        return final_merged_df

print("Utils class defined.")