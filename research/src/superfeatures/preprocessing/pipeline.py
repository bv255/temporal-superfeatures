"""
Ported from `research/PreProcessing_test.ipynb` cell 13 — see `docs/RESTRUCTURING_TODO.md`
and the port plan in `docs/RESEARCH_STRUCTURE.md`. Extracted mechanically from the notebook
cell source (not retyped) to avoid transcription drift; the notebook remains the frozen parity
reference. Two kinds of edits were needed, both purely mechanical:

1. Names the cell relied on getting from the shared notebook kernel namespace (`os`, `F` from
   cell 0's imports) are now imported explicitly here.
2. `run_fold`'s original signature closed over three more kernel globals that only a live
   notebook run defines: `utils` (the driver cell's `Utils` instance), `spark`, and the
   `ConsensusFeatureSelector` class. A standalone module has no such implicit state, so all
   three are now explicit parameters - `utils`/`spark` required (no sensible default),
   `ConsensusFeatureSelector` defaulted to this module's own import so existing call sites
   that don't pass it behave identically. The function body itself is untouched: naming the
   new parameter `ConsensusFeatureSelector` (same as the class it defaults to) means every
   reference to that name inside the body resolves correctly with zero body edits.

`compute_fold_boundaries` (cell 11, the walk-forward year-boundary computation this module's
`run_fold` consumes) now lives in `evaluation/splits.py` instead - applying a fold's boundaries
to slice/screen/write its data is a distinct concern from computing the boundaries themselves.
"""

import os
from pyspark.sql import functions as F

import json
import calendar
from datetime import date

from .consensus import ConsensusFeatureSelector


def write_consensus_artifacts(consensus_result, output_dir: str):
    """
    Write ConsensusFeatureSelector diagnostic artifacts to output_dir. Refactor of the original
    single global "Writing consensus feature selection artifacts" cell into a callable, so it
    can be reused per fold instead of copy-pasted. Same convention as the original
    consensus_feature_selection_artifacts/: plain pandas/json writes, local disk (not HDFS).
    """
    os.makedirs(output_dir, exist_ok=True)
    consensus_result.raw_scores_table.to_csv(f"{output_dir}/raw_scores_table.csv", index=False)
    consensus_result.consensus_table.to_csv(f"{output_dir}/consensus_table.csv", index=False)
    consensus_result.corr_matrix.to_csv(f"{output_dir}/corr_matrix.csv")
    consensus_result.cluster_table.to_csv(f"{output_dir}/cluster_table.csv", index=False)
    consensus_result.rejection_log.to_csv(f"{output_dir}/rejection_log.csv", index=False)
    with open(f"{output_dir}/summary_counts.json", "w") as f:
        json.dump(consensus_result.summary_counts, f, indent=2)
    print(f"write_consensus_artifacts: wrote diagnostics to {output_dir}/")


def write_fold_metadata(output_dir: str, fold_info: dict):
    """Write fold_metadata.json: date boundaries (nominal + embargoed) and row/feature counts."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/fold_metadata.json", "w") as f:
        json.dump(fold_info, f, indent=2, default=str)
    print(f"write_fold_metadata: wrote {output_dir}/fold_metadata.json")


def run_fold(base_df, train_years, eval_years, eval_label, output_dir, candidate_feature_columns,
             utils, spark,
             embargo_months=1, sector_col='factset_sector_desc', target='monthly_return',
             random_seed=42, inner_val_years=None, terminal_cap=None,
             ConsensusFeatureSelector=ConsensusFeatureSelector):
    """
    Run one walk-forward fold end-to-end.

    train_years: (start_year, end_year) inclusive, both bounds on target_year.
    eval_years: (start_year, end_year) inclusive, the held-out target_year window (validation or
        test). Width 1 (start_year == end_year) is the original single-year-per-fold shape;
        wider windows cover that many consecutive years in one eval_label file instead.
    eval_label: 'validation' or 'test' - controls the output filename ({eval_label}.csv).
    inner_val_years: optional, (start_year, end_year) inclusive. When set (final-test folds
        only), carves an additional embargoed validation window out between train_years and
        eval_years - {output_dir}/train.csv covers only up to train_years[1], a new
        {output_dir}/validation.csv covers inner_val_years, and {output_dir}/{eval_label}.csv
        (eval_years) is written exactly as before. inner_val_years isn't new data - it's the
        same years that used to be the tail of the training window, just carved out and
        embargoed the same way train's own tail is. This exists so a downstream GA's
        per-generation fitness search can be scored against validation.csv instead of
        {eval_label}.csv - the eval_label file then stays genuinely untouched by any decision
        (feature selection here, GA selection downstream) until after a winning individual is
        already locked in. When None (development folds), behaves exactly as before: a plain
        train/eval two-way split, no validation.csv written.

    Cleaning/feature-selection steps that fit a threshold to a date window (drop_na_features,
    remove_static_features, ConsensusFeatureSelector) are fit on this fold's TRAIN rows only -
    never on inner-validation or eval rows. The resulting feature list is then applied - never
    independently re-fit - to every other split. An embargo_months-wide gap is purged from the
    tail of the training window, and (when inner_val_year is set) from the tail of the
    inner-validation window too, so neither eval boundary sits immediately adjacent to the split
    before it.

    terminal_cap: passed straight through to ConsensusFeatureSelector's own terminal_cap
        constructor param (default None, its own class default - no cap) - callers wanting fast
        mode's reduced cap pass a smaller value here (see PipelineConfig.fast_terminal_cap).
    """
    eval_start_year, eval_end_year = eval_years
    print(f"\n=== run_fold: {output_dir} (train {train_years[0]}-{train_years[1]}"
          f"{f', validate {inner_val_years[0]}-{inner_val_years[1]}' if inner_val_years is not None else ''}, "
          f"{eval_label} {eval_start_year}-{eval_end_year}) ===")

    train_start_year, train_end_year = train_years
    nominal_train_end_date = date(train_end_year, 12, 31)

    embargo_cutoff_month = 12 - embargo_months
    if not (1 <= embargo_cutoff_month <= 12):
        raise ValueError("embargo_months must be in [0, 11]")
    embargo_cutoff_last_day = calendar.monthrange(train_end_year, embargo_cutoff_month)[1]
    effective_train_cutoff = date(train_end_year, embargo_cutoff_month, embargo_cutoff_last_day)

    eval_start_date = date(eval_start_year, 1, 1)
    eval_end_date = date(eval_end_year, 12, 31)

    select_cols = ['fsym', 'date', sector_col, 'target_date', 'return_year', 'return_month', target] + candidate_feature_columns

    train_df = base_df.filter(
        (F.col('target_date') >= date(train_start_year, 1, 1)) & (F.col('target_date') <= effective_train_cutoff)
    ).select(*select_cols).cache()
    eval_df = base_df.filter(
        (F.col('target_date') >= eval_start_date) & (F.col('target_date') <= eval_end_date)
    ).select(*select_cols).cache()

    train_rows = train_df.count()
    eval_rows = eval_df.count()
    print(f"run_fold: train rows={train_rows} (target_date in [{date(train_start_year, 1, 1)}, {effective_train_cutoff}]), "
          f"{eval_label} rows={eval_rows} (target_date in [{eval_start_date}, {eval_end_date}])")

    if train_rows == 0 or eval_rows == 0:
        raise ValueError(f"run_fold: empty train or {eval_label} slice for {output_dir} - check fold boundaries")

    # --- optional inner-validation split (final-test folds only) - carved out the same way the
    # outer train/eval split is: an embargoed cutoff trims the tail of inner_val_years so it
    # isn't immediately adjacent to eval_years either. The embargo cutoff month applies only to
    # inner_val_years' own last year, same convention as train_end_year above. inner_val_df is
    # only ever used the same way eval_df is below (apply_selected_features) - never fit on,
    # same as eval_df.
    inner_val_df = None
    inner_val_rows = None
    effective_inner_val_cutoff = None
    inner_val_start_date = None
    if inner_val_years is not None:
        inner_val_start_year, inner_val_end_year = inner_val_years
        inner_val_cutoff_last_day = calendar.monthrange(inner_val_end_year, embargo_cutoff_month)[1]
        effective_inner_val_cutoff = date(inner_val_end_year, embargo_cutoff_month, inner_val_cutoff_last_day)
        inner_val_start_date = date(inner_val_start_year, 1, 1)

        inner_val_df = base_df.filter(
            (F.col('target_date') >= inner_val_start_date) & (F.col('target_date') <= effective_inner_val_cutoff)
        ).select(*select_cols).cache()
        inner_val_rows = inner_val_df.count()
        print(f"run_fold: validation rows={inner_val_rows} (target_date in [{inner_val_start_date}, {effective_inner_val_cutoff}])")

        if inner_val_rows == 0:
            raise ValueError(f"run_fold: empty inner-validation slice for {output_dir} - check fold boundaries")

    # prev_month_return/prev_month_sector_return are exempt from every feature-selection step
    # below (drop_na_features, remove_static_features, ConsensusFeatureSelector) and always survive into
    # final_features unconditionally - this matches what GA has always assumed: it computes
    # these two itself, unconditionally, as its baseline predictors alongside whatever
    # super-feature is being scored, so they were never previously subject to any selection
    # process at all. Pulled out of the candidate pool before any selection step runs, then
    # re-added to final_features at the end.
    ALWAYS_KEEP_FEATURES = ['prev_month_return', 'prev_month_sector_return']
    selectable_candidates = [c for c in candidate_feature_columns if c not in ALWAYS_KEEP_FEATURES]
    always_keep_present = [c for c in ALWAYS_KEEP_FEATURES if c in candidate_feature_columns]

    # --- train-only, per-fold column selection (must never see eval rows) ---
    train_features_view = train_df.select('fsym', 'date', *selectable_candidates)
    train_features_view, dropped_na_features, _ = utils.drop_na_features(train_features_view, drop_threshold=0.3)
    surviving_after_na = [c for c in selectable_candidates if c not in dropped_na_features]

    train_features_view, dropped_static_features = utils.remove_static_features(train_features_view, surviving_after_na)
    surviving_after_static = [c for c in surviving_after_na if c not in dropped_static_features]
    print(f"run_fold: candidate features {len(candidate_feature_columns)} ({len(always_keep_present)} always-kept, "
          f"{len(selectable_candidates)} selectable) -> post-drop_na {len(surviving_after_na)} "
          f"-> post-remove_static {len(surviving_after_static)}")

    # ConsensusFeatureSelector: single-pass Spearman |corr| / mutual information / RF
    # permutation-importance consensus (retain iff top-30% by >=2 of 3 methods), then
    # hierarchical correlation clustering at |corr|>=0.80. Fit on this fold's train rows only,
    # same as drop_na_features/remove_static_features above. always_keep_present was pulled out
    # of the candidate pool before this step and is unconditionally re-added afterward, so it's
    # never subject to the consensus vote.
    consensus_train_df = train_df.select('fsym', sector_col, target, *surviving_after_static)
    selector = ConsensusFeatureSelector(spark, random_seed=random_seed, terminal_cap=terminal_cap)
    consensus_result = selector.run(consensus_train_df, feature_columns=surviving_after_static, target=target, sector_col=sector_col)
    final_features = consensus_result.final_features + always_keep_present
    print(f"run_fold: ConsensusFeatureSelector selected {len(consensus_result.final_features)} features "
          f"+ {len(always_keep_present)} always-kept -> {len(final_features)} final features")

    # --- apply the fold's train-fit feature list to every split - never re-fit on inner-val/eval ---
    train_out = utils.apply_selected_features(train_df, candidate_feature_columns, final_features)
    eval_out = utils.apply_selected_features(eval_df, candidate_feature_columns, final_features)
    inner_val_out = (
        utils.apply_selected_features(inner_val_df, candidate_feature_columns, final_features)
        if inner_val_df is not None else None
    )

    # --- write outputs ---
    os.makedirs(output_dir, exist_ok=True)
    utils.write_to_csv(train_out, f"{output_dir}/train.csv")
    utils.write_to_csv(eval_out, f"{output_dir}/{eval_label}.csv")
    if inner_val_out is not None:
        utils.write_to_csv(inner_val_out, f"{output_dir}/validation.csv")

    with open(f"{output_dir}/selected_features.txt", "w") as f:
        f.write("\n".join(final_features))
    print(f"run_fold: wrote {output_dir}/selected_features.txt ({len(final_features)} features)")

    write_consensus_artifacts(consensus_result, f"{output_dir}/diagnostics")

    write_fold_metadata(output_dir, {
        'output_dir': output_dir,
        'train_years': list(train_years),
        'eval_years': list(eval_years),
        'inner_val_years': list(inner_val_years) if inner_val_years is not None else None,
        # Backward-compat scalars for readers that predate multi-year windows (ga/algorithms.py,
        # analysis/summary.py) - the most recent year in each window, since that's what "as of"
        # reporting/sorting by a single eval_year wants. Width-1 windows (today's default) make
        # these identical to eval_years[0]/inner_val_years[0], i.e. a no-op for existing callers.
        'eval_year': eval_years[-1],
        'inner_val_year': inner_val_years[-1] if inner_val_years is not None else None,
        'eval_label': eval_label,
        'embargo_months': embargo_months,
        'nominal_train_end_date': nominal_train_end_date,
        'effective_train_cutoff': effective_train_cutoff,
        'inner_val_start_date': inner_val_start_date,
        'effective_inner_val_cutoff': effective_inner_val_cutoff,
        'eval_start_date': eval_start_date,
        'eval_end_date': eval_end_date,
        'train_row_count': train_rows,
        'inner_val_row_count': inner_val_rows,
        'eval_row_count': eval_rows,
        'n_candidate_features': len(candidate_feature_columns),
        'n_always_kept_features': len(always_keep_present),
        'n_selectable_features': len(selectable_candidates),
        'n_post_drop_na_features': len(surviving_after_na),
        'n_post_remove_static_features': len(surviving_after_static),
        'n_final_features': len(final_features),
    })

    train_df.unpersist()
    eval_df.unpersist()
    if inner_val_df is not None:
        inner_val_df.unpersist()

    print(f"=== run_fold: {output_dir} done ===")
