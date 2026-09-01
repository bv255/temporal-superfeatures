"""
Ported from `research/PreProcessing_test.ipynb` cell 11 — see `docs/RESTRUCTURING_TODO.md`
and the port plan in `docs/RESEARCH_STRUCTURE.md`. `compute_fold_boundaries` (walk-forward
train/validation/holdout year boundaries) - the sibling `run_fold`/`write_consensus_artifacts`/
`write_fold_metadata` functions cell 13 originally shared this file with now live in
`preprocessing/pipeline.py` (applying a fold's boundaries to slice/screen/write its data is a
distinct concern from computing the boundaries themselves).
"""

from pyspark.sql import functions as F

from ..config import PipelineConfig


def compute_fold_boundaries(wf_base_df, config: PipelineConfig) -> dict:
    """
    Wraps `PreProcessing_test.ipynb` cell 11's walk-forward fold boundary computation into a
    callable - unlike `run_fold` (cell 13), cell 11 was inline driver-cell script, not a
    function, so there is no verbatim source to extract; the arithmetic below is transcribed
    from it unchanged (`final_test_start_idx`/`initial_train_end_idx`/the dev/final-test slicing)
    and every fixed constant it used inline (`FINAL_TEST_FRACTION_START`, `TARGET_DEV_FOLDS`) is
    now a `PipelineConfig` field instead, so the same logic can express either the `test` or
    `test_v2` layout via `config.fold_output_dir`.

    Returns a dict with 'all_years'/'n_years'/'min_target_date'/'max_target_date'/
    'initial_train_end_year'/'dev_validation_years'/'final_test_years'/'dev_folds'/
    'final_test_folds' - the same locals the notebook cell computed, just returned instead of
    left as kernel globals for cells 15/17 to close over.

    Each dev_folds/final_test_folds entry carries 'eval_years' (and final_test_folds'
    'inner_val_years') as a (start_year, end_year) tuple, inclusive on both ends -
    config.year_width wide (both dev and final-test folds share this one width). See
    PipelineConfig.target_final_test_folds for how the final-test fold count itself is pinned
    rather than left to fall out of final_test_fraction_start once chunked.
    """
    date_range = wf_base_df.agg(
        F.min('target_date').alias('min_target_date'), F.max('target_date').alias('max_target_date')
    ).collect()[0]
    all_years = sorted(
        row['target_year'] for row in wf_base_df.select(F.year('target_date').alias('target_year')).distinct().collect()
    )
    n_years = len(all_years)

    width = config.year_width

    if config.target_final_test_folds is not None:
        # Reserve exactly target_final_test_folds * width years, counting back from the most
        # recent year - overrides final_test_fraction_start entirely (see its docstring).
        n_final_test_years = config.target_final_test_folds * width
        final_test_start_idx = max(0, n_years - n_final_test_years)
    else:
        final_test_start_idx = max(0, int(n_years * config.final_test_fraction_start))
        # Trim down to a whole number of width-sized folds - any remainder years at the front
        # of what the fraction selected fall back into the dev-validation region instead.
        n_final_test_years = ((n_years - final_test_start_idx) // width) * width
        final_test_start_idx = n_years - n_final_test_years

    n_dev_years = config.target_dev_folds * width
    initial_train_end_idx = max(0, final_test_start_idx - n_dev_years - 1)

    initial_train_end_year = all_years[initial_train_end_idx]
    dev_validation_years = all_years[initial_train_end_idx + 1: final_test_start_idx]
    final_test_years = all_years[final_test_start_idx:]

    if not dev_validation_years or not final_test_years:
        raise ValueError(
            f"Not enough distinct years ({n_years}) to form both a development window and a "
            f"final-test window at the requested target_dev_folds={config.target_dev_folds} "
            f"(x{width}yr) / final-test sizing (target_final_test_folds="
            f"{config.target_final_test_folds}, year_width={width}, "
            f"final_test_fraction_start={config.final_test_fraction_start}) - check "
            f"config.walk_forward_base_path and the universe/feature caps upstream."
        )

    def _chunk(years, chunk_width):
        # Drops a trailing partial chunk (e.g. 7 years at width 2 -> 3 chunks, last year
        # unused) rather than emitting a narrower final fold - every fold stays exactly
        # chunk_width years wide.
        usable = len(years) - (len(years) % chunk_width)
        return [years[i:i + chunk_width] for i in range(0, usable, chunk_width)]

    dev_windows = _chunk(dev_validation_years, width)
    final_test_windows = _chunk(final_test_years, width)

    min_year = all_years[0]

    dev_folds = [
        {
            'fold_index': i,
            'train_years': (min_year, window[0] - 1),
            'eval_years': (window[0], window[-1]),
            'eval_label': 'validation',
            'output_dir': config.fold_output_dir('development', i),
        }
        for i, window in enumerate(dev_windows, start=1)
    ]

    final_test_folds = [
        {
            'fold_index': i,
            'train_years': (min_year, window[0] - width - 1),
            'inner_val_years': (window[0] - width, window[0] - 1),
            'eval_years': (window[0], window[-1]),
            'eval_label': 'test',
            'output_dir': config.fold_output_dir('final_test', i),
        }
        for i, window in enumerate(final_test_windows, start=1)
    ]

    return {
        'all_years': all_years,
        'n_years': n_years,
        'min_target_date': date_range['min_target_date'],
        'max_target_date': date_range['max_target_date'],
        'initial_train_end_year': initial_train_end_year,
        'dev_validation_years': dev_validation_years,
        'final_test_years': final_test_years,
        'dev_folds': dev_folds,
        'final_test_folds': final_test_folds,
    }
