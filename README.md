# temporal-superfeatures

A genetic-algorithm search that evolves symbolic "super-features" over company fundamentals to
predict next-month stock returns, built to answer one question: **do live temporal operators
(lag/delta/growth/rolling-window transforms, evolved *inside* the search rather than
precomputed) actually improve predictive performance over a static-feature baseline?**

Full methodology and write-up: my dissertation. This repo is the implementation, the pre-registered
10-seed evaluation results, and the docs below for anyone digging into a specific design decision.

## Headline result

**H1 not supported.** Across the pre-registered 10-seed matched evaluation (seeds 100–109, both
arms, full scale): observed mean rank-IC delta (temporal ON − OFF) = **+0.000139**, one-sided 95%
lower bound = −0.0065 (crosses zero), one-sided p = 0.290. The live temporal operators show no
statistically significant predictive benefit over the control in this pipeline.

## Pipeline

`preprocessing → GA search (temporal ops ON/OFF) → evaluation vs. baselines → ON-vs-OFF ablation`

1. **Preprocessing** — fundamentals/price data cleaned and aligned to the date each fact was
   actually knowable, split into walk-forward folds with per-fold feature selection.
2. **GA search** — a genetic algorithm evolves symbolic expressions over the selected features,
   run once with temporal operators available (`ga_seed*`) and once without (`ga_no_temporal_seed*`),
   otherwise identical configuration.
3. **Evaluation** — each fold's winning expression is scored against within-run baselines with
   block-bootstrap confidence intervals and Holm-Bonferroni correction; the two arms are then
   compared directly to produce the headline verdict above.

## Repo layout

```
research/
  src/superfeatures/   the pipeline as a package (preprocessing, GA engine, grammar,
                        temporal operators, evaluation/significance, local + Spark fit backends)
  run_preprocessing.py / run_ga.py / compare_ga_runs.py / run_optuna_sweep.py   entry points
  ga_runs/              the pre-registered 10-seed evaluation set, full per-fold results
  tests/                pytest suite for the package
docs/                   methodology, evaluation protocol, and per-topic design notes (see below)
dashboard/              small FastAPI live-progress viewer for cluster-distributed runs
scripts/                standalone analysis/plotting utilities
```

## Docs

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — full pipeline methodology and design rationale
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — scoring, significance testing, the three Rank-IC
  estimands and how they reconcile
- [`docs/figures/`](docs/figures/) — generated plots for the seed-100-109 evaluation

## Requirements

- Python ≥3.10, dependencies in [`research/pyproject.toml`](research/pyproject.toml)
  (`pip install -e research/`) — core: numpy, pandas, scipy, scikit-learn, xgboost, optuna,
  matplotlib/seaborn, pyarrow, pyspark 3.5.5 (+ matching py4j 0.10.9.7). Extras:
  `research[dashboard]` (fastapi, uvicorn) for `dashboard/`, `research[test]` (pytest) for the
  test suite.
- A Spark/YARN cluster with HDFS and a Hive metastore exposing **FactSet** fundamentals and price
  tables — this is licensed institutional data (accessed here via a university cluster) that
  isn't shipped with or reproducible from this repo. Without it, `run_preprocessing.py`/`run_ga.py`
  won't run, but the code, `research/ga_runs/` results, and docs are still fully readable.

## Running it

From `research/`, with the venv active (needs the cluster/data access above):

```bash
python3 run_preprocessing.py                                                     # build folds
python3 run_ga.py --fitness-metric rank_ic --seed <seed>                         # temporal ON
python3 run_ga.py --fitness-metric rank_ic --seed <seed> --no-temporal-operators # temporal OFF
python3 compare_ga_runs.py --seeds <seeds you want to compare>                   # ablation verdict

# seed sweep across the cluster instead of one at a time (--execution cluster per seed):
python3 run_ga_sweep.py --seeds <comma-separated seeds> --max-concurrent <N>

# joint hyperparameter search across both arms (Optuna, dev folds only):
python3 run_optuna_sweep.py --n-trials <N> --parallel-trials <N> --max-concurrent-cluster <N>
```

Test suite (no cluster needed): `pytest research/tests/`.
