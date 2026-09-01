#!/bin/bash
# Waits for the currently-running full-scale run_preprocessing.py (pid 300723, started
# 2026-08-26 20:49:39, post dividend-adjustment fix) to finish, verifies it actually
# reached and freshly wrote the last final-test fold (not just that the process died -
# a crash partway through would leave stale/missing output), and only then launches the
# Optuna hyperparameter sweep (2 seeds - OPTUNA_SEEDS=(11,12), fixed in run_optuna_sweep.py -
# x 2 arms, throttled to 6 concurrent cluster jobs, both script defaults; full scale, no
# --fast, matching the preprocessing run this sweep depends on; 40 trials, script default).
# Runs detached (nohup+disown) so it survives the terminal/session that launched it closing.
set -u

PREPROC_PID=300723
RESEARCH_DIR="/home/bvail/temporal-superfeatures/research"
LAST_FOLD_METADATA="$RESEARCH_DIR/walk_forward_full/final_test/fold_05/fold_metadata.json"
PREPROC_START_EPOCH=1787773779  # 2026-08-26 20:49:39 BST, when this run_preprocessing.py started
WATCHER_LOG="$HOME/watch_preprocessing_then_optuna.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCHER_LOG"; }

log "watcher started, waiting for pid $PREPROC_PID to exit"

while [ -e "/proc/$PREPROC_PID" ]; do
    sleep 60
done

log "pid $PREPROC_PID no longer running - checking whether it finished cleanly"

if [ -f "$LAST_FOLD_METADATA" ]; then
    FOLD_MTIME=$(stat -c '%Y' "$LAST_FOLD_METADATA")
else
    FOLD_MTIME=0
fi

if [ "$FOLD_MTIME" -gt "$PREPROC_START_EPOCH" ]; then
    log "final_test/fold_05/fold_metadata.json was freshly written ($(date -d @$FOLD_MTIME)) - preprocessing completed successfully"
    log "launching Optuna sweep (defaults: 2 seeds, max-concurrent-cluster=6, parallel-trials=2, n-trials=40, full scale)"
    cd "$RESEARCH_DIR" || { log "FATAL: could not cd to $RESEARCH_DIR"; exit 1; }
    SWEEP_LOG="$HOME/optuna_sweep_$(date +%Y%m%d_%H%M%S).log"
    nohup ~/pyspark-venv/bin/python3 run_optuna_sweep.py > "$SWEEP_LOG" 2>&1 &
    disown
    log "Optuna sweep launched (pid $!), logging to $SWEEP_LOG"
else
    log "FAILURE: final_test/fold_05/fold_metadata.json is missing or stale (mtime=$FOLD_MTIME, need > $PREPROC_START_EPOCH)."
    log "This means run_preprocessing.py did NOT reach/finish the last fold - it likely crashed or was killed."
    log "NOT launching the Optuna sweep against incomplete/stale preprocessing output. Check $RESEARCH_DIR for errors and rerun preprocessing manually."
fi

log "watcher done"
