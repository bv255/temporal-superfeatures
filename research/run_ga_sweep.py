"""
Thin loop over run_ga.py --execution cluster for a seed sweep - the motivating case for
--execution cluster (see ~/.claude/plans/clever-scribbling-mccarthy.md and
docs/evaluation_framework.md's 10-matched-seed primary estimand, not yet run end-to-end). Not a
new framework: every seed just becomes its own `run_ga.py --seed N ... --execution cluster`
subprocess, each landing on its own YARN container. No seed list is hardcoded here -
evaluation_framework.md notes the final 10-seed list is "fixed during development" but doesn't
pin specific numbers in-repo yet, and existing ga_runs/ on disk already show an ad hoc set
(7, 8, 10-13, 20, 100-103) rather than a settled list - so --seeds is required, not defaulted.

Usage (mirrors CLAUDE.md's "Comparing a temporal-on run against a temporal-off run" pairing -
run both arms for the same seed set):
    python3 run_ga_sweep.py --seeds 7,8,10,11,12,13,20,100,101,102 --max-concurrent 4 -- --fit-backend local
    python3 run_ga_sweep.py --seeds 7,8,10,11,12,13,20,100,101,102 --max-concurrent 4 -- --fit-backend local --no-temporal-operators

Everything after `--` is passed through to run_ga.py unchanged for every seed (--execution
cluster and --seed N are added here, not repeated by hand).

--max-concurrent caps how many submitted runs are allowed to be outstanding (ACCEPTED/RUNNING,
not yet finished) at once - the cluster is shared with other students, so this is a deliberate
throttle, not YARN's own admission control (which would happily queue all of --seeds' worth at
once and let them start as capacity frees). Polls each in-flight app's YARN status and submits
the next queued seed as soon as one finishes, rather than submitting in rigid batches - a run
that finishes early doesn't sit waiting on slower siblings before the next seed starts.
"""
import argparse
import os
import subprocess
import sys
import time

STAGGER_SECONDS = 5  # spaces out spark-submit client calls so they don't hit the ResourceManager
                      # in one burst.
POLL_SECONDS = 20     # how often to check in-flight apps' YARN status while waiting for a slot.

HADOOP_HOME = "/opt/hadoop"
HADOOP_CONF_DIR = "/opt/hadoop-3.4.1/etc/hadoop"


def _final_state(app_id: str) -> str:
    """Returns this application's YARN Final-State once it's no longer UNDEFINED (SUCCEEDED,
    FAILED, or KILLED), or None while it's still in flight (or status couldn't be parsed - treated
    as still in flight rather than dropping it silently).

    The verdict is trustworthy now (2026-08-24) - it used to never be, back when the cluster-child
    ran master("local[*]"): that structurally never registered a YARN SparkContext, so the
    ApplicationMaster reported every run FAILED regardless of real outcome (see run_ga.py's
    _build_spark_session/cluster_submit.py docstrings), and this function used to just collapse
    everything to a bool for that reason - a genuine failure and a genuine success were
    indistinguishable, so the caller couldn't do anything with the verdict beyond "slot freed."
    Now that cluster-child runs request a real (if tiny) executor and register normally, FAILED
    here means the run actually failed - the caller surfaces that instead of silently treating it
    the same as SUCCEEDED."""
    env = dict(os.environ, HADOOP_HOME=HADOOP_HOME, HADOOP_CONF_DIR=HADOOP_CONF_DIR)
    yarn_bin = os.path.join(HADOOP_HOME, "bin", "yarn")
    result = subprocess.run([yarn_bin, "application", "-status", app_id],
                             capture_output=True, text=True, env=env)
    for line in result.stdout.splitlines():
        if "Final-State" in line:
            state = line.split(":", 1)[1].strip()
            return None if state == "UNDEFINED" else state
    return None


def _submit(seed: int, passthrough: list) -> str:
    cmd = [sys.executable, "run_ga.py", "--seed", str(seed), "--execution", "cluster", *passthrough]
    print(f"Submitting seed {seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"WARNING: seed {seed} submission failed (exit {result.returncode}) - "
              f"continuing with the remaining seeds.", file=sys.stderr)
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Submitted YARN application "):
            return line.split()[3]
    print(f"WARNING: seed {seed} submitted but no application id found in its output.", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True,
                         help="comma-separated seed list, e.g. 7,8,10,11,12,13,20,100,101,102")
    parser.add_argument("--max-concurrent", type=int, required=True,
                         help="max number of submitted-but-unfinished runs at once")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    pending = [int(s) for s in args.seeds.split(",")]
    in_flight = {}  # seed -> app_id
    app_ids = {}    # seed -> app_id, everything ever submitted successfully
    outcomes = {}   # seed -> YARN Final-State, once known

    while pending or in_flight:
        while pending and len(in_flight) < args.max_concurrent:
            seed = pending.pop(0)
            app_id = _submit(seed, passthrough)
            if app_id is not None:
                in_flight[seed] = app_id
                app_ids[seed] = app_id
            if pending and len(in_flight) < args.max_concurrent:
                time.sleep(STAGGER_SECONDS)

        if not in_flight:
            break  # everything remaining failed to submit - nothing left to wait on

        time.sleep(POLL_SECONDS)
        for seed, app_id in list(in_flight.items()):
            state = _final_state(app_id)
            if state is not None:
                print(f"Seed {seed} ({app_id}) finished: {state} - slot freed.")
                outcomes[seed] = state
                del in_flight[seed]

    print("\nSubmitted:")
    for seed, app_id in app_ids.items():
        print(f"  seed {seed}: {app_id} ({outcomes.get(seed, 'still running?')})")
    if len(app_ids) < len(args.seeds.split(",")):
        print(f"WARNING: only {len(app_ids)}/{len(args.seeds.split(','))} seeds submitted successfully.")
    failed_seeds = [seed for seed, state in outcomes.items() if state != "SUCCEEDED"]
    if failed_seeds:
        print(f"WARNING: {len(failed_seeds)} seed(s) did NOT report SUCCEEDED and likely need "
              f"rerunning: {failed_seeds}. Verify against each one's fold_result.json on disk "
              f"(the authoritative signal - see cluster_submit.py's docstring) before assuming "
              f"this list is complete or that a SUCCEEDED seed produced every fold.")


if __name__ == "__main__":
    main()
