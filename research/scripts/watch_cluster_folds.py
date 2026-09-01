"""
Live per-fold progress for in-flight `--execution cluster` run_ga.py runs - not part of the
pipeline, not imported by anything. Exists because a cluster-child's real output (fold_result.json
etc.) only reaches HDFS/bialobog once the *whole* multi-fold run finishes (run_ga.py's `finally`
block pushes config.output_dir in one shot - see its own comment) or fails outright, so there's no
way to see "did fold_03 finish yet" on a run that's still going by looking at local disk or HDFS.
This instead tails each running app's own AM container stdout log straight off its NodeManager
(the same log `yarn logs -applicationId <id>` would eventually show, but readable while the app is
still RUNNING) and watches for the two lines run_ga.py itself already prints around a fold's
boundaries:
    ===== Fold fold_01 (final_test, eval_year=2018) =====      <- fold start
    Fold fold_01: true held-out test RMSE (...) = 0.0682...    <- fold done (winner scored)

Usage:
    ~/pyspark-venv/bin/python3 research/scripts/watch_cluster_folds.py
    ~/pyspark-venv/bin/python3 research/scripts/watch_cluster_folds.py --poll-seconds 15
    ~/pyspark-venv/bin/python3 research/scripts/watch_cluster_folds.py --manual-seed application_1787288171816_0086:103

Seed numbers are recovered by parsing every ~/sweep_logs/*.log (run_ga_sweep.py's own
"Submitting seed N: ..." / "Submitted YARN application ..." print pairs - see run_ga_sweep.py's
_submit) - covers every sweep ever run, not just the current one, so a run submitted yesterday and
still (somehow) alive would still get labeled. A run submitted by hand (not through
run_ga_sweep.py, so it has no sweep-log entry) needs --manual-seed appid:seed to get a label -
otherwise it's still watched, just reported by its raw application_ id.

Elapsed time per fold is wall-clock observed by this script (time between first seeing that fold's
"===== Fold" start line and its "true held-out test RMSE" completion line), polled every
--poll-seconds - so it's accurate to within one poll interval, and (unlike fold_result.json's own
authoritative elapsed_seconds, unreachable until the whole run finishes/pushes) it's the earliest
possible signal, available the moment the fold actually finishes rather than only once the entire
run - all folds - is done.

Network note: curl calls against the skrzat* NodeManagers occasionally fail with "Couldn't resolve
host" (curl exit 6) on this cluster - confirmed transient (an immediate retry succeeds) while
building this, not a real outage - so every curl call here retries a few times before giving up for
that poll cycle.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HADOOP_CONF_DIR = "/opt/hadoop-3.4.1/etc/hadoop"
FOLD_START_RE = re.compile(r"===== Fold (fold_\d+) \((\w+), eval_year=(\d+)\) =====")
FOLD_END_RE = re.compile(r"Fold (fold_\d+): true held-out test RMSE .*= ([\d.]+)")
_CURL_RETRIES = 3
_CURL_RETRY_DELAY = 2.0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_curl(args: list) -> str:
    """subprocess curl with a few retries - see this file's module docstring on the transient
    'Couldn't resolve host' flakiness observed against skrzat*'s NodeManagers."""
    last_err = None
    for attempt in range(1, _CURL_RETRIES + 1):
        try:
            r = subprocess.run(["curl", "-s", *args], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout:
                return r.stdout
            last_err = f"rc={r.returncode} stderr={r.stderr[:200]!r}"
        except Exception as e:
            last_err = repr(e)
        if attempt < _CURL_RETRIES:
            time.sleep(_CURL_RETRY_DELAY)
    _log(f"[warn] curl {args[-1]} failed after {_CURL_RETRIES} attempts: {last_err}")
    return ""


def discover_seed_map(manual_seeds: dict) -> dict:
    """appid -> label (e.g. "105" or "105/no_temporal"), recovered from every ~/sweep_logs/*.log's
    own 'Submitting seed N: .../run_ga.py --seed N --execution cluster ...' / 'Submitted YARN
    application application_X' print pairs (run_ga_sweep.py's _submit), plus whatever
    --manual-seed appid:seed pairs were passed on the CLI (for runs submitted by hand, which have
    no sweep-log entry at all - always labeled by bare seed number, arm unknown).

    The arm (temporal vs `--no-temporal-operators`) is read directly off the 'Submitting seed N:'
    line itself (it's the full command being submitted, flags and all) rather than off the sweep
    log's filename - running two sweeps concurrently (e.g. the temporal and no-temporal arms at
    once) means seed numbers collide across arms, so the label needs to disambiguate or two
    different runs silently get reported under the identical "seed 105" tag."""
    m = dict(manual_seeds)
    for logpath in glob.glob(os.path.expanduser("~/sweep_logs/*.log")):
        try:
            text = open(logpath).read()
        except Exception:
            continue
        pending = None
        for line in text.splitlines():
            sm = re.match(r"Submitting seed (\d+):", line)
            if sm:
                seed_num = sm.group(1)
                pending = f"{seed_num}/no_temporal" if "--no-temporal-operators" in line else seed_num
                continue
            am = re.search(r"Submitted YARN application (application_\S+)", line)
            if am and pending is not None:
                m[am.group(1)] = pending
                pending = None
    return m


def discover_running_apps() -> list:
    """Every bvail run_ga.py YARN app currently ACCEPTED/RUNNING (not FAILED/KILLED/SUCCEEDED
    yet) - covers apps discover_seed_map's sweep-log scan doesn't know about at all (e.g. one
    submitted by hand, hence --manual-seed) as long as it's actually still alive."""
    out = subprocess.run(
        ["/opt/hadoop/bin/yarn", "application", "-list", "-appStates", "ACCEPTED,RUNNING"],
        capture_output=True, text=True, timeout=25,
        env={**os.environ, "HADOOP_CONF_DIR": HADOOP_CONF_DIR},
    ).stdout
    apps = []
    for line in out.splitlines():
        # Columns are tab-separated but each value is right-padded with spaces for on-screen
        # alignment (e.g. "...SPARK\t     bvail\troot.bvail..."), so a plain "\tbvail\t" substring
        # check never matches - the tab lands before the padding, not before "bvail" itself. Split
        # on tabs and strip each field instead.
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) >= 4 and fields[1] == "run_ga.py" and fields[3] == "bvail":
            apps.append(fields[0])
    return apps


class ClusterFoldWatcher:
    def __init__(self, manual_seeds: dict, poll_seconds: float):
        self.manual_seeds = manual_seeds
        self.poll_seconds = poll_seconds
        self.host_cache = {}
        self.container_cache = {}
        self.fold_start_seen = {}   # (appid, fold_name) -> wallclock first seen
        self.completed = set()      # (appid, fold_name) already reported
        self.dead_apps = set()

    def get_host(self, appid: str):
        if appid in self.host_cache:
            return self.host_cache[appid]
        out = _run_curl([f"http://bialobog:8088/ws/v1/cluster/apps/{appid}"])
        m = re.search(r'"amHostHttpAddress"\s*:\s*"([^"]+)"', out)
        if m and m.group(1) != "N/A":
            self.host_cache[appid] = m.group(1).split(":")[0]
            return self.host_cache[appid]
        return None

    def get_container(self, appid: str):
        if appid in self.container_cache:
            return self.container_cache[appid]
        try:
            out = subprocess.run(
                ["/opt/hadoop/bin/yarn", "applicationattempt", "-list", appid],
                capture_output=True, text=True, timeout=25,
                env={**os.environ, "HADOOP_CONF_DIR": HADOOP_CONF_DIR},
            ).stdout
        except Exception as e:
            _log(f"[warn] get_container({appid}) failed: {e!r}")
            return None
        m = re.search(r"container_\S+", out)
        if m:
            self.container_cache[appid] = m.group(0)
            return self.container_cache[appid]
        return None

    def poll_once(self):
        seed_map = discover_seed_map(self.manual_seeds)
        running = discover_running_apps()
        appids = sorted(set(seed_map) | set(running)) if running else sorted(seed_map)
        for appid in appids:
            if appid in self.dead_apps:
                continue
            seed = seed_map.get(appid, appid)
            host = self.get_host(appid)
            if not host:
                continue
            container = self.get_container(appid)
            if not container:
                continue
            text = _run_curl([
                f"http://{host}:8042/node/containerlogs/{container}/bvail/stdout/?start=0"
            ])
            if not text or "Unknown container" in text:
                if text:  # got a real (non-empty) "Unknown container" page, not just a curl retry-exhaustion blank
                    self.dead_apps.add(appid)
                    _log(f"[info] seed {seed} ({appid}) container no longer reachable - dropping")
                continue
            for sm in FOLD_START_RE.finditer(text):
                key = (appid, sm.group(1))
                self.fold_start_seen.setdefault(key, time.time())
            for em in FOLD_END_RE.finditer(text):
                fold_name = em.group(1)
                key = (appid, fold_name)
                if key in self.completed:
                    continue
                self.completed.add(key)
                rmse = em.group(2)
                started = self.fold_start_seen.get(key)
                if started:
                    dt = time.time() - started
                    elapsed = f"~{dt:.0f}s (~{dt / 60:.1f}min, observed)"
                else:
                    elapsed = "unknown (start marker missed - was already past this fold when first polled)"
                _log(f"seed {seed} ({appid}) {fold_name} COMPLETE - true-test RMSE={rmse} - "
                     f"elapsed {elapsed}")

    def run_forever(self):
        _log("watch_cluster_folds started")
        cycle = 0
        while True:
            cycle += 1
            try:
                self.poll_once()
            except Exception as e:
                _log(f"[error] poll cycle {cycle} raised {e!r} - continuing")
            time.sleep(self.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll-seconds", type=float, default=25.0)
    parser.add_argument("--manual-seed", action="append", default=[],
                         metavar="APPID:SEED",
                         help="label a run this script's sweep-log scan won't find on its own "
                              "(e.g. one submitted by hand, not via run_ga_sweep.py) - repeatable.")
    args = parser.parse_args()

    manual_seeds = {}
    for entry in args.manual_seed:
        appid, _, seed = entry.partition(":")
        if not seed:
            parser.error(f"--manual-seed expects APPID:SEED, got {entry!r}")
        manual_seeds[appid] = int(seed)

    watcher = ClusterFoldWatcher(manual_seeds, args.poll_seconds)
    watcher.run_forever()


if __name__ == "__main__":
    main()
