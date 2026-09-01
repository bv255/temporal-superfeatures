"""
Very barebones live dashboard for `--execution cluster` seed sweeps. Wraps
`research/scripts/watch_cluster_folds.py`'s own log-tailing/YARN-discovery machinery in a small
FastAPI app instead of printing to stdout, and additionally checks whether each seed/arm's
output has already been pulled down locally via `research/scripts/pull_cluster_output.py` (a
finished cluster run's output only reaches `research/ga_runs/` once someone runs that script by
hand - nothing does it automatically, see that script's own docstring).

Setup (one-time, into the existing project venv):
    ~/pyspark-venv/bin/pip install fastapi uvicorn

Run (from anywhere):
    ~/pyspark-venv/bin/python3 -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8899 --app-dir ~/temporal-superfeatures

Then tunnel in like Jupyter (CLAUDE.md's "Interactive Jupyter" section) and open
http://127.0.0.1:8899/ in a normal browser:
    ssh -L 8899:127.0.0.1:8899 bvail@bialobog.cs.ucl.ac.uk

GET /seeds returns the same data as JSON, if you'd rather curl/script against it than look at the
table.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = os.path.expanduser("~/temporal-superfeatures")
RESEARCH_DIR = os.path.join(REPO_ROOT, "research")
sys.path.insert(0, os.path.join(RESEARCH_DIR, "scripts"))
import watch_cluster_folds as wcf  # noqa: E402

POLL_SECONDS = 20.0


def _discover_seed_details(manual_seeds: dict) -> dict:
    """appid -> {seed, arm ('temporal'/'no_temporal'/'unknown'), fast (bool|None), family
    (str|None)}. Reimplements watch_cluster_folds.discover_seed_map's sweep-log scan rather than
    calling it directly, since this dashboard also needs --fast and a reconstructed family name
    (to check pull status against research/ga_runs/) that discover_seed_map's plain seed/arm
    label doesn't carry. Family suffix order (base -> _no_temporal -> _seedN, no _local/_rank_ic
    suffix since those are run_ga.py's own script-level defaults now, not opt-in modes) confirmed
    directly against run_ga.py's __main__ block, 2026-08-24.
    """
    details = {appid: {"seed": seed, "arm": "unknown", "fast": None, "family": None}
               for appid, seed in manual_seeds.items()}
    for logpath in glob.glob(os.path.expanduser("~/sweep_logs/*.log")):
        try:
            text = open(logpath).read()
        except Exception:
            continue
        pending = None
        for line in text.splitlines():
            sm = re.match(r"Submitting seed (\d+): (.*)", line)
            if sm:
                seed_num, cmd = int(sm.group(1)), sm.group(2)
                arm = "no_temporal" if "--no-temporal-operators" in cmd else "temporal"
                fast = "--fast" in cmd
                base = "ga_fast" if fast else "ga"
                family = base + ("_no_temporal" if arm == "no_temporal" else "") + f"_seed{seed_num}"
                pending = {"seed": seed_num, "arm": arm, "fast": fast, "family": family}
                continue
            am = re.search(r"Submitted YARN application (application_\S+)", line)
            if am and pending is not None:
                details[am.group(1)] = pending
                pending = None
    return details


def _yarn_final_state(appid: str) -> str:
    """SUCCEEDED/FAILED/KILLED once known, else 'RUNNING'/'UNKNOWN' - same yarn CLI call
    run_ga_sweep.py's own _final_state uses, duplicated here rather than imported since that
    script returns None instead of a display string and lives alongside run_ga.py, not
    watch_cluster_folds.py."""
    env = {**os.environ, "HADOOP_HOME": "/opt/hadoop", "HADOOP_CONF_DIR": wcf.HADOOP_CONF_DIR}
    try:
        result = subprocess.run(
            ["/opt/hadoop/bin/yarn", "application", "-status", appid],
            capture_output=True, text=True, timeout=20, env=env,
        )
    except Exception:
        return "UNKNOWN"
    for line in result.stdout.splitlines():
        if "Final-State" in line:
            state = line.split(":", 1)[1].strip()
            return state if state != "UNDEFINED" else "RUNNING"
    return "UNKNOWN"


def _pulled_dir(family: str | None) -> str | None:
    """Most-recent local research/ga_runs/{family}_* dir, if pull_cluster_output.py has ever
    pulled this family down - None if not (or if family couldn't be reconstructed, e.g. a
    --manual-seed entry with unknown arm/scale)."""
    if not family:
        return None
    matches = sorted(
        glob.glob(os.path.join(RESEARCH_DIR, "ga_runs", f"{family}_*")),
        key=os.path.getmtime, reverse=True,
    )
    return os.path.relpath(matches[0], RESEARCH_DIR) if matches else None


class SeedDashboard:
    """Background poller - reuses watch_cluster_folds.ClusterFoldWatcher's get_host/get_container
    caching and curl-retry helper directly, but builds a structured snapshot instead of printing,
    and adds the pull-status check watch_cluster_folds itself doesn't do."""

    def __init__(self, poll_seconds: float = POLL_SECONDS):
        self.poll_seconds = poll_seconds
        self._w = wcf.ClusterFoldWatcher(manual_seeds={}, poll_seconds=poll_seconds)
        self._lock = threading.Lock()
        self._snapshot: list = []
        self._fold_start_seen: dict = {}    # (appid, fold_name) -> first-seen wallclock
        self._final_state_cache: dict = {}  # appid -> cached terminal YARN state
        self._folds_cache: dict = {}        # appid -> folds dict, cached once an app is terminal
        self._last_updated: str | None = None

    def _poll_appid(self, appid: str, info: dict, running: set) -> dict:
        row = {"appid": appid, **info, "folds": {}}

        if appid in running:
            row["yarn_state"] = "RUNNING"
        elif appid in self._final_state_cache:
            row["yarn_state"] = self._final_state_cache[appid]
        else:
            state = _yarn_final_state(appid)
            self._final_state_cache[appid] = state
            row["yarn_state"] = state

        # Once an app is terminal, its container log (and therefore its fold history) can never
        # change again - cache it after the first successful/failed fetch so a dead app from an
        # old sweep doesn't cost a curl round-trip (with its own retry-on-failure delay) on every
        # single poll cycle forever. A RUNNING app is always fetched fresh.
        if appid in running or appid not in self._folds_cache:
            folds = {}
            host = self._w.get_host(appid)
            container = host and self._w.get_container(appid)
            if host and container:
                text = wcf._run_curl([
                    f"http://{host}:8042/node/containerlogs/{container}/bvail/stdout/?start=0"
                ])
                if text and "Unknown container" not in text:
                    for sm in wcf.FOLD_START_RE.finditer(text):
                        key = (appid, sm.group(1))
                        self._fold_start_seen.setdefault(key, time.time())
                        folds.setdefault(sm.group(1), {"status": "running"})
                    for em in wcf.FOLD_END_RE.finditer(text):
                        fold_name, rmse = em.group(1), em.group(2)
                        started = self._fold_start_seen.get((appid, fold_name))
                        folds[fold_name] = {
                            "status": "complete",
                            "rmse": rmse,
                            "elapsed_seconds": round(time.time() - started) if started else None,
                        }
            if appid not in running:
                self._folds_cache[appid] = folds
            row["folds"] = folds
        else:
            row["folds"] = self._folds_cache[appid]

        row["pulled_dir"] = _pulled_dir(info.get("family"))
        row["pulled"] = (row["pulled_dir"] is not None) if info.get("family") else None
        return row

    def poll_once(self):
        seed_map = _discover_seed_details({})
        running = set(wcf.discover_running_apps())
        appids = sorted(set(seed_map) | running)

        # Each appid's work is almost entirely network I/O (yarn CLI calls, curl against a
        # NodeManager) - sequential polling of a whole sweep log's worth of historical appids
        # (dead ones included) took minutes on a real run, leaving /seeds reporting an empty
        # snapshot the whole time. Fan out across threads, but cap concurrency well below
        # bialobog's 4 physical cores worth of headroom - each `yarn` CLI call spawns its own
        # JVM, so too much concurrency here just makes every call slower (confirmed directly:
        # 16 workers made every single get_container() call blow its own 25s timeout from CPU
        # contention, worse than running them one at a time).
        rows = []
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(appids)))) as pool:
            futures = {
                pool.submit(
                    self._poll_appid, appid,
                    seed_map.get(appid, {"seed": appid, "arm": "unknown", "fast": None, "family": None}),
                    running,
                ): appid
                for appid in appids
            }
            for future in futures:
                rows.append(future.result())

        with self._lock:
            self._snapshot = rows
            self._last_updated = datetime.now(timezone.utc).isoformat()

    def run_forever(self):
        while True:
            try:
                self.poll_once()
            except Exception as e:
                print(f"[dashboard] poll cycle raised {e!r} - continuing", flush=True)
            time.sleep(self.poll_seconds)

    def snapshot(self) -> dict:
        with self._lock:
            return {"updated_at": self._last_updated, "seeds": list(self._snapshot)}


dashboard = SeedDashboard()
app = FastAPI(title="GA seed dashboard")


@app.on_event("startup")
def _start_poller():
    threading.Thread(target=dashboard.run_forever, daemon=True).start()


@app.get("/seeds")
def get_seeds() -> JSONResponse:
    return JSONResponse(dashboard.snapshot())


def _pulled_label(pulled) -> str:
    if pulled is True:
        return "yes"
    if pulled is False:
        return "no"
    return "?"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    data = dashboard.snapshot()
    rows_html = []
    for row in sorted(data["seeds"], key=lambda r: str(r.get("seed"))):
        folds_html = "<br>".join(
            f"{name}: {info['status']}" + (f" (RMSE {info['rmse']})" if info.get("rmse") else "")
            for name, info in sorted(row["folds"].items())
        ) or "&mdash;"
        scale = "fast" if row["fast"] else ("full" if row["fast"] is False else "?")
        rows_html.append(f"""
        <tr>
          <td>{row['seed']}</td>
          <td>{row['arm']}</td>
          <td>{scale}</td>
          <td>{row['yarn_state']}</td>
          <td>{folds_html}</td>
          <td>{_pulled_label(row['pulled'])}</td>
          <td><code>{row['appid']}</code></td>
        </tr>""")
    return f"""<!doctype html>
<html><head><title>GA seed dashboard</title>
<meta http-equiv="refresh" content="20">
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
</style></head>
<body>
<h1>GA seed dashboard</h1>
<p>Updated: {data['updated_at']}</p>
<table>
<tr><th>Seed</th><th>Arm</th><th>Scale</th><th>YARN state</th><th>Folds</th><th>Pulled?</th><th>App ID</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""
