"""
Fingerprint-invalidated checkpoint helpers for GeneticAlgorithm1's lightweight per-generation
progress log (see ga/engine.py's `run(checkpoint_path=..., checkpoint_fingerprint=...)` and
`restore_from_checkpoint`, and ga/algorithms.py's run_ga_for_fold wiring).

The fingerprint gates resume: it hashes together the fold's on-disk inputs
(selected_features.txt + fold_metadata.json - catches "you reran preprocessing for this fold")
and the relevant GAConfig fields (catches "you changed GA parameters"). A checkpoint whose
stored fingerprint doesn't match the CURRENT one is never resumed from - load_checkpoint
returns None (same as a missing file), so the caller starts fresh from generation 0 instead. A
resume therefore never silently replays old population/history against changed data or config.

Checkpoints themselves live on HDFS, not bialobog's local disk - this is the highest-frequency
write in the whole pipeline (once per GA generation, i.e. potentially hundreds of times per
fold), and bialobog's local disk is shared, small, and has repeatedly filled up (ENOSPC) under
other users' usage independent of anything this project does. HDFS has multiple TB free and is
durable across a crash, unlike e.g. /dev/shm. This mirrors the local-vs-HDFS split an adjacent
project on the same infra already uses: small/frequent per-generation state -> HDFS,
low-frequency per-fold diagnostics (PNGs, prediction CSVs, fold_result.json) stay on local disk
as before - unchanged by this module.
"""
import dataclasses
import hashlib
import json
import os
import subprocess
import time

import pyarrow.fs as pafs

_HDFS_HOST = "bialobog"
_HDFS_PORT = 8020
_HDFS_CHECKPOINT_ROOT = "/user/bvail/ga-runs"

_hdfs_singleton = None


def _hdfs():
    """Lazy, module-level singleton - avoids re-establishing the Hadoop RPC connection on every
    single-generation checkpoint write.

    pyarrow's HadoopFileSystem embeds a JVM via libhdfs (JNI), separate from the JVM(s) Spark
    itself launches via spark-submit - it needs CLASSPATH populated with the Hadoop jars
    up front (HADOOP_CONF_DIR alone isn't enough), which run_ga.py's own env setup doesn't do
    since it's Spark-focused. Populated here, once, via `hadoop classpath` rather than requiring
    every caller to remember it.
    """
    global _hdfs_singleton
    if _hdfs_singleton is None:
        if "CLASSPATH" not in os.environ:
            hadoop_home = os.environ.get("HADOOP_HOME", "/opt/hadoop")
            os.environ["CLASSPATH"] = subprocess.run(
                [f"{hadoop_home}/bin/hadoop", "classpath", "--glob"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        # bialobog (the driver host) is itself a nearly-full HDFS datanode, and HDFS's default
        # block placement always tries the writer's own node first - same issue run_ga.py's
        # SparkSession already works around via spark.hadoop.dfs.client.block.write.retries=10 /
        # spark.hadoop.dfs.replication=1 (see _build_spark_session's docstring). This is a
        # separate libhdfs connection that doesn't inherit Spark's hadoop config, so it needs the
        # same two settings applied directly or every write hits the same
        # "Unable to create new block" failure after exhausting the default 3 retries.
        _hdfs_singleton = pafs.HadoopFileSystem(
            _HDFS_HOST, port=_HDFS_PORT, replication=1,
            extra_conf={"dfs.client.block.write.retries": "10"},
        )
    return _hdfs_singleton


def _hdfs_path(path: str) -> str:
    """Maps the same relative, local-disk-shaped path callers already pass (e.g.
    'ga/final_test/fold_02/checkpoint.json') onto an HDFS path under a dedicated root, so the
    call sites in algorithms.py/engine.py needed no changes beyond this module."""
    return f"{_HDFS_CHECKPOINT_ROOT}/{path}"


def compute_fingerprint(fold: dict, config) -> str:
    fold_dir_local = fold["fold_dir_local"]
    with open(f"{fold_dir_local}/selected_features.txt") as f:
        selected_features = f.read()
    with open(f"{fold_dir_local}/fold_metadata.json") as f:
        fold_metadata = f.read()
    # output_dir is "where do I write/look," not "what did the search actually run against" -
    # two runs of an identical config can have different output_dir values (a fresh timestamped
    # ga_runs/ dir vs. an existing one run_ga.py's auto-discovery matched onto - see
    # run_ga.py's _find_matching_run), so leaving it in this hash would make an otherwise-perfect
    # match get rejected as fingerprint-stale for reasons that were never a real config/data
    # change.
    config_dict = dataclasses.asdict(config)
    config_dict.pop("output_dir", None)
    payload = json.dumps(
        {
            "selected_features": selected_features,
            "fold_metadata": fold_metadata,
            "config": config_dict,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_checkpoint(path: str, fingerprint: str):
    """
    Returns the checkpoint dict if `path` exists (on HDFS) and its stored fingerprint matches
    `fingerprint` exactly, else None - a missing file, a corrupt/partial file (e.g. a crash
    mid-write before save_checkpoint's atomic rename ever happened), and a fingerprint mismatch
    are all treated identically: start fresh, never guess.
    """
    hdfs = _hdfs()
    hpath = _hdfs_path(path)
    if hdfs.get_file_info(hpath).type == pafs.FileType.NotFound:
        return None
    try:
        with hdfs.open_input_stream(hpath) as f:
            checkpoint = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if checkpoint.get("fingerprint") != fingerprint:
        return None
    return checkpoint


def read_checkpoint_raw(path: str):
    """
    Whatever checkpoint is currently at `path` (HDFS), regardless of its stored fingerprint -
    for a live viewer (scripts/watch_ga_run.py) that wants "whatever's there right now" rather
    than load_checkpoint's resume-safety guarantee. A fingerprint mismatch there means "don't
    resume from this," which isn't a reason to hide live progress from someone just watching -
    the fold's data/config may have changed since an OLDER checkpoint at this same path, but the
    generation/fitness/diversity numbers in it are still real numbers from a real run.

    Same missing/corrupt-file handling as load_checkpoint: both come back as None.
    """
    hdfs = _hdfs()
    hpath = _hdfs_path(path)
    if hdfs.get_file_info(hpath).type == pafs.FileType.NotFound:
        return None
    try:
        with hdfs.open_input_stream(hpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_checkpoint(path: str, checkpoint: dict, max_attempts: int = 3, retry_delay_seconds: float = 5.0) -> None:
    """
    Atomic write (temp file + HDFS rename) - a crash mid-write leaves the previous checkpoint
    (or no file) intact rather than a corrupted partial one, same guarantee the old local
    tmp-file + os.replace version had.

    TEMPORARY: retries on transient HDFS errors (observed: an opaque `errno 255 "Unknown error
    255"` from pyarrow's libhdfs client mid-run, after dozens of prior successful writes, with
    HDFS itself otherwise healthy - dfsadmin -report showed no corrupt/missing/under-replicated
    blocks and plenty of free space) instead of letting one flaky write kill a multi-hour,
    multi-fold run. Checkpointing is a resumability aid, not correctness-critical to the run
    itself, so a write that still fails after retries is logged and skipped (that generation just
    won't be resumable from) rather than raised. Resets the module-level HDFS connection between
    attempts in case the connection itself, not just the one RPC, is what's wedged. Worth
    replacing with a proper fix (or at least narrowing which exceptions this catches) once the
    root cause is understood - see engine.py's call site, which has no other error handling
    around this.
    """
    hdfs_path = _hdfs_path(path)
    tmp_path = f"{hdfs_path}.tmp"
    parent = hdfs_path.rsplit("/", 1)[0]
    payload = json.dumps(checkpoint).encode("utf-8")

    for attempt in range(1, max_attempts + 1):
        try:
            hdfs = _hdfs()
            hdfs.create_dir(parent, recursive=True)
            with hdfs.open_output_stream(tmp_path) as f:
                f.write(payload)
            hdfs.move(tmp_path, hdfs_path)
            return
        except OSError as e:
            global _hdfs_singleton
            _hdfs_singleton = None
            if attempt == max_attempts:
                print(
                    f"WARNING: checkpoint write to {hdfs_path} failed after {max_attempts} "
                    f"attempts ({e!r}) - continuing run without checkpointing this generation."
                )
                return
            print(
                f"WARNING: checkpoint write to {hdfs_path} failed (attempt {attempt}/"
                f"{max_attempts}: {e!r}), retrying in {retry_delay_seconds:.0f}s..."
            )
            time.sleep(retry_delay_seconds)
