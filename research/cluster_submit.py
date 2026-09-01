"""
`--execution cluster` support for run_ga.py (see ~/.claude/plans/clever-scribbling-mccarthy.md).
Packages the current code + the pre-packed venv (research/scripts/pack_venv.sh) and re-launches
this exact GA run as its own top-level YARN application via
`spark-submit --deploy-mode cluster`, so the whole run lands on one dedicated YARN container
(a skrzat* node) instead of executing on bialobog. Not imported by anything except run_ga.py's
`--execution cluster` branch.
"""
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

SPARK_HOME = "/opt/spark"
HADOOP_HOME = "/opt/hadoop"
HADOOP_CONF_DIR = "/opt/hadoop-3.4.1/etc/hadoop"
PACKED_VENV = os.path.expanduser("~/pyspark-venv.tar.gz")
SUPERFEATURES_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "superfeatures")

_APP_ID_RE = re.compile(r"application_\d+_\d+")


def package_code(dest_dir: str) -> str:
    """
    Freshly zips src/superfeatures on every call (pure Python, no C extensions - confirmed by
    grep over the package - so this can never go stale between submissions the way a cached
    artifact could). Zipped with a top-level `superfeatures/` prefix so `import superfeatures`
    resolves once spark-submit's --py-files puts this zip on the child process's sys.path.
    """
    out_path = os.path.join(dest_dir, "superfeatures_src.zip")
    root = os.path.abspath(SUPERFEATURES_SRC)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                full = os.path.join(dirpath, name)
                arcname = os.path.join("superfeatures", os.path.relpath(full, root))
                zf.write(full, arcname=arcname)
    return out_path


def package_fold_metadata(walk_forward_namespace: str, dest_dir: str) -> str:
    """
    discover_folds()/GAPreprocessing.__init__ read fold_metadata.json/selected_features.txt off
    LOCAL disk (relative paths under e.g. "walk_forward_full/") - only the big train/eval CSVs
    live on HDFS (see CLAUDE.md's "Two different write paths" section). A cluster-child container
    has none of bialobog's local disk (same "/home/bvail is NOT shared with the YARN executors"
    constraint the venv/code hit), so discover_folds would find zero folds without this - tiny
    (under 2M for either namespace, confirmed by du), so shipped fresh every submission same as
    the code zip.

    Archived with the namespace directory's *contents* at the tar root (not the directory
    itself) - --archives <tar>#<name> unpacks into a directory literally named <name>, so if the
    tar's own top-level entry were ALSO named walk_forward_full, fold_metadata.json would land
    at walk_forward_full/walk_forward_full/development/fold_01/... instead of
    walk_forward_full/development/fold_01/... (same double-nesting trap pack_venv.sh's own
    comment describes for the venv archive).
    """
    out_path = os.path.join(dest_dir, f"{walk_forward_namespace}.tar.gz")
    research_root = os.path.dirname(os.path.abspath(__file__))
    namespace_dir = os.path.join(research_root, walk_forward_namespace)
    with tarfile.open(out_path, "w:gz") as tf:
        for entry in os.listdir(namespace_dir):
            tf.add(os.path.join(namespace_dir, entry), arcname=entry)
    return out_path


def submit_cluster_run(argv: list, num_threads: int, driver_memory: str,
                        walk_forward_namespace: str) -> str:
    """
    Submits `run_ga.py <argv>` as its own YARN application (--deploy-mode cluster), sized to
    `num_threads` vcores / `driver_memory` for the AM/driver container - the whole run executes
    there via a master("yarn") SparkSession requesting one deliberately tiny executor (see
    run_ga.py's _build_spark_session docstring), so this is the run's entire YARN footprint: one
    AM container sized for the real per-individual GA work, plus one small satellite executor
    container that exists only so the ApplicationMaster registers with the ResourceManager (see
    below) - not to do any meaningful compute itself.

    Returns the submitted application's YARN app id. Submission itself
    (spark.yarn.submit.waitAppCompletion=false) returns as soon as the app is accepted - it does
    NOT block for the run's actual duration, so the caller is free to submit many of these back
    to back for a seed/arm sweep.

    Historical note (fixed 2026-08-24, kept here since it explains why a "wasteful-looking" extra
    executor is actually load-bearing): this used to request ZERO executors, via
    master("local[*]") instead of master("yarn") - the theory being that a process already placed
    on its own dedicated container has no reason to ask YARN for more. That was wrong in a way
    that silently killed every full-scale run: local[*] never fires the sparkContextInitialized
    callback ApplicationMaster.runDriver() waits on before calling registerApplicationMaster()
    against the RM, so the AM never registered at all, regardless of how long real work inside it
    ran. Setting spark.yarn.am.waitTime very high (this file used to do that) only silenced the
    ApplicationMaster's own internal self-timeout while it waited for that callback - it did NOT
    make the callback fire. The RM's OWN separate liveness monitor
    (yarn.am.liveness-monitor.expiry-interval-ms, 600000ms/10min on this cluster) has no such
    override and killed every real full-scale attempt at 645-777s in regardless (confirmed
    2026-08-24 on application_1787288171816_0079-0082 - one of the four, seed 103, had already
    completed fold_01 with a real printed true-test RMSE when the RM killed it, and that work was
    lost, since the HDFS push only happens in run_ga.py's `finally` block on a process that's
    still alive to reach it). Never caught earlier because the only prior end-to-end validation
    (`--fast --dev-only`) finished in ~2-3 minutes, comfortably inside the 10-minute window -
    nothing to do with correctness, just under the wire. Requesting one real (if tiny) executor
    makes Spark's normal YarnClusterSchedulerBackend register the AM promptly and keep
    heartbeating for the run's whole lifetime, the same well-tested path every ordinary
    Spark-on-YARN cluster-mode job already relies on.

    One consequence of the fix: YARN's own SUCCEEDED/FAILED verdict for this application should
    now be trustworthy (a real YARN-registered SparkContext reports its own completion state
    correctly, unlike local[*]) - but the real signal for whether this run's fold N succeeded is
    still the same one a normal --execution local run already uses: does that fold's
    fold_result.json exist on disk (see run_ga.py's own _load_or_run_fold/fingerprint-based skip
    logic) - NOT this application's YARN state. `run_ga_sweep.py`'s own polling only ever checked
    "did Final-State leave UNDEFINED," tolerant of either verdict, so it needed no change here.
    """
    if not os.path.isfile(PACKED_VENV):
        raise FileNotFoundError(
            f"{PACKED_VENV} not found - run research/scripts/pack_venv.sh once first "
            f"(and again any time ~/pyspark-venv's installed packages change)."
        )

    with tempfile.TemporaryDirectory() as scratch:
        code_zip = package_code(scratch)
        fold_meta_tar = package_fold_metadata(walk_forward_namespace, scratch)
        run_ga_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_ga.py")

        cmd = [
            os.path.join(SPARK_HOME, "bin", "spark-submit"),
            "--master", "yarn",
            "--deploy-mode", "cluster",
            "--archives", f"{PACKED_VENV}#environment,{fold_meta_tar}#{walk_forward_namespace}",
            "--py-files", code_zip,
            "--conf", "spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python3",
            "--conf", "spark.yarn.appMasterEnv.SUPERFEATURES_CLUSTER_CHILD=1",
            "--conf", f"spark.yarn.am.cores={num_threads}",
            "--conf", f"spark.driver.memory={driver_memory}",
            "--conf", "spark.yarn.submit.waitAppCompletion=false",
            "--conf", "spark.eventLog.enabled=false",
            # Not covering up the old local[*] false-FAILED bug anymore (see this function's
            # docstring - a real master("yarn") run reports its own completion state correctly)
            # - kept anyway as a general safety net: without this, YARN's default 2-attempt retry
            # would silently rerun a genuinely-failed (possibly multi-hour) run from scratch on a
            # different node rather than surfacing the failure.
            "--conf", "spark.yarn.maxAppAttempts=1",
            run_ga_py,
            *argv,
        ]

        env = dict(os.environ)
        env["SPARK_HOME"] = SPARK_HOME
        env["HADOOP_HOME"] = HADOOP_HOME
        env["HADOOP_CONF_DIR"] = HADOOP_CONF_DIR
        env["YARN_CONF_DIR"] = HADOOP_CONF_DIR

        print(f"Submitting cluster run: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        print(output)
        if result.returncode != 0:
            raise RuntimeError(f"spark-submit failed with exit code {result.returncode}")

        match = _APP_ID_RE.search(output)
        if match is None:
            raise RuntimeError(
                "spark-submit exited 0 but no application_XXXX_XXXX id found in its output - "
                "check the log above."
            )
        return match.group(0)


if __name__ == "__main__":
    app_id = submit_cluster_run(sys.argv[1:], num_threads=8, driver_memory="8g",
                                 walk_forward_namespace="walk_forward_fast")
    print(f"Submitted: {app_id}")
