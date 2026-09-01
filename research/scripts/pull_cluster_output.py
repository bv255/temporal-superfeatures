"""
Pulls a --execution cluster run's output back down from HDFS to bialobog's local disk (run_ga.py
pushes it there on exit - see main()'s finally block) - a cluster-child container's own local
disk is ephemeral (see CLAUDE.md's "/home/bvail is NOT shared with the YARN executors"), so this
is the other half of getting results back to where _find_matching_run/compare_ga_runs.py/etc.
actually look for them.

Run from research/ (same cwd convention as run_ga.py itself):
    python3 scripts/pull_cluster_output.py ga_runs/ga_seed11_20260824-160512 --family ga_seed11

--family re-points the plain family symlink (e.g. "ga_seed11" -> the pulled-down directory), same
as run_ga.py's own _update_latest_link does for a local run - skip it if you don't want the
symlink touched (e.g. a newer local run already owns it).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/
from run_ga import _update_latest_link  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="e.g. ga_runs/ga_seed11_20260824-160512")
    parser.add_argument("--family", default=None,
                         help="if given, re-point this family symlink (e.g. ga_seed11) at the "
                              "pulled-down output_dir, same as a local run's own -latest_link")
    args = parser.parse_args()

    local_parent = os.path.dirname(args.output_dir) or "."
    os.makedirs(local_parent, exist_ok=True)

    hdfs_path = f"/user/bvail/ga-runs/{args.output_dir}"
    print(f"Pulling {hdfs_path} -> {args.output_dir} ...")
    # Full path, not bare "hdfs" - not on PATH by default on bialobog either (confirmed: `which
    # hdfs` exits 1 here) - see run_ga.py's push-side fix for the container-side version of the
    # same mistake.
    hdfs_bin = os.path.join(os.environ.get("HADOOP_HOME", "/opt/hadoop"), "bin", "hdfs")
    result = subprocess.run([hdfs_bin, "dfs", "-get", "-f", hdfs_path, local_parent],
                             capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"hdfs dfs -get failed (exit {result.returncode}) - was this run "
                          f"actually submitted with --execution cluster?")
    print(f"Pulled to {args.output_dir}")

    if args.family:
        _update_latest_link(args.family, args.output_dir)


if __name__ == "__main__":
    main()
