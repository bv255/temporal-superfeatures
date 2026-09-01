"""
Throwaway validation script for the `--execution cluster` design (see
~/.claude/plans/clever-scribbling-mccarthy.md) - NOT part of the pipeline, not imported by
anything. Submit this by hand via the same spark-submit --deploy-mode cluster recipe
run_ga.py's cluster_submit.py will use, before wiring anything into the real pipeline. Checks,
in order, the open questions the plan flagged as unverified:

1. Did this process actually land on a skrzat* worker node, not bialobog?
2. Does the shipped venv archive + --py-files zip make xgboost/pandas/numpy/superfeatures
   importable here?
3. Does a SparkSession built with .master("local[*]") (no YARN executors requested) still read
   HDFS successfully, using only HADOOP_CONF_DIR (no assumption that /opt/spark or /opt/hadoop
   exist locally on this node - if SPARK_HOME points at a path that doesn't exist here, this
   will fail loudly rather than silently, telling us to fall back to pyspark's own bundled jars).

Run manually first:
    spark-submit --master yarn --deploy-mode cluster \\
      --archives ~/pyspark-venv.tar.gz#environment \\
      --py-files <path to a fresh zip of src/superfeatures> \\
      --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python3 \\
      --conf spark.yarn.am.cores=4 \\
      --conf spark.driver.memory=2g \\
      --conf spark.yarn.maxAppAttempts=1 \\
      research/scripts/cluster_smoke_test.py
Note: this will report FAILED even on success (see cluster_submit.py's submit_cluster_run
docstring) - check the printed [smoke] lines via `yarn logs -applicationId <id>`, not the exit
status. maxAppAttempts=1 above stops YARN from silently retrying it on that false signal.
Then check its output via `yarn logs -applicationId <id>`.
"""
import os
import socket
import sys


def main():
    print(f"[smoke] hostname = {socket.gethostname()}")
    print(f"[smoke] sys.executable = {sys.executable}")
    print(f"[smoke] sys.path[:5] = {sys.path[:5]}")

    print("[smoke] importing xgboost/pandas/numpy/superfeatures...")
    import numpy
    import pandas
    import xgboost
    print(f"[smoke]   numpy {numpy.__version__}, pandas {pandas.__version__}, "
          f"xgboost {xgboost.__version__}")
    import superfeatures
    print(f"[smoke]   superfeatures imported from: {superfeatures.__file__}")

    print("[smoke] setting Hadoop env vars...")
    os.environ.setdefault("HADOOP_HOME", "/opt/hadoop")
    os.environ.setdefault("HADOOP_CONF_DIR", "/opt/hadoop-3.4.1/etc/hadoop")
    os.environ.setdefault("YARN_CONF_DIR", "/opt/hadoop-3.4.1/etc/hadoop")
    spark_home_exists = os.path.isdir("/opt/spark")
    print(f"[smoke] /opt/spark exists on this node: {spark_home_exists}")
    if spark_home_exists:
        os.environ.setdefault("SPARK_HOME", "/opt/spark")
    else:
        print("[smoke] /opt/spark NOT found here - relying on pyspark's own bundled jars "
              "instead of an external SPARK_HOME (this is the fallback the plan flagged).")

    print("[smoke] building local[*] SparkSession...")
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("cluster_smoke_test")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    print("[smoke] SparkSession ready, reading a known HDFS fold file...")
    df = spark.read.option("header", "true").csv(
        "/user/bvail/walk_forward_fast/final_test/fold_01/train.csv", inferSchema=False
    )
    n = df.count()
    print(f"[smoke] read train.csv, row count = {n}")
    print("[smoke] SUCCESS" if n > 0 else "[smoke] FAILURE: zero rows read")
    # Deliberately NOT calling spark.stop() here - see the plan/finding notes: an explicit
    # stop() right before a short cluster-mode PySpark script exits appears to race the
    # ApplicationMaster's own "did the driver ever initialize a SparkContext" check, reporting
    # FAILED even though everything above completed correctly.


if __name__ == "__main__":
    main()
