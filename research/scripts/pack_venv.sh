#!/usr/bin/env bash
# One-time (re-run after any `pip install` into ~/pyspark-venv) setup step for
# `--execution cluster` (see ~/.claude/plans/clever-scribbling-mccarthy.md). Packs the venv's
# third-party deps into a relocatable archive that spark-submit ships to the YARN container via
# --archives, so a GA run doesn't need ~/pyspark-venv to exist on the landing node.
#
# The venv's own `superfeatures` install is editable (a .pth file pointing at
# ~/temporal-superfeatures/research/src, which won't exist on a worker node) - deliberately NOT
# excluded here. cluster_submit.py's --py-files ships a fresh zip of src/superfeatures on every
# submission instead, placed ahead of site-packages on sys.path, so the stale .pth entry is
# simply never reached. (Verified by cluster_smoke_test.py before this was relied on for real.)
set -euo pipefail

VENV=~/pyspark-venv
OUT=~/pyspark-venv.tar.gz

# venv-pack (the usual tool for this) turned out to be unusable here: 0.2.0 (PyPI's only ever
# release, unmaintained since 2020) only supports conda/virtualenv environments, not ones created
# by the stdlib `venv` module (`~/pyspark-venv` is one - see pyvenv.cfg's `command = ... -m venv`)
# - it fails with "Current environment is not a virtual environment" on this venv. Falling back to
# a plain tar instead. The venv's only path dependency is bin/python3 -> /usr/bin/python3 (a
# symlink to the system interpreter, not anything venv-pack's relocation logic would have fixed
# anyway) - this only works if the landing node has the same system Python at the same path,
# which cluster_smoke_test.py checks before this is relied on for a real run.
echo "Packing $VENV -> $OUT (this can take a few minutes, venv is ~1.6G)..."
# Archive the venv's *contents* at the tar root (not nested under a pyspark-venv/ dir) - YARN
# unpacks --archives ~/pyspark-venv.tar.gz#environment into a directory literally named
# "environment", so bin/python3 needs to land at environment/bin/python3, not
# environment/pyspark-venv/bin/python3.
tar -czf "$OUT" -C "$VENV" .
echo "Done: $(du -sh "$OUT" | cut -f1)"
