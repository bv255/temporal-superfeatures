"""
Shared helpers for the package-port parity tests (see docs/RESTRUCTURING_TODO.md and the port
plan referenced there). Factors out the `_load_cell_source`-by-index convention already used
independently by legacy/tests/test_preprocessing_v1.py, legacy/tests/test_ga_v1.py,
tests/test_ga_methodology_additions.py, and tests/test_compare_metrics.py — those four files are
left untouched (they predate this package and still exec their own notebook cells directly), this
conftest only serves new tests added under tests/ for the superfeatures package port.
"""
import json


def load_cell_source(notebook_path: str, cell_index: int) -> str:
    with open(notebook_path) as f:
        nb = json.load(f)
    return "".join(nb["cells"][cell_index]["source"])
