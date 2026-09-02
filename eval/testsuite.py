"""
testsuite.py
------------
Loads the YAML test suite into plain dicts. Kept dead simple on purpose —
adding a new test case means editing data/testsuite.yaml, no code changes.
"""

from pathlib import Path

import yaml

TESTSUITE_PATH = Path(__file__).resolve().parent.parent / "data" / "testsuite.yaml"


def load_test_cases() -> list[dict]:
    with open(TESTSUITE_PATH) as f:
        data = yaml.safe_load(f)
    return data["test_cases"]
