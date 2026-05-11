"""
M-00 Schema drift test.

Compares the harness's local copy of S1_FEATURES against the project's
canonical list. If they ever drift the harness's CSVs would silently
miss columns and every downstream ML test would be invalid. This test
must pass before any other ML test result is trusted.
"""

from __future__ import annotations

from test_harness.generators.flow_csv_gen import S1_FEATURES as HARNESS_FEATURES
from test_harness.utils.project_imports import soft_import


TEST_ID = "M-00"


def run() -> dict:
    monitoring, err = soft_import("monitoring")
    if monitoring is None:
        return {"test_id": TEST_ID, "name": "Schema drift",
                "severity": "CRITICAL", "verdict": "SKIPPED",
                "expected": "harness S1_FEATURES == monitoring.S1_FEATURES",
                "actual":   f"monitoring import failed: {err}"}
    project_features = list(getattr(monitoring, "S1_FEATURES", []))
    if not project_features:
        return {"test_id": TEST_ID, "name": "Schema drift",
                "severity": "CRITICAL", "verdict": "FAIL",
                "expected": "monitoring exports S1_FEATURES",
                "actual":   "monitoring.S1_FEATURES not found / empty"}
    if project_features == HARNESS_FEATURES:
        return {"test_id": TEST_ID, "name": "Schema drift",
                "severity": "CRITICAL", "verdict": "PASS",
                "expected": "feature lists identical",
                "actual":   f"both lists have {len(project_features)} features in same order"}
    only_proj    = [f for f in project_features if f not in HARNESS_FEATURES]
    only_harness = [f for f in HARNESS_FEATURES if f not in project_features]
    same_set     = set(project_features) == set(HARNESS_FEATURES)
    return {"test_id": TEST_ID, "name": "Schema drift",
            "severity": "CRITICAL", "verdict": "FAIL",
            "expected": "feature lists identical",
            "actual": ("set match but ordering differs" if same_set
                       else f"only_project={only_proj[:5]} only_harness={only_harness[:5]}"),
            "raw": {"only_in_project": only_proj,
                    "only_in_harness": only_harness,
                    "n_project": len(project_features),
                    "n_harness": len(HARNESS_FEATURES)}}
