"""
paths.py — Project / harness path resolution.

Resolves the repository root once. Every other module imports from here so
nothing has to guess. The harness is designed to live INSIDE the project
repo at <project_root>/test_harness/, but it also works when copied
elsewhere as long as PROJECT_ROOT is set via env var BOTNET_PROJECT_ROOT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
# Harness root = parent of the utils package.
HARNESS_ROOT: Path = Path(__file__).resolve().parents[1]

# Repo root = parent of the harness package directory.
# We need both on sys.path so that:
#   - `import test_harness.*`          works from the repo root
#   - `import monitoring`, `import app.*`  works from the repo root too
_REPO_ROOT: Path = HARNESS_ROOT.parent

for _p in (str(_REPO_ROOT), str(HARNESS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Project root: env var wins, else assume harness sits inside the repo.
_env = os.environ.get("BOTNET_PROJECT_ROOT")
if _env:
    PROJECT_ROOT: Path = Path(_env).resolve()
else:
    # Walk up from harness/utils looking for monitoring.py — that's the
    # repo's load-bearing top-level file.
    p = HARNESS_ROOT
    found: Path | None = None
    for _ in range(6):
        if (p / "monitoring.py").exists():
            found = p
            break
        if p.parent == p:
            break
        p = p.parent
    PROJECT_ROOT = found if found is not None else HARNESS_ROOT.parent

# Standard project subdirs (these may or may not exist on a given checkout).
APP_DIR        = PROJECT_ROOT / "app"
MODELS_DIR     = PROJECT_ROOT / "models"
SRC_DIR        = PROJECT_ROOT / "src"
EVAL_DIR       = PROJECT_ROOT / "evaluation"
DATA_DIR       = PROJECT_ROOT / "data"

# Harness output dirs.
LOGS_DIR       = HARNESS_ROOT / "logs"
ARTIFACTS_DIR  = HARNESS_ROOT / "artifacts"
REPORTS_DIR    = HARNESS_ROOT / "reports"
OUTPUTS_DIR    = HARNESS_ROOT / "outputs"
CONFIGS_DIR    = HARNESS_ROOT / "configs"

for _d in (LOGS_DIR, ARTIFACTS_DIR, REPORTS_DIR, OUTPUTS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def for_test(test_id: str) -> dict[str, Path]:
    """Return a dict of per-test-id directories, creating them on demand."""
    log_dir = LOGS_DIR / test_id
    art_dir = ARTIFACTS_DIR / test_id
    log_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    return {"logs": log_dir, "artifacts": art_dir}


def project_path_on_sys_path() -> str:
    """Return PROJECT_ROOT as str — the form sys.path wants."""
    return str(PROJECT_ROOT)
