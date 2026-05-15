"""
determinism.py — Set every seed we can reach.

Called once per worker process. Soft-imports torch / numpy so the harness
runs even when those aren't installed (e.g. when only running parser
fuzzing tests on a minimal env).
"""

from __future__ import annotations

import os
import random


DEFAULT_SEED = 1337


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for random, numpy, torch (CPU + CUDA), and PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism. cudnn flags can be expensive on GPU
        # so we don't force them in long-running stress tests.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass
