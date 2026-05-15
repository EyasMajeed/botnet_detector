"""
runner_subprocess.py — Child-process entrypoint.

Reads the JSON payload (target, kwargs, result_path), imports
"module.path:function", calls function(**kwargs), and writes the return
value to result_path. Any uncaught exception is captured into the
result file with a traceback so the orchestrator can render it.

Invoked by isolation.run_isolated, never by hand.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

# Determinism applies inside the child as well.
try:
    from test_harness.utils.determinism import seed_everything
    seed_everything()
except Exception:
    pass


def _resolve(target: str):
    if ":" not in target:
        raise ValueError(f"target must be 'module:function', got {target!r}")
    mod_name, func_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{mod_name} has no attribute {func_name}")
    return getattr(mod, func_name)


def main(payload_path: str) -> int:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    target  = payload["target"]
    kwargs  = payload.get("kwargs", {})
    rpath   = Path(payload["result_path"])

    rpath.parent.mkdir(parents=True, exist_ok=True)
    out: dict
    rc = 0
    try:
        fn  = _resolve(target)
        out = fn(**kwargs) or {}
        if not isinstance(out, dict):
            out = {"_value": out}
        out.setdefault("_ok", True)
    except SystemExit as e:
        # Honour explicit exits but still record them.
        rc = int(e.code) if isinstance(e.code, int) else 1
        out = {"_ok": False, "_exit_code": rc, "_target": target}
    except BaseException as e:                            # noqa: BLE001
        rc = 1
        out = {
            "_ok": False,
            "_target": target,
            "_exception": type(e).__name__,
            "_message": str(e),
            "_traceback": traceback.format_exc(),
        }
    finally:
        try:
            rpath.write_text(json.dumps(out, default=str, indent=2),
                             encoding="utf-8")
        except Exception:
            # Last resort — write a minimal record so orchestrator sees
            # something, even if the result was unserialisable.
            rpath.write_text(json.dumps({"_ok": False,
                                         "_message": "unserialisable result"}),
                             encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
