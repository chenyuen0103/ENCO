#!/usr/bin/env python3
"""Compatibility shim for scripts/eval_cd_configs.py."""
from pathlib import Path
import importlib.util


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_cd_configs.py"
_SPEC = importlib.util.spec_from_file_location("_enco_scripts_eval_cd_configs", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {_SCRIPT}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

globals().update({k: v for k, v in vars(_MODULE).items() if not k.startswith("__")})

if __name__ == "__main__":
    _MODULE.main()
