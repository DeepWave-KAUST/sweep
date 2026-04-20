from __future__ import annotations

import os
import sys
from pathlib import Path


def get_import_mode(argv: list[str] | None = None) -> str:
    argv = sys.argv if argv is None else argv
    env_mode = os.environ.get("SWEEP_IMPORT_MODE")
    if env_mode in {"env", "source"}:
        return env_mode

    for i, arg in enumerate(argv):
        if arg == "--import-mode" and i + 1 < len(argv):
            value = argv[i + 1].strip().lower()
            if value in {"env", "source"}:
                return value
        if arg.startswith("--import-mode="):
            value = arg.split("=", 1)[1].strip().lower()
            if value in {"env", "source"}:
                return value

    return "env"


def configure_example_imports(examples_dir: Path) -> str:
    mode = get_import_mode()
    if mode == "source":
        src_path = str(examples_dir.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    return mode
