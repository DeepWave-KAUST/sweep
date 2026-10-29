"""Every code block of the bundled agent skill must run (skills/run_snippets.py)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "sweep-solver"

_spec = importlib.util.spec_from_file_location("run_snippets", ROOT / "skills" / "run_snippets.py")
run_snippets = importlib.util.module_from_spec(_spec)
sys.modules["run_snippets"] = run_snippets  # dataclasses look the module up
_spec.loader.exec_module(run_snippets)


@pytest.mark.parametrize("md", run_snippets.skill_files(SKILL),
                         ids=lambda p: str(p.relative_to(SKILL)))
def test_skill_snippets_run(md):
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    failures = run_snippets.run_file(md, timeout=300, env=env)
    assert not failures, "\n\n".join(failures)
