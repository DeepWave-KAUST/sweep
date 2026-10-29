import os
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_config import (
    build_ext_kwargs,
    patch_packaging_compat,
    require_shipped_core,
    wheel_platform_kwargs,
)


class build_py_with_skill(build_py):
    """Ship skills/sweep-solver in the wheel as sweep/_skill/sweep-solver."""

    def run(self):
        super().run()
        import shutil

        src = os.path.join(ROOT_DIR, "skills", "sweep-solver")
        dest = os.path.join(self.build_lib, "sweep", "_skill", "sweep-solver")
        if not os.path.isfile(os.path.join(src, "SKILL.md")):
            raise RuntimeError(f"{src} is missing: the wheel must ship the skill")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


patch_packaging_compat()
require_shipped_core()

kwargs = build_ext_kwargs()
kwargs["cmdclass"] = {**kwargs.get("cmdclass", {}), "build_py": build_py_with_skill}
setup(**kwargs, **wheel_platform_kwargs())
