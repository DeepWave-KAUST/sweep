import os
import sys

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_config import (
    build_ext_kwargs,
    patch_packaging_compat,
    require_shipped_core,
    wheel_platform_kwargs,
)


patch_packaging_compat()
require_shipped_core()

setup(**build_ext_kwargs(), **wheel_platform_kwargs())
