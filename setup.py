import os
import sys

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_config import build_ext_kwargs, patch_packaging_compat


patch_packaging_compat()

setup(**build_ext_kwargs())
