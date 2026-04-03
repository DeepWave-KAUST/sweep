import os
import sys

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_config import build_setup_kwargs, patch_packaging_compat


patch_packaging_compat()
os.environ["SWEEP_BUILD_CUDA"] = "1"

setup(**build_setup_kwargs(distribution_name="sweep-cuda"))
