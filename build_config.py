import glob
import inspect
import os
import sys
from distutils import log

from setuptools import find_packages

try:
    import packaging.utils as packaging_utils
except ImportError:
    packaging_utils = None

try:
    import setuptools._core_metadata as setuptools_core_metadata
except ImportError:
    setuptools_core_metadata = None


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_VERSION = "0.0.1"


def env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_metadata_only_invocation():
    metadata_commands = {"egg_info", "dist_info", "prepare_metadata_for_build_wheel"}
    return any(arg in metadata_commands for arg in sys.argv[1:])


def patch_packaging_compat():
    if packaging_utils is None:
        return

    signature = inspect.signature(packaging_utils.canonicalize_version)
    if "strip_trailing_zero" in signature.parameters:
        return

    original = packaging_utils.canonicalize_version

    def canonicalize_version_compat(version, strip_trailing_zero=True):
        return original(version)

    packaging_utils.canonicalize_version = canonicalize_version_compat
    if setuptools_core_metadata is not None:
        setuptools_core_metadata.canonicalize_version = canonicalize_version_compat


def get_sources():
    return (
        glob.glob("src/sweep/csrc/common/**/*.cu", recursive=True)
        + glob.glob("src/sweep/csrc/equations/**/*.cu", recursive=True)
        + ["src/sweep/csrc/bindings.cpp"]
    )


def make_build_extension(BuildExtension):
    def emit(message):
        print(message, file=sys.stderr, flush=True)
        log.info(message)

    class SweepBuildExtension(BuildExtension):
        def run(self):
            self.verbose = max(getattr(self, "verbose", 1), 2)
            for ext in self.extensions:
                sources = list(getattr(ext, "sources", []))
                emit(f"Building CUDA extension '{ext.name}' with {len(sources)} source files")
                for index, source in enumerate(sources, start=1):
                    emit(f"  [{index}/{len(sources)}] {source}")
            super().run()

        def build_extensions(self):
            self.verbose = max(getattr(self, "verbose", 1), 2)
            emit("Starting C++/CUDA compilation")
            super().build_extensions()
            emit("Finished C++/CUDA compilation")

    return SweepBuildExtension


def build_setup_kwargs(distribution_name="sweep", build_cuda=None):
    if build_cuda is None:
        build_cuda = env_flag_enabled("SWEEP_BUILD_CUDA")

    kwargs = {
        "name": distribution_name,
        "version": PACKAGE_VERSION,
        "description": "Seismic Wave Equation Exploration Platform",
        "long_description": open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8").read(),
        "long_description_content_type": "text/markdown",
        "author": "Shaowen Wang",
        "author_email": "shaowen.wang@kaust.edu.sa",
        "url": "https://github.com/DeepWave-KAUST/sweep",
        "project_urls": {
            "Homepage": "https://github.com/DeepWave-KAUST/sweep",
            "Issues": "https://github.com/DeepWave-KAUST/sweep/issues",
        },
        "license": "MIT",
        "python_requires": ">=3.8",
        "package_dir": {"": "src"},
        "packages": find_packages(
            where="src",
            include=["sweep", "sweep.*", "geophyai", "geophyai.*"],
            exclude=["sweep.csrc", "sweep.csrc.*"],
        ),
        "include_package_data": False,
        "entry_points": {
            "console_scripts": ["sweep=sweep.cli:main"],
        },
        "extras_require": {
            "cuda": ["torch"],
            "torch": ["torch"],
            "jax": ["jax", "jaxlib"],
            "all": ["torch", "jax", "jaxlib"],
        },
        "ext_modules": [],
        "cmdclass": {},
    }

    if not build_cuda:
        return kwargs

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        if is_metadata_only_invocation():
            log.warn(
                "Skipping CUDA extension setup during metadata generation because PyTorch "
                "is not installed in the current build environment."
            )
            return kwargs

        raise RuntimeError(
            "Building sweep with SWEEP_BUILD_CUDA=1 requires PyTorch to be installed first, "
            "because the CUDA extension uses torch.utils.cpp_extension. "
            "In a pure JAX environment, install without SWEEP_BUILD_CUDA or install PyTorch "
            "before building the CUDA extension."
        ) from exc

    SweepBuildExtension = make_build_extension(BuildExtension)

    kwargs["ext_modules"] = [
        CUDAExtension(
            name="sweep._C",
            sources=get_sources(),
            include_dirs=[
                os.path.join(ROOT_DIR, "src/sweep/csrc"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/common"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/equations"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-attributes"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--threads=16",
                ],
            },
        )
    ]
    kwargs["cmdclass"] = {
        "build_ext": SweepBuildExtension.with_options(use_ninja=True)
    }
    return kwargs
