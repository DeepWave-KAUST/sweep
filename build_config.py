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


def openmp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        return []
    return ["-fopenmp"]


def configure_cuda_arch_list():
    """Avoid PyTorch's empty GPU-arch auto-detection on login/CPU nodes."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    arch_list = os.environ.get("SWEEP_CUDA_ARCH_LIST", "7.0")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    log.warn(
        "TORCH_CUDA_ARCH_LIST is not set; defaulting to %s. "
        "Set TORCH_CUDA_ARCH_LIST or SWEEP_CUDA_ARCH_LIST to target other GPUs.",
        arch_list,
    )


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
    """Collect C++/CUDA sources for the ``sweep._C`` extension.

    Honours the ``SWEEP_SKIP_CPU`` environment variable: when set to a
    truthy value (1, true, yes, on), the heavy ``cpu/equations/*`` tree
    (~19k lines, often the build-time bottleneck) is *excluded* and a tiny
    stub is linked in its place.  The stub keeps `bindings/module.cpp`
    linking and routes every call to the CUDA path; attempting to use a
    CPU tensor raises a clear TORCH_CHECK message.

    This is intended for users who only ever run on CUDA — typically HPC
    deployments where the CPU C++ path would be dead weight.
    """
    cuda_sources = (
        glob.glob("src/sweep/csrc/cuda/common/**/*.cu", recursive=True)
        + glob.glob("src/sweep/csrc/cuda/equations/**/*.cu", recursive=True)
    )
    binding_sources = ["src/sweep/csrc/bindings/module.cpp"]

    if env_flag_enabled("SWEEP_SKIP_CPU"):
        log.warn(
            "SWEEP_SKIP_CPU=1: skipping cpu/equations/* (~19k LoC); linking "
            "cpu_binding_stub.cpp instead. CPU tensors will raise a clear "
            "error at call time."
        )
        cpu_sources = ["src/sweep/csrc/cpu/cpu_binding_stub.cpp"]
    else:
        cpu_sources = glob.glob("src/sweep/csrc/cpu/**/*.cpp", recursive=True)
        # Defensive: don't accidentally include the stub if it's globbed
        cpu_sources = [
            s for s in cpu_sources
            if not s.endswith("cpu_binding_stub.cpp")
        ]

    return cpu_sources + cuda_sources + binding_sources


def _check_ninja_on_path():
    """Print an actionable note if torch's ninja-binary probe will fail.

    Torch's ``is_ninja_available()`` shells out to ``ninja --version`` on
    PATH, NOT to the bundled Python ``ninja`` package.  If the conda env is
    not activated (e.g. invoking ``/path/to/envs/X/bin/python setup.py``
    directly), the ninja binary at ``<env>/bin/ninja`` is invisible and
    torch silently falls back to the *slow* distutils sequential build —
    ~6× slower in practice.  Loud-warn now, instead of having the user
    discover it 20 minutes into a serial compile.
    """
    import shutil

    if shutil.which("ninja") is not None:
        return  # binary on PATH — torch will use ninja, all good

    # Bundled ninja package?  Tell the user how to expose it.
    bundled = None
    try:
        import ninja as _ninja_pkg
        bundled = os.path.join(os.path.dirname(_ninja_pkg.__file__), "..", "..", "..", "..", "bin", "ninja")
        bundled = os.path.normpath(bundled)
        if not os.path.exists(bundled):
            # Try the conda env layout
            python_exec = sys.executable
            env_bin = os.path.dirname(python_exec)
            candidate = os.path.join(env_bin, "ninja")
            bundled = candidate if os.path.exists(candidate) else None
    except ImportError:
        pass

    msg = (
        "ninja binary not found on PATH.  Torch will fall back to the slow "
        "distutils sequential build (~6× slower).  "
    )
    if bundled and os.path.exists(bundled):
        msg += (
            f"A ninja binary is bundled at {bundled}; either activate your "
            "conda env (`conda activate <env>`) so PATH includes it, or run "
            f"PATH='{os.path.dirname(bundled)}:$PATH' python setup.py ..."
        )
    else:
        msg += "Install ninja-build (`apt install ninja-build` or `pip install ninja`)."
    log.warn(msg)
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def make_build_extension(BuildExtension):
    def emit(message):
        print(message, file=sys.stderr, flush=True)
        log.info(message)

    class SweepBuildExtension(BuildExtension):
        def run(self):
            self.verbose = max(getattr(self, "verbose", 1), 2)
            _check_ninja_on_path()
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
            # Companion packages — each lives in its own repo and is
            # installable on its own. `sweep[full]` pulls the whole
            # ecosystem in one shot. Names match the `[project] name`
            # field in each companion repo's pyproject.toml.
            #
            # NOTE: until every companion is on PyPI, `pip install sweep[full]`
            # will fail to resolve. For local development, use
            # `scripts/install_ecosystem.sh` (editable installs in dep order)
            # instead.
            "full": [
                "torch",
                "sweep-loss",
                "sweep-opt",
                "sweep-io",
                "sweep-preproc",
                "sweep-runner",
                "sweep-tasks",
                "sweep-zoo",
                "sweep-viz",
                "sweep-nn",
            ],
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
    omp_flags = openmp_flags()
    configure_cuda_arch_list()

    kwargs["ext_modules"] = [
        CUDAExtension(
            name="sweep._C",
            sources=get_sources(),
            include_dirs=[
                os.path.join(ROOT_DIR, "src/sweep/csrc"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/bindings"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/shared"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/cuda"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/cuda/common"),
                os.path.join(ROOT_DIR, "src/sweep/csrc/cuda/equations"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-attributes", *omp_flags],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--threads=16",
                    "-Xcompiler=-Wno-deprecated-declarations",
                ],
            },
            extra_link_args=omp_flags,
        )
    ]
    kwargs["cmdclass"] = {
        "build_ext": SweepBuildExtension.with_options(use_ninja=True)
    }
    return kwargs
