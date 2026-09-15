import glob
import inspect
import os
import sys
from distutils import log

try:
    import packaging.utils as packaging_utils
except ImportError:
    packaging_utils = None

try:
    import setuptools._core_metadata as setuptools_core_metadata
except ImportError:
    setuptools_core_metadata = None


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_VERSION = "0.1.0"


def env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def openmp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        return []
    return ["-fopenmp"]


# Carries +PTX on purpose: a binary built for one architecture and nothing else
# dies with "no kernel image is available for execution on the device" the
# moment it meets a newer card, whereas PTX is JIT-ed by the driver and runs.
FALLBACK_CUDA_ARCH_LIST = "7.0+PTX"


def configure_cuda_arch_list():
    """Give PyTorch a GPU-arch target when it cannot work one out itself.

    A FALLBACK, not a default. When a device is visible, torch's own detection
    targets exactly the card in front of it; overriding that with a fixed list
    is how `SWEEP_BUILD_CUDA=1 pip install .` on an Ada workstation produced an
    sm_70-only extension that could not run on the machine that built it.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    override = os.environ.get("SWEEP_CUDA_ARCH_LIST")
    if override:
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return          # torch detects the card it can actually see
    except Exception:
        pass                # no torch, or a torch that cannot answer: fall back

    os.environ["TORCH_CUDA_ARCH_LIST"] = FALLBACK_CUDA_ARCH_LIST
    log.warn(
        "no CUDA device is visible and TORCH_CUDA_ARCH_LIST is not set; "
        "building for %s. The PTX makes it runnable on newer cards, but set "
        "TORCH_CUDA_ARCH_LIST (or SWEEP_CUDA_ARCH_LIST) to target them natively.",
        FALLBACK_CUDA_ARCH_LIST,
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


def build_ext_kwargs(build_cuda=None):
    """Return setup() kwargs for the optional AOT C++/CUDA extension.

    The default distribution is **JIT** (see ``sweep/_jit.py``): one ``py3-none``
    wheel ships the C++/CUDA sources and compiles ``sweep._C`` against the user's
    own torch on first use — so this returns NO ``ext_modules`` and every dep
    comes from ``pyproject.toml``. The ``SWEEP_BUILD_CUDA=1`` path is kept only
    for building optional pre-compiled fast-path wheels (e.g. a GitHub release),
    never for the PyPI wheel.
    """
    if build_cuda is None:
        build_cuda = env_flag_enabled("SWEEP_BUILD_CUDA")

    kwargs = {
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

    # Optional extra nvcc flags (e.g. -DELASTIC3D_LB_MINBLOCKS=6 to retune a
    # forward launch_bounds without editing kernel source).  Space-separated.
    extra_nvcc = os.environ.get("SWEEP_EXTRA_NVCC", "").split()

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
                    *extra_nvcc,
                ],
            },
            # RPATH so the shipped wheel resolves libtorch/libc10 against the
            # USER's torch (auditwheel --exclude keeps those libs external).
            # Belt-and-suspenders: sweep always imports torch before sweep._C.
            extra_link_args=[*omp_flags, "-Wl,-rpath,$ORIGIN/../torch/lib"],
        )
    ]
    kwargs["cmdclass"] = {
        "build_ext": SweepBuildExtension.with_options(use_ninja=True)
    }
    return kwargs
