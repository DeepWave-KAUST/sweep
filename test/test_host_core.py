"""The host (CPU) core: the same ``csrc`` tree as the CUDA core, built by g++
through the host stand-ins (``csrc/host/``) and driven by the same ctypes layer
(``sweep.backend.c``) from CPU tensors.

What is pinned here and why:

* the core says what it is -- ``sweep_core_device() == "cpu"``, the shim's ABI,
  the SAME entry table as the CUDA core, and ``sweep_entry_available``
  false exactly for the equations ``hostcore.HOST_UNPORTED`` leaves out (the
  stub TU and the Python table cannot disagree);
* acoustic2d in every memory mode agrees with eager on the CPU (the reference
  that shares no code with the compiled backward) and, when a GPU is present,
  with the CUDA core at float32 level: same kernels, same drivers, so a
  gradient magnitude that drifts by more than ULPs is a defect.  The deleted
  CPU engine (af58082b) was off by 0.366x here and nothing caught it; the
  magnitude ratio below would have;
* the host core is bit-reproducible run to run (the one atomic on this path,
  the point source, has one writer per cell);
* staging options that move bytes between the compute tier and the host are
  refused on a CPU device -- there is no second tier;
* an equation the host core does not build falls back to eager with a
  warning naming the host core, instead of an error at the first call;
* ``impl='auto'`` takes the host core an explicit ``impl='c'`` runs on --
  shipped, cached, or built on first use with the compiler's output off the
  terminal -- and runs exactly what it runs.

The core is built on first use when a C++ compiler is at hand (~30 s on a
workstation, a few minutes on a CI box); ``SWEEP_TEST_HOST_CORE=0`` skips the
module instead.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradient_cases as gc  # noqa: E402
import backend_gradient_matrix as bgm  # noqa: E402

from sweep.backend.c import hostcore, loader  # noqa: E402
from sweep.propagator.options import BoundarySaving, Ckpt, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("SWEEP_TEST_HOST_CORE", "1").lower() in ("0", "false", "no", "off")
    or not hostcore.available(),
    reason="no host (CPU) core at hand and none buildable (SWEEP_TEST_HOST_CORE=0, or no g++)",
)

CPU = torch.device("cpu")


def _cuda_core_ready() -> bool:
    """A CUDA core at hand WITHOUT compiling -- shipped, or cached and current.
    The CUDA comparisons skip otherwise: a multi-minute nvcc build is not a
    test of the host core (and ``loader.core_lib("cuda")`` would run one)."""
    if not torch.cuda.is_available():
        return False
    from sweep.backend.c import jit
    try:
        return jit._shipped_core()[0] is not None or jit._cached_local_core() is not None
    except Exception:
        return False


CFG = gc.CUDA_SUITE_CONFIG          # 48x56, nt=120, so=4, abcn=30: the gate's grid
ACOUSTIC = gc.CASES[0]
assert ACOUSTIC.name == "acoustic2d"
MODES = ("full", "bs", "ckpt_chunk", "ckpt_recursive")


def _c_solver(case, mode, device):
    """``gradient_cases.make_solver`` with the one device-dependent knob: on a
    CPU device the boundary ring lives on the compute device ('gpu'), the
    CUDA matrix stages it through host RAM."""
    eq = gc.make_equation(case, device, CFG)
    common = dict(shape=case.shape, abcn=CFG.abcn, dh=10.0, dt=CFG.dt, dev=device, pml_type=case.pml_type,
                  nt=CFG.nt, B=1, receiver_type=case.receiver_type, source_type=case.source_type)
    if mode == "full":
        return PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **common)
    if mode == "bs":
        return PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": True, "storage": "gpu"}, **common)
    if mode == "ckpt_chunk":
        opts = CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=CFG.ckpt_chunks)))
        return PropTorch(eq, backend="torch", impl="c", cuda_options=opts, **common)
    if mode == "ckpt_recursive":
        opts = CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="recursive", count=CFG.ckpt_count)))
        return PropTorch(eq, backend="torch", impl="c", cuda_options=opts, **common)
    raise ValueError(mode)


def _run(solver, case, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gc.run_case(solver, case, device, CFG)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _ratio(a, b):
    return (a.double().norm() / b.double().norm().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------- #
# 1. identity: device, ABI, entry table, availability
# --------------------------------------------------------------------------- #
class TestIdentity:
    def test_core_reports_cpu_and_the_shim_abi(self):
        from sweep.backend.c import abi
        lib = loader.core_lib("cpu")
        assert lib.sweep_core_device() == b"cpu"
        assert lib.sweep_core_abi_version() == abi.ABI_VERSION
        from sweep.backend.c import jit
        import re
        m = re.search(r"SWEEP_ENTRY_COUNT\s*=\s*(\d+)", (jit._CSRC / "core" / "capi.h").read_text())
        assert lib.sweep_entry_count() == int(m.group(1))

    def test_unavailable_entries_are_exactly_the_unported_equations(self):
        lib = loader.core_lib("cpu")
        names = [lib.sweep_entry_name(i).decode() for i in range(lib.sweep_entry_count())]
        avail = {n: lib.sweep_entry_available(i) for i, n in enumerate(names)}
        unported = {n for n in names if any(n.startswith(c + "_") for c in hostcore.HOST_UNPORTED_NAMES)}
        assert unported, "the unported list names at least visco_acoustic2d"
        assert {n for n, a in avail.items() if a == 0} == unported
        assert all(a == 1 for n, a in avail.items() if n not in unported)
        assert lib.sweep_entry_available(len(names)) == -1 and lib.sweep_entry_available(-1) == -1
        for c_name in hostcore.HOST_UNPORTED_NAMES:
            assert not hostcore.equation_available(c_name)
        assert hostcore.equation_available("acoustic2d") and hostcore.equation_available(None)

    @pytest.mark.skipif(not _cuda_core_ready(), reason="no CUDA core at hand without compiling")
    def test_entry_table_is_the_cuda_cores_table(self):
        from sweep.backend.c.entries import _entry_table
        cpu = loader.core_lib("cpu")
        cuda = loader.core_lib("cuda")
        assert _entry_table(cpu) == _entry_table(cuda)
        assert cuda.sweep_core_device() == b"cuda"

    def test_a_stub_entry_throws_a_message_naming_the_equation(self):
        """The unported equation's entries exist (the dispatcher names them)
        and fail as a message through sweep_call, never a missing symbol."""
        import ctypes
        lib = loader.core_lib("cpu")
        names = [lib.sweep_entry_name(i).decode() for i in range(lib.sweep_entry_count())]
        entry = names.index("visco_acoustic2d_forward")
        from sweep.backend.c import abi
        core, out = abi.ForwardInputCore(), abi.ForwardOutputCore()
        err = ctypes.create_string_buffer(loader._ERR_CAP)
        rc = lib.sweep_call(entry, ctypes.addressof(core), ctypes.addressof(out), err, loader._ERR_CAP)
        assert rc != 0 and b"visco_acoustic2d is not built into the host (CPU) core" in err.value
        assert lib.sweep_visco_fft_workspace_bytes(1, 16, 16) == 0

    def test_stub_generator_reads_every_equation_header(self):
        from sweep.backend.c import jit
        eqs = sorted(p.name for p in (jit._CSRC / "cuda" / "equations").iterdir() if p.is_dir())
        text = hostcore.stub_source(jit._CSRC, eqs)
        n_decl = sum(len(decls) for eq in eqs
                     for _ns, decls in hostcore.parse_equation_header(jit._CSRC / "cuda" / "equations" / eq / f"{eq}.h"))
        assert text.count("SWEEP_CHECK(false") == n_decl >= 100
        assert "bool sweep_entry_is_stub(int entry)" in text

    def test_diagnostics_mention_the_host_core(self):
        from sweep.backend.torch import binding
        d = binding.diagnostics()
        assert d["host_core"]["available"] is True
        assert d["host_core"]["unported"] == list(hostcore.HOST_UNPORTED_EQUATIONS)

    def test_the_wheel_drawer_gets_the_core_and_its_sidecar_only(self, tmp_path, monkeypatch):
        """``python -m sweep.build --cpu --core`` builds in the cache and files
        the .so and its core.json -- what the wheel ships, and what 'auto'
        takes -- not the build's objects (it used to build in the drawer)."""
        if hostcore.cached_local_core() is None:
            pytest.skip("no current cached host core: the drawer test would rebuild it")
        import hashlib
        import json
        from sweep import build
        drawer = tmp_path / "lib" / "cpu"
        assert build.main(["--cpu", "--core", "--out", str(drawer)]) == 0
        assert sorted(p.name for p in drawer.iterdir()) == ["core.json", "libsweep_core.so"]
        so = drawer / "libsweep_core.so"
        side = json.loads((drawer / "core.json").read_text())
        assert side["sha256"] == hashlib.sha256(so.read_bytes()).hexdigest()
        assert hostcore._shipped_fits(side) == ""
        monkeypatch.setenv("SWEEP_CORE_CPU", str(so))
        assert hostcore.shipped_core() == (so.resolve(), "ok (SWEEP_CORE_CPU)")


# --------------------------------------------------------------------------- #
# 2. acoustic2d on the host core: eager, the CUDA core, itself
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def eager_cpu():
    return _run(bgm.make_eager_solver(ACOUSTIC, "full", CPU, CFG), ACOUSTIC, CPU)


@pytest.fixture(scope="module")
def host_results():
    return {mode: _run(_c_solver(ACOUSTIC, mode, CPU), ACOUSTIC, CPU) for mode in MODES}


class TestAcoustic2D:
    def test_resolves_to_c_on_a_cpu_device(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)        # a demotion warning here is a failure
            warnings.simplefilter("ignore", DeprecationWarning)  # the case library's legacy spelling
            solver = _c_solver(ACOUSTIC, "full", CPU)
        assert solver.impl == "c"

    @pytest.mark.parametrize("mode", MODES)
    def test_matches_eager_cpu(self, mode, eager_cpu, host_results):
        """The eager leg shares no code with the compiled backward: the same
        tolerances the gradient matrix uses (test/backend_gradient_matrix.py)."""
        rec_e, g_e = eager_cpu
        rec_c, g_c = host_results[mode]
        assert _rel(rec_c, rec_e) < 1e-3
        for name in ("vp", "wavelet"):
            assert _cos(g_c[name], g_e[name]) > 0.9999, name
            assert abs(_ratio(g_c[name], g_e[name]) - 1) < 1e-3, name

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the CUDA core")
    @pytest.mark.parametrize("mode", MODES)
    def test_matches_the_cuda_core(self, mode, host_results):
        """Same kernels, same drivers: float32-level agreement, and a gradient
        MAGNITUDE within 1e-4 of the CUDA core's -- the check the deleted CPU
        engine (0.366x) would have failed.  Not bitwise: nvcc --use_fast_math
        contracts FMAs and approximates divisions.  (The checkpoint modes once
        needed 1e-4 here: their replay updated psi in place, a race between
        cells that the two cores resolved differently -- TestCheckpointReplay.)
        """
        if not _cuda_core_ready():
            pytest.skip("no CUDA core at hand without compiling")
        cuda = torch.device("cuda")
        ref = _run(_c_solver(ACOUSTIC, mode, cuda), ACOUSTIC, cuda)
        rec_c, g_c = host_results[mode]
        rec_r, g_r = ref
        assert _rel(rec_c, rec_r) < 1e-5
        assert _rel(g_c["vp"], g_r["vp"]) < 1e-5
        assert _rel(g_c["wavelet"], g_r["wavelet"]) < 1e-5
        for name in ("vp", "wavelet"):
            assert _cos(g_c[name], g_r[name]) > 1 - 1e-7, name
            assert abs(_ratio(g_c[name], g_r[name]) - 1) < 1e-4, name

    def test_memory_modes_agree_with_each_other(self, host_results):
        rec0, g0 = host_results["full"]
        for mode in MODES[1:]:
            rec, g = host_results[mode]
            assert torch.equal(rec, rec0), mode          # the forward is the same code in every mode
            assert torch.equal(g["wavelet"], g0["wavelet"]), mode
            if mode.startswith("ckpt"):                  # the replay is the forward, bit for bit
                assert torch.equal(g["vp"], g0["vp"]), mode
            else:                                        # bs: the reconstruction is not
                assert _rel(g["vp"], g0["vp"]) < 1e-4, mode

    def test_run_to_run_bit_exact(self, host_results):
        rec, g = host_results["full"]
        rec2, g2 = _run(_c_solver(ACOUSTIC, "full", CPU), ACOUSTIC, CPU)
        assert torch.equal(rec, rec2)
        assert torch.equal(g["vp"], g2["vp"]) and torch.equal(g["wavelet"], g2["wavelet"])


# --------------------------------------------------------------------------- #
# 3. what a CPU device refuses or demotes
# --------------------------------------------------------------------------- #
def _suite_run(solver, mode, device):
    """One solver_gradient_mode_suite config (the bitgate's grid and seeds):
    the record and the model gradients."""
    import tempfile
    import numpy as np
    import solver_gradient_mode_suite as S
    torch.manual_seed(0)
    np.random.seed(0)
    spec, scenario = S.SOLVERS[solver], S.SCENARIOS["interior"]
    args = S.build_parser().parse_args([])
    args.no_plot = True
    shape = S.shape_for(spec, args)
    sources, receivers = S.make_geometry(spec, shape, scenario, args)
    _, init_models, grad_flags = S.make_models(spec, shape)
    wavelet = torch.tensor(S.ricker(args.nt, args.dt, args.freq, args.delay), device=device)
    prop = S.build_solver(spec, "c", mode, scenario, shape, device, args,
                          Path(tempfile.mkdtemp(prefix="host_core_")), f"{solver}_{mode}")
    assert prop.impl == "c"
    fwd = S.run_forward(prop, wavelet, sources, receivers, init_models, device)
    observed = torch.randn(*fwd.shape, generator=torch.Generator().manual_seed(1234)) * 1e-3
    grad = S.run_gradient(prop, wavelet, sources, receivers, observed, init_models, grad_flags,
                          spec.model_names, device)
    return fwd, grad["grads"]


def _tiny_forward(device, impl):
    """acoustic2d forward whose whole wavefield is denormal: the wavelet is
    scaled to 1e-39 (float32's smallest normal is 1.18e-38)."""
    import numpy as np
    from sweep.equations import Acoustic
    nt, n = 60, 40
    eq = Acoustic(spatial_order=4, device=device, backend="torch")
    kw = dict(use_ckpt=False, boundary_saving_config={"enabled": False}) if impl == "c" else dict(use_compile=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop = PropTorch(eq, backend="torch", impl=impl, shape=(n, n), abcn=10, dh=10.0, dt=1e-3,
                         dev=device, nt=nt, B=1, **kw)
    assert prop.impl == impl
    t = np.arange(nt) * 1e-3 - 0.02
    wav = torch.tensor(1e-39 * (1 - 2 * (np.pi * 15 * t) ** 2) * np.exp(-(np.pi * 15 * t) ** 2),
                       dtype=torch.float32, device=device)
    src = np.array([[[n // 2, n // 2]]], dtype=np.int64)
    rec = np.array([[[x, n // 2 + 3] for x in range(4, n - 4, 4)]], dtype=np.int64)
    vp = torch.full((n, n), 2000.0, device=device)
    with torch.no_grad():
        return prop(wav, src, rec, models=[vp]).cpu()


class TestDenormals:
    """The host core flushes denormals to zero, as the CUDA core does
    (--use_fast_math), and leaves the caller's floating-point mode as it found
    it.  Denormal operands cost x86 a microcode assist each; a decaying
    wavefield is full of them (measured 2.3-2.5x on 1500-step runs)."""

    def test_the_host_core_flushes_like_the_device(self):
        host = _tiny_forward(CPU, "c")
        assert torch.count_nonzero(host) == 0, "a denormal wavefield survived the host core"
        eager = _tiny_forward(CPU, "eager")
        assert torch.count_nonzero(eager) > 0, "eager (IEEE) should keep the denormal record"
        if _cuda_core_ready():
            assert torch.count_nonzero(_tiny_forward(torch.device("cuda"), "c")) == 0

    def test_the_callers_mode_is_restored(self):
        import numpy as np
        _tiny_forward(CPU, "c")
        # numpy runs in this (the calling) thread: a leaked FTZ/DAZ would make
        # the product 0 and the subnormal compare equal to 0.
        x = np.float32(1e-40)
        assert x * np.float32(1.0) != 0.0 and x > 0.0


class TestLossyBoundaryStorage:
    """Boundary saving with fp16 / bf16 / int8 storage on the host core.

    The int8 and fp16 payloads are normalised per block by the block's max|v|
    (boundarysaver.cu).  The device kernels run flush-to-zero (--use_fast_math),
    so a block whose max is denormal reads as an all-zero block; the host keeps
    IEEE denormals, and its first port let such a max through the ``> 0`` guard,
    overflowed 1/max to inf and wrote 0 * inf = NaN into the payload.  elastic2d
    hits it (its velocity faces hold values that small): every gradient came
    back NaN with fp16 storage.  Now the host reads that block as zeros too.
    """

    @pytest.mark.parametrize("mode", ["bs_gpu_fp16", "bs_gpu_bf16", "bs_gpu_int8"])
    def test_elastic2d_gradients_are_finite_and_match_the_cuda_core(self, mode):
        rec, grads = _suite_run("elastic2d", mode, CPU)
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"{name} gradient has non-finite values"
        if not _cuda_core_ready():
            return
        rec_c, grads_c = _suite_run("elastic2d", mode, torch.device("cuda"))
        # Lossy storage: a few quantisation buckets flip between the two cores
        # (inputs differ at ULP level).  Measured <= 2e-5 for elastic2d; the
        # storage's own error against fp32 boundary saving is 4e-4..9e-3.
        assert _rel(rec, rec_c) < 1e-5
        for name in grads_c:
            assert _rel(grads[name], grads_c[name]) < 1e-3, name


class TestCpuDevicePolicy:
    def _common(self, eq):
        return dict(shape=ACOUSTIC.shape, abcn=CFG.abcn, dh=10.0, dt=CFG.dt, dev=CPU, pml_type="cpmlr",
                    nt=CFG.nt, B=1)

    def test_host_staging_is_refused_not_downgraded(self):
        eq = gc.make_equation(ACOUSTIC, CPU, CFG)
        with pytest.raises(NotImplementedError, match="storage='cpu'"):
            PropTorch(eq, backend="torch", impl="c", memory=BoundarySaving(storage="cpu"), **self._common(eq))
        with pytest.raises(NotImplementedError, match="storage='disk'"):
            PropTorch(eq, backend="torch", impl="c", memory=BoundarySaving(storage="disk", disk_dir="/tmp"),
                      **self._common(eq))
        with pytest.raises(NotImplementedError, match="storage='cpu'"):
            PropTorch(eq, backend="torch", impl="c", memory=Ckpt(mode="chunk", storage="cpu"), **self._common(eq))
        # the legacy dict spelling goes through the same refusal
        with pytest.raises(NotImplementedError, match="storage='cpu'"):
            PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                      boundary_saving_config={"enabled": True, "storage": "cpu"}, **self._common(eq))

    def test_auto_takes_the_host_core_explicit_c_runs(self, monkeypatch):
        """'auto' runs the host core an explicit impl='c' runs on -- the one
        the install ships or this module built: the same run, bit for bit, and
        no warning.  Eager options or host staging keep it on eager; so does
        having no core at hand and no compiler to build one, quietly."""
        from sweep.propagator import torch as ptorch
        eq = gc.make_equation(ACOUSTIC, CPU, CFG)
        explicit = PropTorch(eq, backend="torch", impl="c", **self._common(eq))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            auto = PropTorch(eq, backend="torch", **self._common(eq))
        assert auto.impl == "c"
        rec_a, grads_a = _run(auto, ACOUSTIC, CPU)
        rec_c, grads_c = _run(explicit, ACOUSTIC, CPU)
        assert torch.equal(rec_a, rec_c)
        assert all(torch.equal(grads_a[k], grads_c[k]) for k in grads_c)
        for passed in (dict(eager_options=EagerOptions(use_compile=False)), dict(use_compile=False),
                       dict(memory=BoundarySaving(storage="cpu"))):
            assert PropTorch(eq, backend="torch", **passed, **self._common(eq)).impl == "eager", passed
        monkeypatch.setattr(ptorch, "_HOST_CORE", None)
        monkeypatch.setattr(hostcore, "available", lambda: False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert PropTorch(eq, backend="torch", **self._common(eq)).impl == "eager"

    def test_auto_builds_the_core_when_none_is_at_hand(self, monkeypatch, capfd):
        """No shipped core and no current cache: the first 'auto' construction
        on a CPU device builds the core (the drawer this module built, so ninja
        has nothing to do) and takes it -- announced on stderr, the compiler's
        output kept off the terminal."""
        from sweep.propagator import torch as ptorch
        if hostcore.cached_local_core() is None:
            pytest.skip("no current cached host core (a shipped one serves): this would build one")
        monkeypatch.delenv("SWEEP_CORE_CPU", raising=False)
        monkeypatch.setattr(hostcore, "shipped_core", lambda: (None, "none shipped (test)"))
        monkeypatch.setattr(hostcore, "cached_local_core", lambda: None)
        monkeypatch.setattr(hostcore, "_core_so", None)
        monkeypatch.setattr(ptorch, "_HOST_CORE", None)
        builds = []
        real_build = hostcore.build_core
        monkeypatch.setattr(hostcore, "build_core", lambda verbose=True: builds.append(verbose) or real_build(verbose))
        eq = gc.make_equation(ACOUSTIC, CPU, CFG)
        capfd.readouterr()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert PropTorch(eq, backend="torch", **self._common(eq)).impl == "c"
            assert PropTorch(eq, backend="torch", **self._common(eq)).impl == "c"
        assert builds == [False], "built once, its output kept off the terminal"
        err = capfd.readouterr().err
        assert "building the host (CPU) core" in err and "host core built and cached" in err
        assert "ninja" not in err

    def test_unported_equation_falls_back_with_a_warning(self):
        from sweep.equations import ViscoAcoustic
        eq = ViscoAcoustic(spatial_order=CFG.spatial_order, device=CPU, backend="torch")
        with pytest.warns(UserWarning, match="not built into the host \\(CPU\\) core"):
            solver = PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                               boundary_saving_config={"enabled": False},
                               shape=ACOUSTIC.shape, abcn=CFG.abcn, dh=10.0, dt=CFG.dt, dev=CPU,
                               pml_type="cpmlr", nt=CFG.nt, B=1)
        assert solver.impl == "eager"

    def test_entries_route_models_to_the_core_of_their_device(self):
        """Every entry picks its core per call (``entries._core_for``): a CPU
        model lands on the host core (stream 0), a CUDA model on the CUDA core
        (torch's current stream) -- real cores, not fakes.  (A deliberately
        incomplete input is not the way to show routing: an input with no
        wavelet trips a ``Buf`` assert in the runner constructor and aborts the
        process on BOTH cores before any SWEEP_CHECK speaks.)"""
        from sweep.backend.c import entries, loader
        lib, stream = entries._core_for([torch.zeros(1, 1, 8, 8)], "acoustic2d_forward")
        assert lib is loader.core_lib("cpu")
        assert lib.sweep_core_device() == b"cpu"
        assert stream == 0
        if _cuda_core_ready():
            lib, stream = entries._core_for([torch.zeros(1, 1, 8, 8, device="cuda")], "acoustic2d_forward")
            assert lib is loader.core_lib("cuda")
            assert lib.sweep_core_device() == b"cuda"
            assert stream == torch.cuda.current_stream().cuda_stream


# --------------------------------------------------------------------------- #
# 5-7. properties of a whole process: SWEEP_HOST_ROWS and OMP_NUM_THREADS are
# read once per process, hence subprocesses.  An interpreter costs ~3 s of
# imports and a case 0.1-2 s, so each setting gets ONE process that runs every
# case of its test.
# --------------------------------------------------------------------------- #
_HERE = str(Path(__file__).resolve().parent)


def _run_cases(worker, out, cases, **env):
    """Run ``worker`` in a fresh interpreter, with ``env`` on top of this
    process's environment; it runs every case and saves {case: tensors}."""
    import json
    import subprocess
    p = subprocess.run([sys.executable, "-c", worker, str(out), json.dumps(cases), _HERE],
                       env=dict(os.environ, **env), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    res = torch.load(out)
    assert len(res) == len(cases), sorted(res)
    return res


# 5. the row launcher: interior cells as SIMD loops (launch/cell_path.h)
_ROW_WORKER = r"""
import json
import sys
import numpy as np
import torch
from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving, Full
from sweep.propagator.torch import PropTorch


def run(mode, topo, order):
    nz, nx, nt, B = 60, 72, 150, 2
    kw = {}
    if topo == "hill":
        x = np.arange(nx)
        kw = dict(topography=(3.0 + 6.0 * np.exp(-((x - nx / 2) ** 2) / (2 * (nx / 8) ** 2))).round().astype(np.int64),
                  topo_method="image")
    eq = Acoustic(spatial_order=order, device="cpu", backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=(nz, nx), abcn=12, dh=10.0, dt=1e-3, dev="cpu",
                     pml_type="cpmlr", nt=nt, B=B, memory=Full() if mode == "full" else BoundarySaving(), **kw)
    assert prop.impl == "c"
    t = np.arange(nt) * 1e-3 - 0.04
    wav = torch.tensor((1 - 2 * (np.pi * 20 * t) ** 2) * np.exp(-(np.pi * 20 * t) ** 2), dtype=torch.float32)
    src = np.array([[nx // 3, nz // 3], [2 * nx // 3, nz // 2]], dtype=np.int64)   # (nshots, 2)
    rec = np.array([[[x, 14] for x in range(4, nx - 4, 3)]] * B, dtype=np.int64)
    z = torch.arange(nz, dtype=torch.float32)[:, None]
    vp = (2000.0 + 10.0 * z).expand(nz, nx).clone()
    vp[nz // 2:, nx // 3:] += 250.0
    vp.requires_grad_(True)
    d = prop(wav, src, rec, models=[vp])
    w = torch.randn(d.shape, generator=torch.Generator().manual_seed(3))
    (d * w).sum().backward()
    return {"record": d.detach(), "grad": vp.grad.detach()}


out, cases = sys.argv[1], json.loads(sys.argv[2])
torch.save({"-".join(map(str, c)): run(*c) for c in cases}, out)
"""

_ROW_CASES = [("full", "none", 8), ("bs", "none", 8), ("bs", "none", 2), ("full", "hill", 8),
              ("full", "hill", 4)]   # boundary saving + topography is refused on impl='c'


class TestRowLauncher:
    """acoustic2d's three stencil kernels run on the host core through
    ``host/launch.h launch_rows``: per row, the cells in the kernel's interior
    box go through the shared cell body (CellPath::Interior / InteriorOut) in
    one ``omp simd`` loop, every other cell through the kernel itself.  That
    must change nothing: with ``-ffp-contract=off`` each SIMD lane rounds what
    the scalar cell rounded, so the result is bit-identical to the
    virtual-grid launcher (``SWEEP_HOST_ROWS=0``, read once per process --
    hence one subprocess per launcher).  Two shots (the batch rows), PML on
    every face, and an irregular surface whose air cells sit inside the
    interior box (forward and adjoint kernels; boundary saving covers the
    reconstruction kernel)."""

    def test_rows_are_bit_identical_to_the_virtual_grid(self, tmp_path):
        rows = _run_cases(_ROW_WORKER, tmp_path / "rows.pt", _ROW_CASES, SWEEP_HOST_ROWS="1", OMP_NUM_THREADS="4")
        grid = _run_cases(_ROW_WORKER, tmp_path / "grid.pt", _ROW_CASES, SWEEP_HOST_ROWS="0", OMP_NUM_THREADS="4")
        for case, r in rows.items():
            assert torch.count_nonzero(r["grad"]) > 0, case
            assert torch.equal(r["record"], grid[case]["record"]), case
            assert torch.equal(r["grad"], grid[case]["grad"]), case


# 6. checkpoint replay: a property of the code, not of the thread count
_GRAD_WORKER = r"""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, sys.argv[3])
import solver_gradient_mode_suite as S


def run(solver, mode, *argv):
    torch.manual_seed(0)
    np.random.seed(0)
    spec, scenario = S.SOLVERS[solver], S.SCENARIOS["interior"]
    args = S.build_parser().parse_args(list(argv))
    args.no_plot = True
    shape = S.shape_for(spec, args)
    sources, receivers = S.make_geometry(spec, shape, scenario, args)
    _, init_models, grad_flags = S.make_models(spec, shape)
    wavelet = torch.tensor(S.ricker(args.nt, args.dt, args.freq, args.delay))
    prop = S.build_solver(spec, "c", mode, scenario, shape, "cpu", args,
                          Path(tempfile.mkdtemp(prefix="host_grad_")), f"{solver}_{mode}")
    assert prop.impl == "c"
    fwd = S.run_forward(prop, wavelet, sources, receivers, init_models, "cpu")
    observed = torch.randn(*fwd.shape, generator=torch.Generator().manual_seed(1234)) * 1e-3
    grad = S.run_gradient(prop, wavelet, sources, receivers, observed, init_models, grad_flags,
                          spec.model_names, "cpu")
    return {k: v.detach().cpu() for k, v in grad["grads"].items()}


out, cases = sys.argv[1], json.loads(sys.argv[2])
torch.save({" ".join(c): run(*c) for c in cases}, out)
"""


class TestCheckpointReplay:
    """The checkpoint backward re-runs the forward kernel from snapshots.  That
    kernel reads the CPML memory psi at neighbouring cells, so the replay state
    carries the forward's psi double-buffer (rotated by swap_pml): bound
    without it, each cell updated psi in place while its neighbours read it --
    a race the GPU's block order happened to resolve the forward's way and the
    host's OpenMP row chunks did not (the gradient moved with
    OMP_NUM_THREADS, ~1e-6).  Pinned: one thread or four, chunk or recursive,
    the checkpoint gradient is the full-mode gradient, bit for bit."""

    SOLVERS = ("acoustic2d", "vrz2d", "born2d")

    def test_ckpt_gradient_is_thread_count_independent_and_equals_full(self, tmp_path):
        ckpt = [(s, m) for s in self.SOLVERS for m in ("ckpt_chunk", "ckpt_recursive")]
        t4 = _run_cases(_GRAD_WORKER, tmp_path / "t4.pt", [(s, "full") for s in self.SOLVERS] + ckpt,
                        OMP_NUM_THREADS="4")
        t1 = _run_cases(_GRAD_WORKER, tmp_path / "t1.pt", ckpt, OMP_NUM_THREADS="1")
        for solver, mode in ckpt:
            full, key = t4[f"{solver} full"], f"{solver} {mode}"
            for name in full:
                assert torch.count_nonzero(full[name]) > 0, f"{solver} {name}"
                assert torch.equal(t1[key][name], t4[key][name]), f"{key} {name}: 1 vs 4 threads"
                assert torch.equal(t4[key][name], full[name]), f"{key} {name}: vs full"


# --------------------------------------------------------------------------- #
# 7. 3-D lossy boundary storage: one restored value per strip cell
# --------------------------------------------------------------------------- #
class TestLossyBoundary3DRestore:
    """INT8 / scaled-FP16 boundary storage quantizes every strip face with its
    own block scales, so a 3-D strip cell in two or three bands (an edge or
    corner, or on a thin axis a cell both opposite bands cover) is stored as
    two or three slightly different values.  The compact restore kernel visits
    the cell once per band; each visit wrote it, the last thread to run won,
    and the gradient moved with OMP_NUM_THREADS (acoustic3d: int8 ~2e-5, fp16
    ~5e-8 relative; elastic3d int8 up to ~2e-3).  The restore now writes such a
    cell from the band the full-grid scan kernel writes last: one value at any
    thread count -- the scan kernel's (SWEEP_BOUNDARY_NO_COMPACT=1).
    acoustic_vti_1st_3d stands in for the multi-field restore
    (restore_backward_3d_field, shared with elastic3d, das_mu3d and
    elastic_tti_sg3d), where int8 and fp16 take one branch.

    The gate's grid with a 10-cell absorbing pad instead of 30: the bands sit
    on the faces of the physical box (boundary/strip.cuh), so their cells and
    blocks are the same, on a grid of 7x fewer cells."""

    CASES = [("acoustic3d", "bs_gpu_int8"), ("acoustic3d", "bs_gpu_fp16"), ("acoustic_vti_1st_3d", "bs_gpu_int8")]

    def test_restore_is_thread_count_independent(self, tmp_path):
        cases = [c + ("--abcn", "10") for c in self.CASES]
        ref = _run_cases(_GRAD_WORKER, tmp_path / "t1.pt", cases, OMP_NUM_THREADS="1")
        runs = {"t4": _run_cases(_GRAD_WORKER, tmp_path / "t4.pt", cases, OMP_NUM_THREADS="4"),
                "scan_t4": _run_cases(_GRAD_WORKER, tmp_path / "scan_t4.pt", cases, OMP_NUM_THREADS="4",
                                      SWEEP_BOUNDARY_NO_COMPACT="1")}
        for case, g_ref in ref.items():
            for name in g_ref:
                assert torch.count_nonzero(g_ref[name]) > 0, f"{case} {name}"
                for tag, g in runs.items():
                    assert torch.equal(g[case][name], g_ref[name]), f"{case} {name}: {tag} vs 1 thread"
