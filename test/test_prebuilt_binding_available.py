"""``is_torch_binding_available()`` must not require a CUDA toolkit when the
compiled ``sweep._C`` is already built.

The regression it guards: the probe used to ask ``jit.can_build()`` — "could I
JIT-compile this?" — which needs nvcc. On a machine that installed with
``SWEEP_BUILD_CUDA=1`` (or ran ``setup.py build_ext --inplace``) and then ran
without a CUDA toolkit on PATH, the answer was False even though ``sweep._C``
imported fine, so ``impl='c'`` silently fell back to eager with only a warning
— a ~10-30x slowdown whose cause is invisible from the symptom.
"""

import pytest

import sweep


class TestExtensionOriginClassification:
    """The suffix test that decides "real binary" vs "JIT shim"."""

    @pytest.mark.parametrize("origin", [
        "/x/sweep/_C.cpython-312-x86_64-linux-gnu.so",
        "/x/sweep/_C.abi3.so",
        "/x/sweep/_C.cp312-win_amd64.pyd",
    ])
    def test_compiled_extensions_are_recognised(self, origin):
        # Only assert on suffixes this interpreter actually knows about;
        # EXTENSION_SUFFIXES is platform-specific.
        from importlib.machinery import EXTENSION_SUFFIXES
        if not origin.endswith(tuple(EXTENSION_SUFFIXES)):
            pytest.skip(f"{origin} is not an extension suffix on this platform")
        assert sweep._is_extension_origin(origin) is True

    @pytest.mark.parametrize("origin", [
        "/x/sweep/_C.py",            # the lazy JIT shim
        "/x/sweep/_C/__init__.py",
        "",
        None,
    ])
    def test_source_and_missing_origins_are_not_extensions(self, origin):
        assert sweep._is_extension_origin(origin) is False


@pytest.fixture
def torch_present(monkeypatch):
    """The probe returns False outright when torch is missing. These tests are
    about the decision AFTER that gate, so pin it — otherwise they quietly pass
    for the wrong reason on a torch-less interpreter."""
    real_find_spec = sweep.find_spec
    monkeypatch.setattr(
        sweep, "find_spec",
        lambda name: object() if name == "torch" else real_find_spec(name))


class TestAvailabilityProbe:
    def test_prebuilt_binding_beats_a_missing_toolkit(self, monkeypatch, torch_present):
        """This is the bug: built kernels + no nvcc must still report available."""
        from sweep.backend.c import jit
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        monkeypatch.setattr(
            jit, "can_build",
            lambda: (False, "no suitable CUDA toolkit found (need nvcc >=12.4 ...)"))

        assert sweep.is_torch_binding_available() is True

    def test_without_a_prebuilt_binding_the_toolkit_decides(self, monkeypatch, torch_present):
        """The JIT path is unchanged: no binary on disk means nvcc is required."""
        from sweep.backend.c import jit
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)

        monkeypatch.setattr(jit, "can_build", lambda: (False, "no nvcc"))
        assert sweep.is_torch_binding_available() is False

        monkeypatch.setattr(jit, "can_build", lambda: (True, "ok"))
        assert sweep.is_torch_binding_available() is True

    def test_probe_never_triggers_the_jit_compile(self, monkeypatch, torch_present):
        """The whole point of the probe is to answer without a surprise ~3 min
        build, so it must not reach ``jit.load()`` (the compiled shim) nor
        ``core_path()`` and the local core build behind it, on either path."""
        from sweep.backend.c import jit

        def explode(*_a, **_k):
            raise AssertionError("is_torch_binding_available() triggered a compile")

        for entry in ("load", "core_path", "_local_core", "_build_core", "_stage"):
            monkeypatch.setattr(jit, entry, explode)
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        sweep.is_torch_binding_available()
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        sweep.is_torch_binding_available()

    def test_spec_lookup_is_resilient(self, monkeypatch):
        """A broken importer must degrade to 'ask the toolkit', not raise out of
        a predicate every propagator construction calls."""
        def boom(_name):
            raise ImportError("meta path finder is unhappy")

        monkeypatch.setattr(sweep, "find_spec", boom)
        assert sweep._prebuilt_binding_present() is False


def test_missing_torch_short_circuits(monkeypatch):
    """No torch, no compiled path — regardless of what is on disk."""
    monkeypatch.setattr(sweep, "find_spec", lambda name: None)
    assert sweep.is_torch_binding_available() is False


class TestBindingDiagnostics:
    """``sweep.backend.torch.binding`` asked the same wrong question in three
    more places, and its answers are what a confused user reads first."""

    def test_is_available_follows_the_one_probe(self, monkeypatch):
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "is_torch_binding_available", lambda: True)
        assert binding.is_available() is True
        monkeypatch.setattr(sweep, "is_torch_binding_available", lambda: False)
        assert binding.is_available() is False

    def test_prebuilt_counts_as_compiled(self, monkeypatch):
        """An ahead-of-time extension IS the built backend; reporting 'not
        compiled' because the JIT cache is empty sends people to rebuild
        something they already have."""
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        assert binding.is_compiled() is True

    def test_diagnostics_explains_usable_without_a_toolkit(self, monkeypatch):
        from sweep.backend.c import jit
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        monkeypatch.setattr(jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(jit, "_find_cuda_home", lambda: None)

        d = binding.diagnostics()
        assert d["usable"] is True
        assert d["prebuilt"] is True
        assert d["already_compiled"] is True
        # usable with no cuda_home is exactly the pair that needs explaining
        assert d["cuda_home"] is None
        assert "pre-built" in d["reason"]

    def test_diagnostics_unchanged_on_the_jit_path(self, monkeypatch):
        from sweep.backend.c import jit
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        monkeypatch.setattr(jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(jit, "_find_cuda_home", lambda: None)

        d = binding.diagnostics()
        assert d["usable"] is False
        assert d["prebuilt"] is False
        assert d["reason"] == "no nvcc"

    def test_shim_names_the_compiled_extension(self, monkeypatch):
        """A ``SWEEP_BUILD_CUDA=1`` install's ``sweep._C`` IS the pybind shim --
        it shadows the ctypes layer -- so ``shim`` must say so. It used to read
        "ctypes" there, because only ``SWEEP_JIT_FULL`` was consulted."""
        from sweep.backend.c import jit
        from sweep.backend.torch import binding

        monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
        monkeypatch.setattr(jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        assert binding.diagnostics()["shim"] == "pybind"
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        assert binding.diagnostics()["shim"] == "ctypes"
        monkeypatch.setenv("SWEEP_JIT_FULL", "1")
        assert binding.diagnostics()["shim"] == "pybind"

    SHIPPED_CORE_KEYS = {"path", "reason", "tag", "available"}

    def test_shipped_core_lists_the_cores_the_install_carries(self, monkeypatch):
        """``shipped_core.available`` -- the ``lib/<tag>/`` drawers holding a
        core -- is a list on the probed path and on the fallback the except
        branch builds when the probe itself blows up, so a reader can always
        test it the same way."""
        from sweep.backend.c import jit
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        monkeypatch.setattr(jit, "_find_cuda_home", lambda: None)
        monkeypatch.setattr(jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(jit, "_shipped_tags", lambda: ["cu12", "cu13"])
        core = binding.diagnostics()["shipped_core"]
        assert set(core) == self.SHIPPED_CORE_KEYS
        assert core["available"] == ["cu12", "cu13"]

        def boom():
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(jit, "can_build", boom)
        core = binding.diagnostics()["shipped_core"]
        assert set(core) == self.SHIPPED_CORE_KEYS
        assert core == {"path": None, "reason": "not probed", "tag": "", "available": []}
