"""``is_torch_binding_available()`` must not require a CUDA toolkit when the
compiled ``sweep._C`` is already built.

The regression it guards: the probe used to ask ``_jit.can_build()`` — "could I
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
        from sweep import _jit
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        monkeypatch.setattr(
            _jit, "can_build",
            lambda: (False, "no suitable CUDA toolkit found (need nvcc >=12.4 ...)"))

        assert sweep.is_torch_binding_available() is True

    def test_without_a_prebuilt_binding_the_toolkit_decides(self, monkeypatch, torch_present):
        """The JIT path is unchanged: no binary on disk means nvcc is required."""
        from sweep import _jit
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)

        monkeypatch.setattr(_jit, "can_build", lambda: (False, "no nvcc"))
        assert sweep.is_torch_binding_available() is False

        monkeypatch.setattr(_jit, "can_build", lambda: (True, "ok"))
        assert sweep.is_torch_binding_available() is True

    def test_probe_never_triggers_the_jit_compile(self, monkeypatch, torch_present):
        """The whole point of the probe is to answer without a surprise ~3 min
        build, so it must not reach ``_jit.load()`` on either path."""
        from sweep import _jit

        def explode():
            raise AssertionError("is_torch_binding_available() triggered a compile")

        monkeypatch.setattr(_jit, "load", explode)
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
        from sweep import _jit
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: True)
        monkeypatch.setattr(_jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)

        d = binding.diagnostics()
        assert d["usable"] is True
        assert d["prebuilt"] is True
        assert d["already_compiled"] is True
        # usable with no cuda_home is exactly the pair that needs explaining
        assert d["cuda_home"] is None
        assert "pre-built" in d["reason"]

    def test_diagnostics_unchanged_on_the_jit_path(self, monkeypatch):
        from sweep import _jit
        from sweep.backend.torch import binding

        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        monkeypatch.setattr(_jit, "can_build", lambda: (False, "no nvcc"))
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)

        d = binding.diagnostics()
        assert d["usable"] is False
        assert d["prebuilt"] is False
        assert d["reason"] == "no nvcc"
