import shutil
import subprocess


def test_import_and_backend_helpers():
    import sweep

    assert hasattr(sweep, "backend")
    assert hasattr(sweep, "equations")
    assert isinstance(sweep.backend.torch.is_available(), bool)
    assert isinstance(sweep.backend.jax.is_available(), bool)
    assert isinstance(sweep.backend.torch.cuda.is_available(), bool)
    assert isinstance(sweep.is_torch_binding_available(), bool)


def test_equation_registry_contains_core_equations():
    import sweep.equations as eq

    classes = eq._equation_classes()

    for name in ("Acoustic", "Acoustic3D", "Elastic", "Elastic3D"):
        assert name in classes

    supported = set(eq.torch_binding_supported_equations())
    assert {"Acoustic", "Acoustic3D", "Elastic", "Elastic3D"}.issubset(supported)


def test_installed_cli_lists_equations():
    executable = shutil.which("sweep")
    assert executable is not None, "Missing 'sweep' CLI. Install the package with pip first."

    result = subprocess.run(
        [executable, "list", "equations"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Available equations:" in result.stdout
    assert "Acoustic" in result.stdout
    assert "Torch Binding" in result.stdout
