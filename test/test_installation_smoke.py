import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


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

    classes = eq.equation_classes()

    for name in ("Acoustic", "Acoustic3D", "Elastic", "Elastic3D"):
        assert name in classes

    supported = set(eq.torch_binding_supported_equations())
    assert {"Acoustic", "Acoustic3D", "Elastic", "Elastic3D"}.issubset(supported)


def test_installed_cli_lists_equations():
    # pip puts the script in this interpreter's scripts directory, which is not
    # on PATH when a job calls the env's python by its absolute path.
    executable = shutil.which("sweep", path=sysconfig.get_path("scripts")) or shutil.which("sweep")
    assert executable is not None, "Missing 'sweep' CLI. Install the package with pip first."

    # The script imports whichever sweep comes first on its path: make that the
    # tree under test, as conftest does here (a tested wheel has no src/ beside it).
    pythonpath = os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [executable, "list", "equations"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=pythonpath),
    )

    assert result.returncode == 0, result.stderr
    assert "Available equations:" in result.stdout
    assert "Acoustic" in result.stdout
    assert "Torch Binding" in result.stdout
