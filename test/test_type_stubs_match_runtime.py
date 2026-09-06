"""A `.pyi` that disagrees with its module is worse than no `.pyi`.

Two things had gone wrong at once:

* ``options.pyi`` shadows ``options.py`` for a type checker and declared none of
  ``Full`` / ``BoundarySaving`` / ``Ckpt`` / ``as_memory_strategy`` -- so the
  import every current doc example uses was an error under the stub, while the
  deprecated spellings it did declare type-checked fine;
* the package had no ``py.typed`` and shipped no ``.pyi`` in its wheel, so none
  of the four stubs was ever consulted by anybody.

These tests compare each stub to the module it stands for, by name and by
signature, so a stub cannot silently fall behind again.
"""
import ast
import importlib
import inspect
import pathlib

import pytest

import sweep

PKG = pathlib.Path(sweep.__file__).resolve().parent
STUBS = sorted(PKG.rglob("*.pyi"))
MODULES = {p: "sweep." + str(p.relative_to(PKG).with_suffix("")).replace("/", ".")
           for p in STUBS}


def _declared(tree):
    """Top-level names a stub declares."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _import(path):
    try:
        return importlib.import_module(MODULES[path])
    except Exception as exc:            # optional backend (jax) not installed
        pytest.skip(f"{MODULES[path]} not importable here: {type(exc).__name__}")


def test_there_are_stubs_to_check():
    assert STUBS, "no .pyi found -- this test would be vacuously green"


def test_the_package_is_pep561_typed():
    """Without this marker a checker ignores the package and every stub with it."""
    assert (PKG / "py.typed").exists(), (
        "sweep ships .pyi files but no py.typed, so PEP 561 makes them invisible")


@pytest.mark.parametrize("path", STUBS, ids=lambda p: p.name)
def test_stub_declares_no_name_the_module_lacks(path):
    module = _import(path)
    tree = ast.parse(path.read_text())
    invented = {n for n in _declared(tree)
                if not n.startswith("_") and not hasattr(module, n)}
    assert not invented, f"{path.name} declares names {module.__name__} does not have: {invented}"


@pytest.mark.parametrize("path", STUBS, ids=lambda p: p.name)
def test_stub_covers_the_public_memory_api(path):
    """The names the docs tell users to import must survive the stub."""
    module = _import(path)
    if module.__name__ != "sweep.propagator.options":
        pytest.skip("only options.pyi shadows the memory-strategy exports")
    declared = _declared(ast.parse(path.read_text()))
    required = {"Full", "BoundarySaving", "Ckpt", "MemoryStrategy",
                "as_memory_strategy", "check_memory_supported"}
    assert required <= declared, f"options.pyi hides {sorted(required - declared)}"


def _stub_keywords(tree, cls, func):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == func:
                    a = sub.args
                    return ({p.arg for p in a.kwonlyargs} |
                            {p.arg for p in a.args if p.arg != "self"},
                            a.kwarg is not None)
    return None, False


@pytest.mark.parametrize(
    "stub, cls, func",
    [("propagator/torch.pyi", "PropTorch", "__init__"),
     ("propagator/torch.pyi", "PropTorch", "forward"),
     ("propagator/_c.pyi", "_CompiledPropagator", "forward")],
)
def test_stub_signature_matches_the_runtime(stub, cls, func):
    path = PKG / stub
    module = _import(path)
    tree = ast.parse(path.read_text())
    stubbed, has_kwargs = _stub_keywords(tree, cls, func)
    assert stubbed is not None, f"{stub} does not declare {cls}.{func}"
    runtime = {n for n, p in inspect.signature(getattr(module, cls), ).parameters.items()
               if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} if func == "__init__" else \
              {n for n, p in inspect.signature(getattr(getattr(module, cls), func)).parameters.items()
               if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n != "self"}
    missing = runtime - stubbed
    assert not missing, (
        f"{stub} hides parameters of {cls}.{func} that the runtime accepts: "
        f"{sorted(missing)}")
    assert has_kwargs or not (stubbed - runtime)
