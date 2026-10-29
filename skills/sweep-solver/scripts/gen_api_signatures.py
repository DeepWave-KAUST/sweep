#!/usr/bin/env python3
"""Generate references/api-signatures.md from the installed `sweep` package.

Uses `inspect` on the public classes and functions, and instantiates every
registered finite-difference equation on the CPU to read its models, fields and
capability flags. Nothing is compiled and no GPU is needed.

    python gen_api_signatures.py                       # print to stdout
    python gen_api_signatures.py -o ../references/api-signatures.md
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import io
import textwrap
import warnings
from pathlib import Path


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for para in doc.split("\n\n"):
        para = " ".join(para.split())
        if para:
            return para
    return ""


def sig(obj, drop_self=True) -> str:
    try:
        s = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    params = list(s.parameters.values())
    if drop_self and params and params[0].name in ("self", "cls"):
        s = s.replace(parameters=params[1:])
    return str(s)


def block(out, title, text, doc=""):
    out.write(f"### `{title}`\n\n")
    out.write("```text\n" + "\n".join(textwrap.wrap(text, 100, subsequent_indent="    ",
                                                   break_long_words=False,
                                                   break_on_hyphens=False)) + "\n```\n\n")
    if doc:
        out.write(doc + "\n\n")


def section_toplevel(out):
    import sweep
    out.write("## `sweep` (top level)\n\n")
    for name in ("precompile", "is_torch_binding_available"):
        fn = getattr(sweep, name)
        block(out, f"sweep.{name}", f"{name}{sig(fn)}", first_line(fn))
    from sweep.backend.torch import binding
    for name in ("is_available", "is_compiled", "diagnostics"):
        fn = getattr(binding, name)
        block(out, f"sweep.backend.torch.binding.{name}", f"{name}{sig(fn)}", first_line(fn))


def section_proptorch(out):
    from sweep.propagator.base import PropBase
    from sweep.propagator.torch import PropTorch
    out.write("## `sweep.propagator.torch.PropTorch`\n\n")
    out.write(first_line(PropTorch) + "\n\n")
    own = [p for p in inspect.signature(PropTorch.__init__).parameters.values()
           if p.kind is inspect.Parameter.KEYWORD_ONLY]
    base = [p for p in inspect.signature(PropBase.__init__).parameters.values()
            if p.name != "self" and p.kind not in (inspect.Parameter.VAR_KEYWORD,)]
    merged = ", ".join(str(p) for p in base) + ", *, " + ", ".join(str(p) for p in own)
    block(out, "PropTorch(...)", f"PropTorch({merged})",
          "`PropTorch.__init__` is `(*args, backend, impl, backend_options, eager_options, "
          "cuda_options, memory, **kwargs)`; `*args`/`**kwargs` go to the base constructor "
          "shown here (parameters of `PropBase.__init__`). Any other keyword raises `TypeError`.")
    block(out, "PropTorch.forward / __call__", f"forward{sig(PropTorch.forward)}",
          "Returns the record `(B, nt, nrec, len(receiver_type))`. With `impl='eager'` and "
          "`return_wavefield=True`: `(record, snapshots)`. Extra call keywords accepted by "
          "the eager backend: `snapshot_times`, `snapshot_interval`, `snapshot_grid`.")
    from sweep.propagator._torch_eager import _PropTorchEager
    for name in ("register_gradient", "clear_gradients", "imaging"):
        fn = getattr(_PropTorchEager, name)
        block(out, f"PropTorch.{name}  (impl='eager' only)", f"{name}{sig(fn)}", first_line(fn))


def section_options(out):
    import sweep.propagator.options as o
    out.write("## `sweep.propagator.options`\n\n")
    for name in ("Full", "BoundarySaving", "Ckpt", "EagerOptions", "CUDAOptions"):
        cls = getattr(o, name)
        block(out, name, f"{name}{sig(cls, drop_self=False)}", first_line(cls))
    import sweep.propagator as p
    block(out, "sweep.propagator.last_grad_split_iii",
          f"last_grad_split_iii{sig(p.last_grad_split_iii)}", first_line(p.last_grad_split_iii))


def _ast_init_signature(path: Path, cls_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return "(" + ast.unparse(item.args) + ")"
    return "(...)"


def section_propjax(out):
    out.write("## `sweep.propagator.jax.PropJax`\n\n")
    try:
        from sweep.propagator.jax import PropJax
        init, fwd = sig(PropJax.__init__), sig(PropJax.forward)
    except ImportError:
        import sweep.propagator as p
        path = Path(p.__file__).with_name("jax.py")
        init = _ast_init_signature(path, "PropJax").replace("(self, ", "(")
        fwd = "(wavelet, sources, receivers, models=None, return_wavefield=False, adj=False, **kwargs)"
    from sweep.propagator.base import PropBase
    block(out, "PropJax(...)", f"PropJax{init}",
          "`*args`/`**kwargs` are the `PropBase.__init__` parameters listed under PropTorch. "
          "Needs `pip install 'sweep-solver[jax]'` and an equation built with `backend='jax'`.")
    block(out, "PropJax.forward / __call__", f"forward{fwd}")
    del PropBase


def section_sem(out):
    from sweep.sem import PropSEM
    import sweep.sem.equations as sem
    from sweep.sem.prop_base import propsem_models
    out.write("## `sweep.sem`\n\n")
    block(out, "PropSEM(...)", f"PropSEM{sig(PropSEM.__init__)}", first_line(PropSEM))
    block(out, "PropSEM.__call__", f"__call__{sig(PropSEM.__call__)}")
    out.write("| SEM equation class | `models=` |\n|---|---|\n")
    for name, cls in sorted(sem.equation_classes().items()):
        try:
            models = propsem_models(cls)
        except Exception as exc:  # noqa: BLE001
            models = f"? ({type(exc).__name__})"
        out.write(f"| `{name}` | `{models}` |\n")
    out.write("\n")


def section_misc(out):
    out.write("## Signals, datasets, parallel\n\n")
    import sweep.signal as s
    block(out, "sweep.signal.ricker", f"ricker{sig(s.ricker)}", first_line(s.ricker))
    from sweep import datasets
    for name in ("load", "catalog", "info", "available", "register"):
        fn = getattr(datasets, name)
        block(out, f"sweep.datasets.{name}", f"{name}{sig(fn)}", first_line(fn))
    block(out, "sweep.datasets.load_marmousi", f"load_marmousi{sig(datasets.load_marmousi)}",
          first_line(datasets.load_marmousi))
    from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh, unpad_from_mesh
    block(out, "sweep.parallel.ModelParallel", f"ModelParallel{sig(ModelParallel.__init__)}",
          first_line(ModelParallel))
    block(out, "sweep.parallel.MeshTopology", f"MeshTopology{sig(MeshTopology.__init__)}",
          first_line(MeshTopology))
    block(out, "sweep.parallel.pad_to_mesh", f"pad_to_mesh{sig(pad_to_mesh)}", first_line(pad_to_mesh))
    block(out, "sweep.parallel.unpad_from_mesh", f"unpad_from_mesh{sig(unpad_from_mesh)}",
          first_line(unpad_from_mesh))


def yn(v):
    return "yes" if v else "no"


def section_equations(out):
    from sweep.equations import equation_classes
    out.write("## Finite-difference equations (`sweep.equations`)\n\n")
    out.write("Instantiated on the CPU with defaults (`backend='torch'`). `src`/`rec` = default "
              "`source_type`/`receiver_type`; `src ok`/`rec ok` = every field accepted. "
              "`c` = compiled CUDA kernels exist (`impl='c'`). `FS` = free surface accepted; "
              "`per-edge` = per-edge free surface on eager/c. `BS` = boundary saving on c/eager. "
              "`(B,...) models` = per-shot batched models on impl='c'.\n\n")
    rows, sigs, seen = [], [], set()
    for name, cls in sorted(equation_classes().items()):
        alias = "" if cls.__name__ == name else f" (= `{cls.__module__.split('.')[-1]}.{cls.__name__}`)"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq = cls(device="cpu", backend="torch")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"| `{name}`{alias} | instantiation failed: {type(exc).__name__} |" + " |" * 10)
            continue
        src_ok = [f.name for f in eq.available_source_fields()]
        rec_ok = [f.name for f in eq.available_receiver_fields()]
        dim = getattr(eq, "ndim", "?")
        rows.append(
            f"| `{name}`{alias} | {dim} | `{list(eq.models)}` | `{list(eq.default_source_fields)}` | "
            f"`{src_ok}` | `{list(eq.default_receiver_fields)}` | `{rec_ok}` | "
            f"`{eq.default_pml_type}` | {yn(cls.supports_torch_binding())} | "
            f"{yn(cls.supports_free_surface)} | "
            f"{yn(cls.supports_per_edge_free_surface)}/{yn(cls.supports_per_edge_free_surface_c)} | "
            f"{yn(cls.supports_boundary_saving_c)}/{yn(cls.supports_boundary_saving_eager)} | "
            f"{yn(cls.supports_batched_models)} |")
        if id(cls) not in seen:  # one line per class, under its canonical name
            seen.add(id(cls))
            names = [n for n, c in equation_classes().items() if c is cls]
            label = cls.__name__ if cls.__name__ in names else names[0]
            sigs.append((label, sig(cls.__init__)))
    out.write("| name | dim | models (order) | src | src ok | rec | rec ok | pml | c | FS | "
              "per-edge e/c | BS c/e | (B,...) models |\n")
    out.write("|" + "---|" * 13 + "\n")
    out.write("\n".join(rows) + "\n\n")
    out.write("### Constructors\n\n```text\n")
    for name, s in sigs:
        out.write("\n".join(textwrap.wrap(f"{name}{s}", 100, subsequent_indent="    ",
                                         break_long_words=False, break_on_hyphens=False)) + "\n")
    out.write("```\n\n")
    facades = importlib.import_module("sweep.equations")
    for name in ("AcousticAniso", "DAS"):
        obj = getattr(facades, name, None)
        if obj is not None:
            target = obj.__new__ if name == "AcousticAniso" else obj.__init__
            block(out, f"sweep.equations.{name} (facade)", f"{name}{sig(target)}", first_line(obj))


def generate() -> str:
    import sweep
    out = io.StringIO()
    out.write("# API signatures\n\n")
    out.write(f"<!-- generated by scripts/gen_api_signatures.py from sweep {sweep.__version__}; "
              "do not edit by hand -->\n\n")
    out.write("Generated with `inspect` from the installed package. Regenerate after upgrading "
              "sweep: `python scripts/gen_api_signatures.py -o references/api-signatures.md`.\n\n")
    out.write("Contents: top level and binding probes; PropTorch; memory/eager options; PropJax; "
              "PropSEM and SEM equations; signals, datasets, parallel; finite-difference equations "
              "(capability table, constructors).\n\n")
    for fn in (section_toplevel, section_proptorch, section_options, section_propjax,
               section_sem, section_misc, section_equations):
        fn(out)
    return out.getvalue().rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args(argv)
    text = generate()
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output} ({text.count(chr(10))} lines)")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
