"""sweep CLI — engine introspection only (`list equations`, `show <Eq>`).

Both solver families answer: the finite-difference equations of
`sweep.equations` and the spectral-element ones of `sweep.sem`.

The task-running / FWI / LSRTM YAML driver is in `sweepx`:
`pip install sweepx` and use `sweep-tasks run task.yaml`.
"""

import argparse
import difflib
import inspect


def _format_yes_no(value):
    return "yes" if value else "no"


def _format_models(models):
    return str(models) if models is not None else "N/A"


def _read_class_metadata(cls, name):
    try:
        descriptor = inspect.getattr_static(cls, name)
    except AttributeError:
        return None

    if isinstance(descriptor, property):
        try:
            return descriptor.fget(cls)
        except Exception:
            return None

    return descriptor


def _iter_equation_rows():
    import sweep
    import sweep.equations as eq

    torch_binding_available = sweep.is_torch_binding_available()
    rows = []

    for name, obj in sorted(eq.equation_classes().items()):
        models = _read_class_metadata(obj, "models")
        binding_supported = eq.supports_torch_binding(obj)
        rows.append(
            {
                "name": name,
                "models": _format_models(models),
                "torch_binding_support": _format_yes_no(binding_supported),
                "torch_binding_available": _format_yes_no(
                    binding_supported and torch_binding_available
                ),
            }
        )

    return rows


def _sem_registry():
    """``{name: class}`` of the spectral-element equations (``sweep.sem``).

    They are not in the FD registry: they do not subclass ``WaveEquation``
    and run with ``PropSEM``, not ``PropTorch``.
    """
    from sweep.sem.equations import equation_classes

    return equation_classes()


def _iter_sem_rows():
    import sweep
    from sweep.sem.equations import summary
    from sweep.sem.prop_base import propsem_models
    from sweep.sem.prop_c import C_SUPPORTED

    torch_binding_available = sweep.is_torch_binding_available()
    rows = []

    for name, cls in sorted(_sem_registry().items()):
        info = summary(cls)
        binding_supported = cls in C_SUPPORTED
        rows.append(
            {
                "name": name,
                "mesh": f"{info['mesh']} {info['ndim']}-D"
                        + (", topography" if info["topography"] else ""),
                "models": _format_models(propsem_models(cls)),
                "torch_binding_support": _format_yes_no(binding_supported),
                "torch_binding_available": _format_yes_no(
                    binding_supported and torch_binding_available
                ),
            }
        )

    return rows


def _print_table(rows, headers):
    widths = {
        key: max([len(header)] + [len(row[key]) for row in rows])
        for header, key in headers
    }
    print(
        "  "
        + "  ".join(header.ljust(widths[key]) for header, key in headers)
    )
    print(
        "  "
        + "  ".join("-" * widths[key] for _, key in headers)
    )

    for row in rows:
        print(
            "  "
            + "  ".join(row[key].ljust(widths[key]) for _, key in headers)
        )


def list_equations():
    print("Available equations:\n")
    print("Finite difference (sweep.equations, run with PropTorch):\n")
    _print_table(_iter_equation_rows(), [
        ("Equation", "name"),
        ("Models", "models"),
        ("Torch Binding", "torch_binding_support"),
        ("Binding Ready", "torch_binding_available"),
    ])
    print("\nSpectral element (sweep.sem, run with PropSEM):\n")
    _print_table(_iter_sem_rows(), [
        ("Equation", "name"),
        ("Mesh", "mesh"),
        ("Models", "models"),
        ("Torch Binding", "torch_binding_support"),
        ("Binding Ready", "torch_binding_available"),
    ])


#: Not equations, but the names users most plausibly type at ``sweep show``:
#: they live in ``sweep.equations`` and dispatch to real equation classes.
_FACADES = ("DAS", "AcousticAniso")


def list_wavefields(class_name):
    """Print one equation's wavefields and models. Returns a process exit code.

    Resolution goes through the registry, not ``getattr(sweep.equations, ...)``.
    The module namespace also holds ``FieldSpec``, ``WaveEquation``, the facades
    and every imported symbol, so the attribute lookup answered for names that
    are not equations at all -- and then described them as facades.
    """
    import sweep
    import sweep.equations as eq

    registry = eq.equation_classes()
    cls = registry.get(class_name)

    if cls is None:
        sem = _sem_registry()
        if class_name in sem:
            return _show_sem_equation(class_name, sem[class_name])
        if class_name in _FACADES:
            print(f"\n=== {class_name} ===")
            print(f"  {class_name} is a facade/dispatcher, not a wave equation:")
            print(f"  it selects a raw equation class from its constructor "
                  f"arguments.")
            print(f"  Use `sweep show <RawClass>` on the one it dispatches to.")
            return 1
        print(f"No such wave equation: {class_name}")
        near = difflib.get_close_matches(class_name, sorted(registry) + sorted(sem),
                                         n=3)
        if near:
            print(f"  Did you mean: {', '.join(near)}?")
        print("  `sweep list equations` names all "
              f"{len(registry)} finite-difference and {len(sem)} spectral-element "
              f"equations.")
        return 1

    wavefields = _read_class_metadata(cls, "wavefields")
    models = _read_class_metadata(cls, "models")

    print(f"\n=== {class_name} ===")
    binding_supported = eq.supports_torch_binding(cls)
    binding_available = binding_supported and sweep.is_torch_binding_available()

    if wavefields is not None:
        print(f"  Wavefields: {wavefields}")
    else:
        print("  Wavefields: N/A")

    if models is not None:
        print(f"  Needed models: {models}")
    else:
        print("  Needed models: N/A")

    print(f"  Torch binding support: {_format_yes_no(binding_supported)}")
    print(f"  Torch binding available: {_format_yes_no(binding_available)}")
    return 0


#: What ``summary()["cpml"]`` means for a caller of PropSEM.
_SEM_CPML = {
    "yes": "yes",
    "split": "yes (an acoustic and an elastic profile)",
    "NO": "no -- a reflecting box, abcn=0 only",
}


def _show_sem_equation(class_name, cls):
    """``sweep show`` for a spectral-element equation (``sweep.sem``)."""
    import sweep
    from sweep.sem.equations import summary
    from sweep.sem.prop_base import propsem_models
    from sweep.sem.prop_c import C_SUPPORTED

    info = summary(cls)
    binding_supported = cls in C_SUPPORTED
    binding_available = binding_supported and sweep.is_torch_binding_available()

    print(f"\n=== {class_name} ===")
    print("  Spectral element: sweep.sem, run with PropSEM")
    print(f"  Mesh: {info['mesh']}, {info['ndim']}-D, "
          f"topography: {_format_yes_no(info['topography'])}")
    print(f"  C-PML: {_SEM_CPML[info['cpml']]}")
    print(f"  Wavefields: {info['fields']}")
    # an equation that names its PropSEM sources (Biot) takes exactly those
    names = list(getattr(cls, "propsem_source_names", None) or ())
    if names:
        print(f"  Source types: {names}")
    else:
        print(f"  Source fields: {info['sources']}")
    print(f"  Receiver types: {info['receivers'] + info['derived']}")
    print(f"  Needed models: {propsem_models(cls)}")
    print(f"  Torch binding support: {_format_yes_no(binding_supported)}")
    print(f"  Torch binding available: {_format_yes_no(binding_available)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='sweep',
        description='SWEEP (Seismic Wave Equation Exploration Platform) — engine introspection. '
                    'For YAML-driven FWI / LSRTM runs, install `sweepx` and use `sweep-tasks run`.',
    )
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List components')
    show_parser = subparsers.add_parser('show', help='Show details for a wave equation class')
    datasets_parser = subparsers.add_parser(
        'datasets', help='Benchmark velocity models (list/info/download/where)')

    list_parser.add_argument('component', choices=['equations'], help='Component type to list')
    show_parser.add_argument('component', help='Wave equation class name to show wavefields for')
    datasets_parser.add_argument('args', nargs=argparse.REMAINDER,
                                 help='subcommand + args (see `sweep datasets -h`)')

    args = parser.parse_args()

    if args.command == 'list' and args.component == 'equations':
        list_equations()
        return 0
    if args.command == 'show':
        return list_wavefields(args.component)
    if args.command == 'datasets':
        from sweep.datasets.cli import main as datasets_main
        return datasets_main(args.args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
