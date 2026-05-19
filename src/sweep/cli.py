"""sweep CLI — engine introspection only (`list equations`, `show <Eq>`).

Task-running / FWI / LSRTM YAML driver lives in the `sweep-tasks` companion
package: `pip install sweep-tasks` and use `sweep-tasks run task.yaml`.
"""

import argparse
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

    for name, obj in sorted(eq._equation_classes().items()):
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


def list_equations():
    rows = _iter_equation_rows()
    headers = [
        ("Equation", "name"),
        ("Models", "models"),
        ("Torch Binding", "torch_binding_support"),
        ("Binding Ready", "torch_binding_available"),
    ]
    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for header, key in headers
    }

    print("Available equations:\n")
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


def list_wavefields(class_name):
    import sweep
    import sweep.equations as eq

    if not hasattr(eq, class_name):
        print(f"No such wave equation: {class_name}")
        return

    cls = getattr(eq, class_name)
    if not inspect.isclass(cls):
        print(f"{class_name} is not a valid wave equation")
        return

    # The `DAS` and `AcousticAniso` facades live in `sweep.equations` but are
    # not `WaveEquation` subclasses — they dispatch to raw equation classes.
    if not issubclass(cls, eq.WaveEquation):
        print(f"\n=== {class_name} ===")
        print(f"  {class_name} is a facade/dispatcher, not a raw wave equation.")
        print(f"  Use `sweep show <RawClass>` to inspect the underlying equation.")
        return

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


def main():
    parser = argparse.ArgumentParser(
        prog='sweep',
        description='SWEEP (Seismic Wave Equation Exploration Platform) — engine introspection. '
                    'For YAML-driven FWI / LSRTM runs, install `sweep-tasks` and use `sweep-tasks run`.',
    )
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List components')
    show_parser = subparsers.add_parser('show', help='Show details for a wave equation class')

    list_parser.add_argument('component', choices=['equations'], help='Component type to list')
    show_parser.add_argument('component', help='Wave equation class name to show wavefields for')

    args = parser.parse_args()

    if args.command == 'list' and args.component == 'equations':
        list_equations()
        return 0
    if args.command == 'show':
        list_wavefields(args.component)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
