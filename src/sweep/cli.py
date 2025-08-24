import argparse
import inspect

def list_equations():
    import sweep.equations as eq
    import inspect

    print("Available equations:\n")

    for name in dir(eq):
        if name.startswith("_"):
            continue
        obj = getattr(eq, name)

        if inspect.isclass(obj):
            models = None
            try:
                instance = obj()
                models_attr = getattr(instance, 'models', None)
                if isinstance(models_attr, property):
                    models = models_attr.__get__(instance)
                else:
                    models = models_attr
            except Exception:
                models = None

            models_str = f"models: {models}" if models is not None else "models: N/A"
            print(f"  - {name:<16} {models_str}")

def list_wavefields(class_name):
    import sweep.equations as eq
    import inspect

    if not hasattr(eq, class_name):
        print(f"No such wave equation: {class_name}")
        return

    cls = getattr(eq, class_name)
    if not inspect.isclass(cls):
        print(f"{class_name} is not a valid wave equation")
        return

    wavefields = None
    models = None
    try:
        instance = cls()
        wavefields = getattr(instance, 'wavefields', None)
        models = getattr(instance, 'models', None)
    except Exception as e:
        print(f"Cannot instantiate {class_name}: {e}")

    print(f"\n=== {class_name} ===")
    if wavefields is not None:
        print(f"  Wavefields: {wavefields}")
    else:
        print("  Wavefields: N/A")

    if models is not None:
        print(f"  Needed models: {models}")
    else:
        print("  Needed models: N/A")


def main():
    parser = argparse.ArgumentParser(prog='sweep', description='SWEEP (Seismic Wave Equation Exploration Platform)')
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List components')
    show_parser = subparsers.add_parser('show', help='List components')

    list_parser.add_argument('component', choices=['equations'], help='Component type to list')
    show_parser.add_argument('component', help='Wave equation class name to show wavefields for')

    args = parser.parse_args()

    if args.command == 'list' and args.component == 'equations':
        list_equations()
    elif args.command == 'show':
        list_wavefields(args.component)
    else:
        parser.print_help()

