import argparse
import inspect

def list_equations():
    import geophyai.equations as eq
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
                models = getattr(instance, 'models', None)
            except Exception:
                try:
                    attr = getattr(obj, 'models', None)
                    if isinstance(attr, property):
                        models = attr.__get__(None, obj)
                    else:
                        models = attr
                except Exception:
                    models = None

            models_str = f"models: {models}" if models is not None else "models: N/A"
            print(f"  - {name:<16} {models_str}")


def main():
    parser = argparse.ArgumentParser(prog='geophyai', description='GeophyAI CLI')
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List components')
    list_parser.add_argument('component', choices=['equations'], help='Component type to list')

    args = parser.parse_args()

    if args.command == 'list' and args.component == 'equations':
        list_equations()
    else:
        parser.print_help()

