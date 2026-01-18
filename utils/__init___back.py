import inspect
import pkgutil
import importlib

__all__ = []

package_name = __name__

# Scan all the available equations in the current package
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        continue

    module = importlib.import_module(f".{module_name}", package_name)

    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)

        # A available equation is a class with a 'models' attribute
        if inspect.isclass(obj) and hasattr(obj, 'models'):
            globals()[name] = obj
            __all__.append(name)

del inspect, pkgutil, importlib, name, obj, module, module_name, is_pkg, package_name
