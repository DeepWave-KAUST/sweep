import os
import inspect
import importlib.util

def find_classes_with_models(module_path, module_name):
    """
    Scans the specified module path for classes that have a 'models' property.
    Returns a dictionary mapping class names to their module names.
    """
    classes = {}

    for fname in os.listdir(module_path):
        if not fname.endswith('.py') or fname == '__init__.py':
            continue

        mod_name = fname[:-3]
        full_mod_name = f"{module_name}.{mod_name}"
        file_path = os.path.join(module_path, fname)

        spec = importlib.util.spec_from_file_location(full_mod_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != full_mod_name:
                continue
            if hasattr(obj, 'models'):
                classes[name] = mod_name

    return classes

def generate_init_py(classes_dict, output_path):
    """
    Generates an __init__.py file that imports all classes from the given dictionary.
    """
    lines = []
    lines.append("# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY\n")

    for cls_name, mod_name in classes_dict.items():
        lines.append(f"from .{mod_name} import {cls_name}")

    lines.append("\n__all__ = [")
    for cls_name in classes_dict.keys():
        lines.append(f"    '{cls_name}',")
    lines.append("]\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_path} with {len(classes_dict)} classes.")

if __name__ == '__main__':
    module_dir = os.path.abspath("/ibex/user/wangs0j/geophyai/src/geophyai/equations")
    module_pkg = "geophyai.equations"

    classes = find_classes_with_models(module_dir, module_pkg)
    output_file = os.path.join(module_dir, '__init__.py')
    generate_init_py(classes, output_file)
