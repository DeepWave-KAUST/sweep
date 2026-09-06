#!/usr/bin/env python3
"""Assert ``sweep.equations``'s public surface is unchanged.

P1 replaces the hand-maintained import wall + ``globals()`` scan with a
decorator registry. The registry is only correct if it reproduces the OLD
surface exactly:

* ``dir(sweep.equations)`` -- every public name, including the author-named and
  ``*Default*`` aliases and the submodule names that ``from . import x`` binds;
* ``_equation_classes()`` -- name -> class, same class object per name (so
  ``ElasticAPM is Elastic`` and ``Elastic3D is not Elastic`` stay true);
* ``torch_binding_supported_equations()`` -- exactly the same 24 names. This one
  is subtle: it is ``callable(cls._C)``, so moving ``_C`` onto the base class
  unconditionally would silently promote every equation into the set.
  ``AcousticCurvilinear`` / ``ElasticCurvilinear`` are IN the set even though
  their ``_C`` only raises, so they must keep an explicit raising ``_C``.

Usage:
    $PY gate/check_equations_api.py gate/golden_equations_api.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    golden_path = Path(sys.argv[1] if len(sys.argv) > 1
                       else Path(__file__).parent / "golden_equations_api.json")
    golden = json.loads(golden_path.read_text())

    import sweep.equations as E

    now_dir = sorted(n for n in dir(E) if not n.startswith("_"))
    now_ec = {k: v.__module__ + "." + v.__qualname__
              for k, v in E.equation_classes().items()} \
        if hasattr(E, "equation_classes") else \
        {k: v.__module__ + "." + v.__qualname__
         for k, v in E._equation_classes().items()}
    now_tb = sorted(E.torch_binding_supported_equations())

    fails = []

    def cmp_set(label, want, got, allow_new=()):
        want_s, got_s = set(want), set(got)
        miss = sorted(want_s - got_s)
        extra = sorted(got_s - want_s - set(allow_new))
        if miss or extra:
            fails.append(f"{label}: missing={miss} unexpected-new={extra}")

    # P1 deliberately ADDS the registry's public API (P1.1/P1.2) and the
    # slot-table submodule (P1.3). Nothing else may appear, and nothing may
    # disappear -- an entry here is a decision that was made, not a leak.
    ALLOWED_NEW = ("register_equation", "get_equation", "equation_classes",
                   "equation_method", "list_equations", "slot_table")
    cmp_set("dir()", golden["dir"], now_dir, allow_new=ALLOWED_NEW)
    cmp_set("equation_classes() names", golden["equation_classes"], now_ec)
    cmp_set("torch_binding_supported_equations()", golden["torch_binding"], now_tb)

    for name in sorted(set(golden["equation_classes"]) & set(now_ec)):
        if golden["equation_classes"][name] != now_ec[name]:
            fails.append(f"equation_classes()[{name!r}]: "
                         f"{golden['equation_classes'][name]} -> {now_ec[name]}")

    # Identity relations that a name->class map alone does not pin down.
    ident = [
        ("ElasticAPM", "Elastic", True),
        ("Elastic3D", "Elastic", False),
        ("AcousticVTILiang", "AcousticVTI", True),
        ("AcousticTTIAlkhalifah", "AcousticTariq", True),
        ("DASElastic", "DASZhao", True),
    ]
    for a, b, same in ident:
        ca, cb = getattr(E, a, None), getattr(E, b, None)
        if ca is None or cb is None:
            fails.append(f"identity check: {a} or {b} missing")
        elif (ca is cb) != same:
            fails.append(f"identity: ({a} is {b}) should be {same}, got {ca is cb}")

    if fails:
        print("EQUATIONS API CHANGED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"equations API unchanged: {len(now_dir)} public names, "
          f"{len(now_ec)} registered equations, {len(now_tb)} with torch binding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
