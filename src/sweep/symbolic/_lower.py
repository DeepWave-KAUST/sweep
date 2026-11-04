"""Lower a system of PDEs to the update of each field.

A second-order field ``f`` with ``f_tt = RHS`` steps as

    f_next = 2 f - f_prev + dt**2 * RHS

and RHS must be linear in the fields, each term a model-only coefficient times
either a second derivative along one axis or a field itself. The terms are
gathered per field into groups of axes that share a coefficient, so that
``vp**2 * laplace(p)`` and ``vp**2 * (p.dz2 + p.dx2)`` lower to the same
``vp**2 * (d2z + d2x)``. Anything else is refused here, before code is written.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.core.symbol import Str

from ._expr import AXES, D1, D2, Dt, Field, Lap, Model


def axes_of(ndim):
    if ndim == 2:
        return ("z", "x")
    if ndim == 3:
        return AXES
    raise ValueError(f"ndim must be 2 or 3, got {ndim!r}")


@dataclass(frozen=True)
class Group:
    """``coef * (d2 field / d axes[0]^2 + ...)``, axes in z, y, x order."""
    coef: sympy.Expr
    field: Field
    axes: tuple


@dataclass(frozen=True)
class Update:
    """``field_next = 2 field - field_prev + dt**2 * (groups + local)``."""
    field: Field
    groups: tuple
    local: tuple          # ((coef, field), ...)


@dataclass(frozen=True)
class Lowered:
    ndim: int
    fields: tuple         # updated fields, declaration order
    models: tuple         # model order of the propagator's ``models=``
    updates: tuple        # one per field, same order
    d2: tuple             # ((field, axis), ...) second derivatives taken


def _solve_for_dt2(eq):
    expr = eq.lhs - eq.rhs
    dts = expr.atoms(Dt)
    if not dts:
        raise ValueError(f"{eq}: no time derivative; each equation advances one field, e.g. Eq(p.dt2, ...)")
    first = [d for d in dts if d.order != 2]
    if first:
        raise NotImplementedError(
            f"{eq}: {first[0]} -- only second time derivatives are supported so far "
            "(no first-order systems, no damping terms in f.dt)")
    if len(dts) > 1:
        raise ValueError(f"{eq}: more than one time derivative ({', '.join(map(str, dts))})")
    dt = dts.pop()
    field = dt.field
    if not isinstance(field, Field):
        raise TypeError(f"{eq}: {dt} is not the time derivative of a Field")
    T = sympy.Dummy("T")
    e = expr.xreplace({dt: T})
    a = sympy.diff(e, T)
    if a.has(T) or a == 0:
        raise ValueError(f"{eq}: not linear in {dt}")
    if _fields_and_ops(a):
        raise ValueError(f"{eq}: the factor {a} of {dt} must not contain fields or derivatives")
    rhs = -e.xreplace({T: 0})
    return field, (rhs if a == 1 else rhs / a)


def _fields_and_ops(expr):
    return expr.atoms(D1, D2, Dt, Lap) | {s for s in expr.free_symbols if isinstance(s, Field)}


def _decompose(eq, rhs, ndim):
    """``{(field, axis): coef}`` of the second derivatives and ``{field: coef}``
    of the bare fields, with RHS == their sum."""
    axes = axes_of(ndim)
    for op in rhs.atoms(Dt):
        raise NotImplementedError(f"{eq}: {op} on the right-hand side is not supported")
    for op in rhs.atoms(D1):
        raise NotImplementedError(
            f"{eq}: first spatial derivative {op} -- only second derivatives "
            "(f.dz2, f.dx2, laplace(f)) are supported so far")
    laps = {}
    for lap in rhs.atoms(Lap):
        if not isinstance(lap.field, Field):
            raise TypeError(f"{eq}: {lap} is not the Laplacian of a Field")
        laps[lap] = sympy.Add(*[D2(lap.field, Str(ax)) for ax in axes])
    rhs = rhs.xreplace(laps)
    d2s = sorted(rhs.atoms(D2), key=sympy.default_sort_key)
    for op in d2s:
        if not isinstance(op.field, Field):
            raise TypeError(f"{eq}: {op} is not a derivative of a Field")
        if op.axis not in axes:
            raise ValueError(f"{eq}: {op} is a {op.axis}-derivative in a {ndim}-D build")
    holes = {op: sympy.Dummy(f"D2_{op.field}_{op.axis}") for op in d2s}
    e = rhs.xreplace(holes)
    fields = sorted((s for s in e.free_symbols if isinstance(s, Field)), key=sympy.default_sort_key)
    holes.update({f: sympy.Dummy(f"F_{f}") for f in fields})
    e = e.xreplace({f: holes[f] for f in fields})

    slots = set(holes.values())
    coefs = {}
    for atom, s in holes.items():
        c = sympy.diff(e, s)
        if c.free_symbols & slots:
            raise ValueError(f"{eq}: not linear in the fields ({atom} is multiplied by a field)")
        coefs[atom] = c
    rest = e - sympy.Add(*[c * holes[a] for a, c in coefs.items()])
    if sympy.expand(rest) != 0 and sympy.simplify(rest) != 0:
        raise ValueError(
            f"{eq}: the term {sympy.simplify(rest)} contains no field; a source "
            "term is not part of the PDE (sources are injected by the propagator)")
    for atom, c in coefs.items():
        unknown = [s for s in c.free_symbols if not isinstance(s, Model)]
        if unknown:
            raise ValueError(f"{eq}: {', '.join(map(str, unknown))} in the coefficient of {atom} "
                             "is neither a Model nor a number")

    d2 = {}
    local = {}
    for atom, c in coefs.items():
        if isinstance(atom, D2):
            d2[(atom.field, atom.axis)] = c
        else:
            local[atom] = c
    return d2, local


def lower(equations, ndim, models=None):
    axes = axes_of(ndim)
    updates = {}
    for eq in equations:
        field, rhs = _solve_for_dt2(eq)
        if field in updates:
            raise ValueError(f"field {field} has more than one equation")
        updates[field] = (eq, rhs)
    fields = tuple(sorted(updates, key=lambda f: f._order))

    out, used_models, d2_taken = [], set(), []
    for field in fields:
        eq, rhs = updates[field]
        d2, local = _decompose(eq, rhs, ndim)
        for (f, _), c in d2.items():
            if f not in updates:
                raise ValueError(f"{eq}: field {f} has no equation of its own")
        for f in local:
            if f not in updates:
                raise ValueError(f"{eq}: field {f} has no equation of its own")
        for c in (*d2.values(), *local.values()):
            used_models |= c.free_symbols
        groups = []
        for f in fields:
            by_coef = {}
            for ax in axes:
                c = d2.get((f, ax))
                if c is None or c == 0:
                    continue
                by_coef.setdefault(c, []).append(ax)
                if (f, ax) not in d2_taken:
                    d2_taken.append((f, ax))
            groups += [Group(c, f, tuple(a)) for c, a in by_coef.items()]
        local_terms = tuple((local[f], f) for f in fields if f in local and local[f] != 0)
        out.append(Update(field, tuple(groups), local_terms))

    if models is None:
        model_order = tuple(sorted(used_models, key=lambda m: m._order))
    else:
        model_order = tuple(models)
        for m in model_order:
            if not isinstance(m, Model):
                raise TypeError(f"models= takes Model symbols, got {m!r}")
        if len(set(model_order)) != len(model_order):
            raise ValueError("models= lists a model twice")
        missing = used_models - set(model_order)
        if missing:
            raise ValueError(f"models= misses {', '.join(sorted(map(str, missing)))}, used by the equations")
        unused = set(model_order) - used_models
        if unused:
            raise ValueError(f"models= lists {', '.join(sorted(map(str, unused)))}, which no equation uses")

    d2_taken.sort(key=lambda fa: (fa[0]._order, axes.index(fa[1])))
    return Lowered(ndim, fields, model_order, tuple(out), tuple(d2_taken))
