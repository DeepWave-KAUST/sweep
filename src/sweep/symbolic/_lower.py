"""Lower a system of PDEs to the update of each field.

A second-order field ``f`` with ``f_tt = RHS`` steps as

    f_next = 2 f - f_prev + dt**2 * RHS

and RHS must be linear in the fields, each term a model-only coefficient times
either a second derivative along one axis or a field itself. The terms are
gathered per field into groups of axes that share a coefficient, so that
``vp**2 * laplace(p)`` and ``vp**2 * (p.dz2 + p.dx2)`` lower to the same
``vp**2 * (d2z + d2x)``.

A first-order field ``f`` with ``f_t = RHS`` steps as ``f = f + dt * RHS``, in
the order the equations are written (a leapfrog: later equations read the
fields earlier ones updated), and RHS must be a model-only coefficient times
first derivatives of other fields. Each derivative is a staggered difference
from the field's position to the position of the field it updates (see
``lower_first_order``).

Anything else is refused here, before code is written.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.core.symbol import Str

from ._expr import AXES, D1, D2, Coefficient, Dt, Field, Lap, Model


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
    coefficients: tuple   # Coefficients the step uses, declaration order
    updates: tuple        # one per field, same order
    d2: tuple             # ((field, axis), ...) second derivatives taken


def _solve_for_dt(eq):
    """``(field, order, rhs)`` of ``Eq(f.dt<order>, ...)`` solved for the time derivative."""
    expr = eq.lhs - eq.rhs
    dts = expr.atoms(Dt)
    if not dts:
        raise ValueError(f"{eq}: no time derivative; each equation advances one field, e.g. Eq(p.dt2, ...)")
    orders = {d.order for d in dts}
    if len(orders) > 1:
        raise NotImplementedError(
            f"{eq}: {' and '.join(sorted(map(str, dts)))} -- a first time derivative next to "
            "a second one (no damping terms in f.dt) is not supported")
    if len(dts) > 1:
        raise ValueError(f"{eq}: more than one time derivative ({', '.join(map(str, dts))})")
    dt = dts.pop()
    field = dt.field
    if not isinstance(field, Field):
        raise TypeError(f"{eq}: {dt} is not the time derivative of a Field")
    if dt.order not in (1, 2):
        raise NotImplementedError(f"{eq}: {dt} -- time derivatives are of order 1 or 2")
    if dt.order != field.time_order:
        raise ValueError(
            f"{eq}: {dt} advances {field}, which was declared with time_order={field.time_order}; "
            f"declare Field({str(field)!r}, time_order={dt.order}, ...)")
    T = sympy.Dummy("T")
    e = expr.xreplace({dt: T})
    a = sympy.diff(e, T)
    if a.has(T) or a == 0:
        raise ValueError(f"{eq}: not linear in {dt}")
    if _fields_and_ops(a):
        raise ValueError(f"{eq}: the factor {a} of {dt} must not contain fields or derivatives")
    rhs = -e.xreplace({T: 0})
    return field, dt.order, (rhs if a == 1 else rhs / a)


def _fields_and_ops(expr):
    return expr.atoms(D1, D2, Dt, Lap) | {s for s in expr.free_symbols if isinstance(s, Field)}


def _check_coefficient(eq, atom, c):
    unknown = [s for s in c.free_symbols if not isinstance(s, (Model, Coefficient))]
    if unknown:
        raise ValueError(f"{eq}: {', '.join(map(str, unknown))} in the coefficient of {atom} "
                         "is neither a Model, a Coefficient nor a number")


def _models_and_coefficients(symbols, models=None):
    """The model order of the propagator's ``models=`` and the coefficients, in
    declaration order, that the coefficient ``symbols`` need: a coefficient's
    own models and coefficients count."""
    used_models, coefs, stack = set(), set(), list(symbols)
    while stack:
        s = stack.pop()
        if isinstance(s, Coefficient):
            if s not in coefs:
                coefs.add(s)
                stack.extend(s.expr.free_symbols)
        elif isinstance(s, Model):
            used_models.add(s)
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
    return model_order, tuple(sorted(coefs, key=lambda c: c._order))


def _decompose(eq, rhs, ndim):
    """``{(field, axis): coef}`` of the second derivatives and ``{field: coef}``
    of the bare fields, with RHS == their sum."""
    axes = axes_of(ndim)
    for op in rhs.atoms(Dt):
        raise NotImplementedError(f"{eq}: {op} on the right-hand side is not supported")
    for op in rhs.atoms(D1):
        raise NotImplementedError(
            f"{eq}: first spatial derivative {op} in a second-order equation, which takes "
            "second derivatives (f.dz2, f.dx2, laplace(f)); first derivatives belong to a "
            "first-order system (Field(..., time_order=1))")
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
        _check_coefficient(eq, atom, c)

    d2 = {}
    local = {}
    for atom, c in coefs.items():
        if isinstance(atom, D2):
            d2[(atom.field, atom.axis)] = c
        else:
            local[atom] = c
    return d2, local


def lower(equations, ndim, models=None):
    """Lower the system: a :class:`Lowered` for second-order fields, a
    :class:`LoweredFirst` for a first-order (staggered-grid) system."""
    solved = [(eq, *_solve_for_dt(eq)) for eq in equations]
    orders = {order for _, _, order, _ in solved}
    if len(orders) > 1:
        raise NotImplementedError(
            "the system mixes first-order (f.dt) and second-order (f.dt2) fields; "
            "mixed-order systems are not supported yet")
    if orders == {1}:
        return lower_first_order(solved, ndim, models)
    return _lower_second_order(solved, ndim, models)


def _by_field(solved):
    updates = {}
    for eq, field, _, rhs in solved:
        if field in updates:
            raise ValueError(f"field {field} has more than one equation")
        updates[field] = (eq, rhs)
    return updates


def _lower_second_order(solved, ndim, models):
    axes = axes_of(ndim)
    updates = _by_field(solved)
    fields = tuple(sorted(updates, key=lambda f: f._order))

    out, used, d2_taken = [], set(), []
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
            used |= c.free_symbols
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

    model_order, coefs = _models_and_coefficients(used, models)
    d2_taken.sort(key=lambda fa: (fa[0]._order, axes.index(fa[1])))
    return Lowered(ndim, fields, model_order, coefs, tuple(out), tuple(d2_taken))


# ---- first-order (staggered-grid) systems --------------------------------------

@dataclass(frozen=True)
class Deriv:
    """``d field / d axis`` taken at the position of the fields it updates: a
    forward difference when they sit half a cell after ``field`` along
    ``axis``, a backward one when half a cell before. ``half``: they sit half a
    cell off the integer node along ``axis``, so the CPML uses the half-node
    profiles."""
    field: Field
    axis: str
    forward: bool
    half: bool


@dataclass(frozen=True)
class Term:
    """``coef * (derivs[0] + derivs[1] + ...)``."""
    coef: sympy.Expr
    derivs: tuple


@dataclass(frozen=True)
class FirstUpdate:
    """``field = field + dt * RHS``. ``terms`` holds one Term per distinct
    coefficient: one Term is written ``field +- dt*coef * (derivs)``, several
    ``field + dt * (coef * (derivs) + ...)``. ``derivs``: every derivative the
    update reads, in the order its CPML corrections run."""
    field: Field
    terms: tuple
    derivs: tuple


@dataclass(frozen=True)
class SubStep:
    """Updates that read none of the fields they update: one shear of the
    leapfrog, reversible at ``-dt``. ``taken``: the derivatives, in the order
    they are taken."""
    updates: tuple
    taken: tuple


@dataclass(frozen=True)
class LoweredFirst:
    ndim: int
    fields: tuple         # updated fields, declaration order
    models: tuple
    coefficients: tuple
    substeps: tuple
    derivs: tuple         # every derivative taken, memory-field order


def _axis_key(axes):
    """Sort key of a derivative: x, y, z (the order in which the sums are written)."""
    return lambda d: (-axes.index(d.axis), d.field._order)


def _first_order_terms(eq, rhs):
    """``[(coef, (D1, ...)), ...]`` with ``rhs == sum(coef * sum(D1s))``: each top-level
    term of ``rhs`` as written (a coefficient times a derivative or a plain sum
    of derivatives), or term by term when it is not of that form."""
    for op in rhs.atoms(Dt):
        raise NotImplementedError(f"{eq}: {op} on the right-hand side is not supported")
    for op in rhs.atoms(D2, Lap):
        raise NotImplementedError(
            f"{eq}: {op} -- a first-order equation takes first derivatives only (f.dx, f.dz)")
    ops = sorted(rhs.atoms(D1), key=sympy.default_sort_key)
    for op in ops:
        if not isinstance(op.field, Field):
            raise TypeError(f"{eq}: {op} is not a derivative of a Field")
    holes = {op: sympy.Dummy(f"D1_{op.field}_{op.axis}") for op in ops}
    e = rhs.xreplace(holes)
    bare = sorted((s for s in e.free_symbols if isinstance(s, Field)), key=sympy.default_sort_key)
    if bare:
        raise NotImplementedError(
            f"{eq}: {', '.join(map(str, bare))} without a derivative -- a first-order equation "
            "takes first derivatives of the other fields only (no damping or coupling terms yet)")
    slots = set(holes.values())
    for s in slots:
        c = sympy.diff(e, s)
        if c.free_symbols & slots:
            raise ValueError(f"{eq}: not linear in the derivatives")
    rest = sympy.expand(e.xreplace({s: 0 for s in slots}))
    if rest != 0:
        raise ValueError(
            f"{eq}: the term {rest} contains no field; a source term is not part of "
            "the PDE (sources are injected by the propagator)")
    if not ops:
        raise ValueError(f"{eq}: the right-hand side has no derivative")

    terms = []
    for t in sympy.Add.make_args(rhs):
        c, g = t.as_independent(*ops, as_Add=False)
        parts = sympy.Add.make_args(g)
        if all(isinstance(p, D1) for p in parts) and len(set(parts)) == len(parts):
            terms.append((c, parts))
            continue
        for tt in sympy.Add.make_args(sympy.expand(t)):         # not a plain sum: term by term
            cc, gg = tt.as_independent(*ops, as_Add=False)
            if not isinstance(gg, D1):
                raise ValueError(f"{eq}: not linear in the derivatives ({tt})")
            terms.append((cc, (gg,)))
    for c, parts in terms:
        _check_coefficient(eq, parts[0], c)
    return terms


def _deriv(eq, target, op, axes, ndim):
    """The staggered difference of ``op`` taken at ``target``'s position."""
    source, ax = op.field, op.axis
    if ax not in axes:
        raise ValueError(f"{eq}: {op} is a {ax}-derivative in a {ndim}-D build")
    for a in axes:
        if a != ax and target.position(a) != source.position(a):
            raise ValueError(
                f"{eq}: {source} and {target} sit at different {a} positions, so {op} "
                f"does not land on {target}; stagger them alike along {a}")
    step = target.position(ax) - source.position(ax)
    if step == 0:
        raise NotImplementedError(
            f"{eq}: {source} and {target} sit at the same {ax} position -- a collocated "
            f"first derivative is not supported; stagger one of them along {ax}")
    if abs(step) != 1:
        raise ValueError(f"{eq}: {source} and {target} sit a whole cell apart along {ax}; "
                         "a staggered difference spans half a cell")
    return Deriv(source, ax, step > 0, target.position(ax) != 0)


def lower_first_order(solved, ndim, models=None):
    """Lower a first-order system.

    The statements come in an order that reproduces the shipped 2-D staggered
    equations (``Acoustic1st``, ``Elastic``) bit for bit, gradient included --
    autograd accumulates in the order the graph was built:

    * the equations split, in the order written, into sub-steps: runs of
      equations that read none of the fields the run updates;
    * a sub-step first takes its derivatives: per updated field in declaration
      order, its derivatives by the declaration order of their fields;
    * then, per equation in the order written, the CPML corrections of its
      derivatives not corrected yet (along z, y, x), then its update.
    """
    axes = axes_of(ndim)
    updates = _by_field(solved)
    fields = tuple(sorted(updates, key=lambda f: f._order))
    for f in fields:
        bad = [a for a in f.stagger if a not in axes]
        if bad:
            raise ValueError(f"field {f} is staggered along {bad[0]} in a {ndim}-D build")

    used, by_field = set(), {}
    for eq, field, _, rhs in solved:
        terms = []
        for c, parts in _first_order_terms(eq, rhs):
            derivs = []
            for op in parts:
                if op.field not in updates:
                    raise ValueError(f"{eq}: field {op.field} has no equation of its own")
                if op.field == field:
                    raise ValueError(f"{eq}: {field}'s equation reads {op}, a derivative of itself")
                derivs.append(_deriv(eq, field, op, axes, ndim))
            used |= c.free_symbols
            terms.append((c, tuple(sorted(derivs, key=_axis_key(axes)))))
        # One term per distinct coefficient (written once, as a sum of derivatives).
        merged = {}
        for c, derivs in terms:
            merged[c] = tuple(sorted(merged.get(c, ()) + derivs, key=_axis_key(axes)))
        terms = sorted((Term(c, d) for c, d in merged.items()), key=lambda t: _axis_key(axes)(t.derivs[0]))
        reads = sorted({d for t in terms for d in t.derivs},
                       key=lambda d: (axes.index(d.axis), d.field._order))
        by_field[field] = (eq, FirstUpdate(field, tuple(terms), tuple(reads)))

    # One derivative of a field along an axis: one difference, one CPML memory.
    seen = {}
    for _, up in by_field.values():
        for d in up.derivs:
            other = seen.setdefault((d.field, d.axis), d)
            if other != d:
                raise NotImplementedError(
                    f"{d.field}.d{d.axis} is taken at two positions (forward and backward "
                    "differences); stagger the fields that read it alike")

    substeps, run, written, read = [], [], set(), set()
    for eq, field, _, _ in solved:
        up = by_field[field][1]
        reads = {d.field for d in up.derivs}
        if run and (reads & written or field in read):
            substeps.append(run)
            run, written, read = [], set(), set()
        run.append(up)
        written.add(field)
        read |= reads
    substeps.append(run)

    out, taken_all = [], set()
    for run in substeps:
        taken = []
        for up in sorted(run, key=lambda u: u.field._order):
            for d in sorted(up.derivs, key=lambda d: (d.field._order, axes.index(d.axis))):
                if d not in taken:
                    taken.append(d)
        for d in taken:
            if d in taken_all:
                raise NotImplementedError(
                    f"{d.field}.d{d.axis} is read in two sub-steps, before and after {d.field} "
                    "is updated; order the equations so that it is read in one")
        taken_all |= set(taken)
        out.append(SubStep(tuple(run), tuple(taken)))

    model_order, coefs = _models_and_coefficients(used, models)
    derivs = sorted(taken_all, key=lambda d: (d.field._order, -axes.index(d.axis)))
    return LoweredFirst(ndim, fields, model_order, coefs, tuple(out), tuple(derivs))
