"""Lower a system of PDEs to the update of each field.

A second-order field ``f`` with ``f_tt = RHS`` steps as

    f_next = 2 f - f_prev + dt**2 * RHS

and RHS must be linear in the fields, each term a model-only coefficient times
either a second derivative along one axis or a field itself. A damping term
``s * f.dt`` in ``f``'s own equation (``p.dt2 = vp**2 * laplace(p) - b * p.dt``)
takes the centred difference ``(f_next - f_prev) / (2 dt)``; solved for
``f_next``,

    f_next = (2 f - (1 + s dt/2) f_prev + dt**2 * RHS) / (1 - s dt/2),

second order in ``dt``, and for ``s <= 0`` stable under the undamped step's
time-step bound. The terms are
gathered per field into groups of axes that share a coefficient, so that
``vp**2 * laplace(p)`` and ``vp**2 * (p.dz2 + p.dx2)`` lower to the same
``vp**2 * (d2z + d2x)``.

A first-order field ``f`` with ``f_t = RHS`` steps as ``f = f + dt * RHS``, in
the order the equations are written (a leapfrog: later equations read the
fields earlier ones updated), and RHS must be a model-only coefficient times
first derivatives of other fields. Each derivative is a staggered difference
from the field's position to the position of the field it updates (see
``lower_first_order``). RHS may also carry second derivatives of ``f`` itself
(``nu * laplace(f)``, ``kappa * f.dz2``: viscous and diffusive terms), taken on
``f``'s own position with the second-order equations' separable stencil, from
``f`` before its update (forward Euler). Such a term is dissipative: the step
can no longer be run backwards in time.

A first-order RHS may also carry fields without a derivative, each times a
model-only coefficient. ``s * f`` in ``f``'s own equation (a damping, a
relaxation ``-f / tau``) is stepped with the trapezoidal rule, stable for any
``dt`` when ``s <= 0``. ``c * g``, another field at ``f``'s position (a
coupling), is read at the time centre of ``f``'s step: its current value when
``g`` is updated in another sub-step of the leapfrog, the mean of its values
before and after its update when ``g`` is updated earlier in the same sub-step
(``g`` updated later in the same sub-step is refused).

Derivatives of order 3 and higher (``u.dx4``, ``v.dx3``) are admitted where
their result lands on the field they update, and are taken with
:class:`sweep.operators.Derivative` without a CPML correction:

* a second-order equation takes even orders of any field (``u.dx4``), centred
  on the shared grid, like its second derivatives;
* a first-order equation takes odd orders of the other fields as staggered
  differences, like its first derivatives (``v.dx3``), and even orders of its
  own field as dissipative terms (``-kappa * u.dx4``, a hyperdiffusion).

An odd derivative in a second-order equation, or of a first-order field in its
own equation (``u.dx3`` in ``u.dt``, the Airy / KdV term), would need a centred
odd stencil, which does not land between grid points and leaves odd and even
points decoupled, and forward Euler is unstable for it; both are refused.

Anything else is refused here, before code is written.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.core.symbol import Str

from ._expr import AXES, D1, D2, Coefficient, Dn, Dt, Field, Lap, Model


def axes_of(ndim):
    if ndim == 2:
        return ("z", "x")
    if ndim == 3:
        return AXES
    raise ValueError(f"ndim must be 2 or 3, got {ndim!r}")


@dataclass(frozen=True)
class Group:
    """``coef * (d^order field / d axes[0]^order + ...)``, axes in z, y, x order."""
    coef: sympy.Expr
    field: Field
    axes: tuple
    order: int = 2


@dataclass(frozen=True)
class Update:
    """``field_next = 2 field - field_prev + dt**2 * (groups + local + high)``,
    with a damping ``damping * field.dt`` taken centred in time."""
    field: Field
    groups: tuple
    local: tuple          # ((coef, field), ...)
    high: tuple = ()      # Groups of derivatives of order 4, 6, ...
    damping: sympy.Expr = sympy.S.Zero


@dataclass(frozen=True)
class Lowered:
    ndim: int
    fields: tuple         # updated fields, declaration order
    models: tuple         # model order of the propagator's ``models=``
    coefficients: tuple   # Coefficients the step uses, declaration order
    updates: tuple        # one per field, same order
    d2: tuple             # ((field, axis), ...) second derivatives taken
    high: tuple = ()      # ((field, axis, order), ...) derivatives of order 4, 6, ... taken

    @property
    def damped(self):
        """``((field, s), ...)`` of the fields with a damping ``s * f.dt``."""
        return tuple((u.field, u.damping) for u in self.updates if u.damping != 0)


def _solve_for_dt(eq):
    """``(field, order, rhs)`` of ``Eq(f.dt<order>, ...)`` solved for the time derivative."""
    expr = eq.lhs - eq.rhs
    dts = expr.atoms(Dt)
    if not dts:
        raise ValueError(f"{eq}: no time derivative; each equation advances one field, e.g. Eq(p.dt2, ...)")
    second = [d for d in dts if d.order == 2]
    if second and len(dts) > 1:
        # f.dt next to f.dt2: a damping of f, which the second-order lowering
        # takes from the right-hand side.
        if len(second) > 1:
            raise ValueError(f"{eq}: more than one second time derivative ({', '.join(map(str, second))})")
        others = [d for d in dts if d.order == 1 and d.field != second[0].field]
        if others:
            raise NotImplementedError(
                f"{eq}: {', '.join(map(str, others))} next to {second[0]} -- a first time "
                f"derivative of another field is not supported; only {second[0].field}.dt, "
                f"a damping of {second[0].field} itself, may stand next to {second[0]}")
        dts = set(second)
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
    return expr.atoms(D1, D2, Dn, Dt, Lap) | {s for s in expr.free_symbols if isinstance(s, Field)}


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
    """``{(field, axis): coef}`` of the second derivatives, ``{(field, axis,
    order): coef}`` of the even ones of order 4 and higher and ``{field: coef}``
    of the bare fields, with RHS == their sum."""
    axes = axes_of(ndim)
    for op in rhs.atoms(Dt):
        raise NotImplementedError(
            f"{eq}: {op} on the right-hand side is not supported; a first time derivative "
            "may only be the field's own (a damping f.dt in f's equation)")
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
    for op in rhs.atoms(Dn):
        if op.order % 2:
            raise NotImplementedError(
                f"{eq}: {op}, an odd derivative in a second-order equation: its centred stencil "
                "leaves odd and even grid points decoupled. Second-order fields take even "
                "derivatives (f.dx2, f.dx4); odd ones belong to a staggered first-order system")
    d2s = sorted(rhs.atoms(D2, Dn), key=sympy.default_sort_key)
    for op in d2s:
        if not isinstance(op.field, Field):
            raise TypeError(f"{eq}: {op} is not a derivative of a Field")
        if op.axis not in axes:
            raise ValueError(f"{eq}: {op} is a {op.axis}-derivative in a {ndim}-D build")
    holes = {op: sympy.Dummy(f"D{op.order}_{op.field}_{op.axis}") for op in d2s}
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

    d2, high, local = {}, {}, {}
    for atom, c in coefs.items():
        if isinstance(atom, D2):
            d2[(atom.field, atom.axis)] = c
        elif isinstance(atom, Dn):
            high[(atom.field, atom.axis, atom.order)] = c
        else:
            local[atom] = c
    return d2, high, local


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

    out, used, d2_taken, high_taken = [], set(), [], []
    for field in fields:
        eq, rhs = updates[field]
        rhs, damping = _split_damping(eq, field, rhs)
        d2, high, local = _decompose(eq, rhs, ndim)
        for f, *_ in (*d2, *high):
            if f not in updates:
                raise ValueError(f"{eq}: field {f} has no equation of its own")
        for f in local:
            if f not in updates:
                raise ValueError(f"{eq}: field {f} has no equation of its own")
        for c in (*d2.values(), *high.values(), *local.values(), damping):
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
        high_groups = _high_groups(high, fields, axes, high_taken)
        out.append(Update(field, tuple(groups), local_terms, high_groups, damping))

    model_order, coefs = _models_and_coefficients(used, models)
    d2_taken.sort(key=lambda fa: (fa[0]._order, axes.index(fa[1])))
    high_taken.sort(key=lambda t: (t[0]._order, t[2], axes.index(t[1])))
    return Lowered(ndim, fields, model_order, coefs, tuple(out), tuple(d2_taken), tuple(high_taken))


def _split_damping(eq, field, rhs):
    """``(rest, s)`` with ``rhs == rest + s * field.dt``, ``s`` model-only."""
    dt = Dt(field, 1)
    if not rhs.has(dt):
        return rhs, sympy.S.Zero
    D = sympy.Dummy("Dt1")
    e = sympy.expand(rhs.xreplace({dt: D}))
    s = sympy.diff(e, D)
    if s.has(D) or _fields_and_ops(s):
        raise ValueError(f"{eq}: not linear in the fields ({dt} is multiplied by a field or a derivative)")
    _check_coefficient(eq, dt, s)
    return sympy.expand(e - s * D), s


def _high_groups(high, fields, axes, taken):
    """``{(field, axis, order): coef}`` as Groups of the axes that share a
    coefficient, per field and order; records each derivative in ``taken``."""
    groups = []
    for f in fields:
        for n in sorted({n for g, _, n in high if g == f}):
            by_coef = {}
            for ax in axes:
                c = high.get((f, ax, n))
                if c is None or c == 0:
                    continue
                by_coef.setdefault(c, []).append(ax)
                if (f, ax, n) not in taken:
                    taken.append((f, ax, n))
            groups += [Group(c, f, tuple(a), n) for c, a in by_coef.items()]
    return tuple(groups)


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
    order: int = 1        # 3, 5, ...: a staggered difference of that order, no CPML


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
    update reads, in the order its CPML corrections run. ``d2``: the
    dissipative terms, one :class:`Group` of second derivatives of ``field``
    per distinct coefficient, after the terms in the sum."""
    field: Field
    terms: tuple
    derivs: tuple
    d2: tuple = ()
    damping: sympy.Expr = sympy.S.Zero   # s in ``s * field``: stepped with the trapezoidal rule
    local: tuple = ()                    # Local terms: other fields without a derivative


@dataclass(frozen=True)
class Local:
    """``coef * field`` in another field's equation, ``field`` without a
    derivative and at the same position. ``average``: ``field`` is updated
    earlier in the same sub-step (the same time level), so the term reads the
    mean of its values before and after that update (the trapezoidal rule)."""
    coef: sympy.Expr
    field: Field
    average: bool = False


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

    @property
    def dissipative(self):
        """The dissipative terms, ``((field, Group), ...)`` in the order written."""
        return tuple((up.field, g) for sub in self.substeps for up in sub.updates for g in up.d2)

    @property
    def damped(self):
        """``((field, s), ...)``: the fields whose equation carries ``s * field``."""
        return tuple((up.field, up.damping) for sub in self.substeps for up in sub.updates
                     if up.damping != 0)


def _axis_key(axes):
    """Sort key of a derivative: x, y, z (the order in which the sums are written)."""
    return lambda d: (-axes.index(d.axis), d.field._order, d.order)


def _split_dissipative(eq, field, rhs, axes, ndim):
    """``(rest, groups)``: the even derivatives of ``field`` itself (second, and
    fourth or higher) split off ``rhs`` as Groups of the axes that share a
    coefficient (z, y, x order, one order per Group), and what remains of
    ``rhs``, its terms kept as written. An even derivative of another field is
    refused: it would not land on ``field``'s position; so is an odd derivative
    of order 3 or higher of ``field`` itself."""
    for op in rhs.atoms(Dt):
        raise NotImplementedError(f"{eq}: {op} on the right-hand side is not supported")
    laps = {}
    for lap in rhs.atoms(Lap):
        if not isinstance(lap.field, Field):
            raise TypeError(f"{eq}: {lap} is not the Laplacian of a Field")
        laps[lap] = sympy.Add(*[D2(lap.field, Str(ax)) for ax in axes])
    rhs = rhs.xreplace(laps)
    for op in rhs.atoms(Dn):
        if op.order % 2 and op.field == field:
            raise NotImplementedError(
                f"{eq}: {op}, an odd derivative of {field} in its own equation (a dispersive "
                "term, as in the Airy or KdV equation): it would need a centred odd stencil, "
                "which leaves odd and even grid points decoupled, and forward Euler is unstable "
                "for it. Odd derivatives of the other fields are taken as staggered differences")
    ops = sorted([op for op in rhs.atoms(D2, Dn) if op.order % 2 == 0], key=sympy.default_sort_key)
    for op in ops:
        if not isinstance(op.field, Field):
            raise TypeError(f"{eq}: {op} is not a derivative of a Field")
        if op.field != field:
            raise NotImplementedError(
                f"{eq}: {op} -- a first-order equation takes even derivatives of its own field "
                f"only (a viscous or diffusive term such as nu*laplace({field}))")
        if op.axis not in axes:
            raise ValueError(f"{eq}: {op} is a {op.axis}-derivative in a {ndim}-D build")
    if not ops:
        return rhs, ()
    holes = {op: sympy.Dummy(f"D{op.order}_{op.field}_{op.axis}") for op in ops}
    e = rhs.xreplace(holes)
    slots = set(holes.values())
    coefs = {}
    for op, s in holes.items():
        c = sympy.diff(e, s)
        if c.free_symbols & slots or _fields_and_ops(c):
            raise ValueError(f"{eq}: not linear in the derivatives ({op} is multiplied by a field "
                             "or a derivative)")
        _check_coefficient(eq, op, c)
        coefs[(op.order, op.axis)] = c
    groups = []
    for n in sorted({n for n, _ in coefs}):
        by_coef = {}
        for ax in axes:
            c = coefs.get((n, ax))
            if c is not None and c != 0:
                by_coef.setdefault(c, []).append(ax)
        groups += [Group(c, field, tuple(a), n) for c, a in by_coef.items()]
    rest = e.xreplace({s: 0 for s in slots})
    return rest, tuple(groups)


def _split_local(eq, field, rhs):
    """``(rest, damping, local)``: the terms of ``rhs`` that are a model-only
    coefficient times a field without a derivative, split off -- ``damping``
    the coefficient of ``field`` itself, ``local`` ``[(coef, other), ...]`` --
    and what remains of ``rhs``, its terms kept as written."""
    ops = rhs.atoms(D1, D2, Dn, Lap)
    holes = {op: sympy.Dummy(f"op{k}") for k, op in enumerate(sorted(ops, key=sympy.default_sort_key))}
    e = rhs.xreplace(holes)
    bare = sorted((s for s in e.free_symbols if isinstance(s, Field)), key=sympy.default_sort_key)
    if not bare:
        return rhs, sympy.S.Zero, []
    slots = set(holes.values())
    damping, local = sympy.S.Zero, []
    for g in bare:
        c = sympy.diff(e, g)
        if c.free_symbols & slots or _fields_and_ops(c):
            raise ValueError(f"{eq}: not linear in the fields ({g} is multiplied by a field or a derivative)")
        _check_coefficient(eq, g, c)
        if g == field:
            damping = c
        else:
            local.append((c, g))
    rest = e.xreplace({g: 0 for g in bare}).xreplace({v: k for k, v in holes.items()})
    return rest, damping, local


def _first_order_terms(eq, rhs, dissipative=False):
    """``[(coef, (D1, ...)), ...]`` with ``rhs == sum(coef * sum(D1s))``: each top-level
    term of ``rhs`` as written (a coefficient times a derivative or a plain sum
    of derivatives), or term by term when it is not of that form. ``rhs`` is
    what :func:`_split_dissipative` left; ``dissipative``: it split off terms,
    so ``rhs`` may be empty (pure diffusion)."""
    ops = sorted(rhs.atoms(D1, Dn), key=sympy.default_sort_key)
    for op in ops:
        if not isinstance(op.field, Field):
            raise TypeError(f"{eq}: {op} is not a derivative of a Field")
    holes = {op: sympy.Dummy(f"D{op.order}_{op.field}_{op.axis}") for op in ops}
    e = rhs.xreplace(holes)
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
        if dissipative:
            return []
        raise ValueError(f"{eq}: the right-hand side has no derivative")

    terms = []
    for t in sympy.Add.make_args(rhs):
        c, g = t.as_independent(*ops, as_Add=False)
        parts = sympy.Add.make_args(g)
        if all(isinstance(p, (D1, Dn)) for p in parts) and len(set(parts)) == len(parts):
            terms.append((c, parts))
            continue
        for tt in sympy.Add.make_args(sympy.expand(t)):         # not a plain sum: term by term
            cc, gg = tt.as_independent(*ops, as_Add=False)
            if not isinstance(gg, (D1, Dn)):
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
            f"odd derivative is not supported; stagger one of them along {ax}")
    if abs(step) != 1:
        raise ValueError(f"{eq}: {source} and {target} sit a whole cell apart along {ax}; "
                         "a staggered difference spans half a cell")
    return Deriv(source, ax, step > 0, target.position(ax) != 0, op.order)


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

    The second derivatives of a dissipative term are taken just before the
    update of their field, from its value at the start of the step (the one
    update that changes it), and get no CPML correction.
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
        rhs, d2 = _split_dissipative(eq, field, rhs, axes, ndim)
        for g in d2:
            used |= g.coef.free_symbols
        rhs, damping, local = _split_local(eq, field, rhs)
        used |= damping.free_symbols
        for c, g in local:
            if g not in updates:
                raise ValueError(f"{eq}: field {g} has no equation of its own")
            moved = [a for a in axes if g.position(a) != field.position(a)]
            if moved:
                raise ValueError(
                    f"{eq}: {g} and {field} sit at different {moved[0]} positions, so {c*g} does "
                    f"not land on {field}; a term without a derivative needs the two fields "
                    f"staggered alike")
            used |= c.free_symbols
        terms = []
        for c, parts in _first_order_terms(eq, rhs, dissipative=bool(d2 or local or damping != 0)):
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
                       key=lambda d: (axes.index(d.axis), d.field._order, d.order))
        by_field[field] = (eq, FirstUpdate(field, tuple(terms), tuple(reads), d2, damping,
                                           tuple(Local(c, g) for c, g in local)))

    # One derivative of a field along an axis: one difference, one CPML memory.
    seen = {}
    for _, up in by_field.values():
        for d in up.derivs:
            other = seen.setdefault((d.field, d.axis, d.order), d)
            if other != d:
                raise NotImplementedError(
                    f"{d.field}.d{d.axis}{'' if d.order == 1 else d.order} is taken at two positions "
                    "(forward and backward differences); stagger the fields that read it alike")

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

    # A field read without a derivative: from an earlier or a later sub-step,
    # its current value is centred in time (the leapfrog); from the same
    # sub-step, the mean of its values before and after its update, so it must
    # be updated before the field that reads it.
    for k, run in enumerate(substeps):
        order = [up.field for up in run]
        for i, up in enumerate(run):
            if not up.local:
                continue
            local = []
            for t in up.local:
                if t.field in order[i + 1:]:
                    raise NotImplementedError(
                        f"{by_field[up.field][0]}: reads {t.field}, which is updated later in the same "
                        f"sub-step (the same time level); write {t.field}'s equation before "
                        f"{up.field}'s, so that the term reads the mean of {t.field} before and after "
                        "its update")
                local.append(Local(t.coef, t.field, t.field in order[:i]))
            run[i] = FirstUpdate(up.field, up.terms, up.derivs, up.d2, up.damping, tuple(local))

    out, taken_all = [], set()
    for run in substeps:
        taken = []
        for up in sorted(run, key=lambda u: u.field._order):
            for d in sorted(up.derivs, key=lambda d: (d.field._order, axes.index(d.axis), d.order)):
                if d not in taken:
                    taken.append(d)
        for d in taken:
            if d in taken_all:
                raise NotImplementedError(
                    f"{d.field}.d{d.axis}{'' if d.order == 1 else d.order} is read in two sub-steps, "
                    f"before and after {d.field} is updated; order the equations so that it is read in one")
        taken_all |= set(taken)
        out.append(SubStep(tuple(run), tuple(taken)))

    model_order, coefs = _models_and_coefficients(used, models)
    derivs = sorted(taken_all, key=lambda d: (d.field._order, -axes.index(d.axis), d.order))
    return LoweredFirst(ndim, fields, model_order, coefs, tuple(out), tuple(derivs))
