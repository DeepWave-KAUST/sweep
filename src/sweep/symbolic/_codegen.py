"""Write the eager equation class of a lowered PDE as Python source.

The step is written the way the hand-written equations write theirs
(``sweep.equations.acoustic`` / ``acoustic3d`` for second-order fields,
``acoustic1st`` / ``elastic`` for staggered first-order systems): the same
operators, the same CPML call, the same order of operations, the same
products of models and ``dt`` made once per rollout, so a PDE that restates a
shipped equation reproduces it bit for bit.
"""
from __future__ import annotations

import itertools

import sympy
from sympy.core.symbol import Str
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

from ._expr import D2, Dn, Lap
from ._lower import axes_of

_TORCH_FUNCS = {"sqrt": "sqrt", "exp": "exp", "log": "log", "sin": "sin", "cos": "cos",
                "tan": "tan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "Abs": "abs"}


class _TorchPrinter(StrPrinter):
    """sympy's ``str`` with the elementary functions as ``torch.*`` calls and
    constant subexpressions folded to float literals."""

    def _print_Function(self, expr):
        if expr.is_number:
            return repr(float(expr))
        name = _TORCH_FUNCS.get(type(expr).__name__)
        if name is None:
            raise NotImplementedError(f"function {type(expr).__name__} in a coefficient has no torch equivalent here")
        return f"torch.{name}({', '.join(self._print(a) for a in expr.args)})"

    def _print_Pow(self, expr, rational=False):
        if expr.is_number and not (expr.base.is_Integer and expr.exp.is_Integer):
            return repr(float(expr))
        if expr.exp == sympy.S.Half:
            return f"torch.sqrt({self._print(expr.base)})"
        if expr.exp == -sympy.S.Half:
            return f"1 / torch.sqrt({self._print(expr.base)})"
        return super()._print_Pow(expr, rational)

    def _print_Pi(self, expr):
        return repr(float(expr))

    def _print_Exp1(self, expr):
        return repr(float(expr))


_printer = _TorchPrinter()


def _once(text):
    """``text``, a product of models, coefficients, ``dt`` and numbers, made once
    per rollout by ``sweep.equations._invariants.once``, as the shipped
    equations make theirs (``vp**2 * dt**2``, ``dt / rho``). The key is the
    text: the same product in several equations is one tensor."""
    return f"once(invariants, {text!r}, lambda: {text})"


def _coef_times(coef, rest):
    """``coef * dt**2 * rest`` with ``coef`` printed at multiplication
    precedence and ``coef * dt**2`` made once per rollout."""
    if coef == 1:
        return f"dt**2 * {rest}"
    c = f"{_printer.parenthesize(coef, PRECEDENCE['Mul'])} * dt**2"
    return f"{_once(c) if coef.free_symbols else c} * {rest}"


class Names:
    """Every name the generated class exposes or uses for a field."""

    def __init__(self, low):
        self.multi = len(low.fields) > 1
        # CPML memory fields: x before z, as the shipped equations list them.
        self.cpml = sorted(low.d2, key=lambda fa: (fa[0]._order, -axes_of(low.ndim).index(fa[1])))

    def prev(self, f):
        return f"{f}_prev"

    def psi(self, f, ax):
        return f"psi{ax}_{f}" if self.multi else f"psi{ax}"

    def zeta(self, f, ax):
        return f"zeta{ax}_{f}" if self.multi else f"zeta{ax}"


def _q(s):
    return repr(s)


def _dn_call(field, axis, order, stagger, spacing):
    """The call of a derivative of order 3 or higher: the class's
    ``sweep.operators.Derivative`` of that order and stagger (``self._dn``)."""
    return f"self._dn[{order}, {stagger!r}]({field}, {axis!r}, {spacing})"


def _high_name(field, axis, order):
    return f"_d{order}{axis}_{field}"


def _unpack(names):
    return f"({names[0]},)" if len(names) == 1 else ", ".join(names)


def _model_lines(low):
    """The step's first lines on the models: the coefficients come from
    ``prepare_models``, which the propagator runs once per run."""
    mods = [str(m) for m in low.models]
    if low.coefficients:
        return [f"{_unpack(mods + [str(c) for c in low.coefficients])} = self.prepare_models(models)"]
    return [f"{_unpack(mods)} = models"] if mods else []


def _prepare_models(low):
    """The ``prepare_models`` method: the models, then each coefficient, computed
    once per run in declaration order. Prepared models pass through, so the
    step may call it again."""
    if not low.coefficients:
        return []
    mods = [str(m) for m in low.models]
    names = mods + [str(c) for c in low.coefficients]
    lines = [
        "def prepare_models(self, models):",
        '    """The models followed by the coefficients, computed once per run."""',
        f"    if len(models) == {len(names)}:",
        "        return list(models)",
        f"    {_unpack(mods)} = models",
    ]
    lines += [f"    {c} = {_printer.doprint(c.expr)}" for c in low.coefficients]
    lines += [f"    return [{', '.join(names)}]", ""]
    return [f"    {line}" if line else "" for line in lines]


def _model_specs(low):
    out = []
    for m in low.models:
        aliases = f", aliases={tuple(m.aliases)!r}" if m.aliases else ""
        out.append(f"ModelSpec({_q(str(m))}{aliases}, description={_q(m.description)}, unit={_q(m.unit)})")
    return out


def _check_names(names):
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f"name(s) {', '.join(clash)} used twice (fields, models, coefficients, "
                         "aliases and the generated fields share one namespace)")


def generate(name, low, *, free_surface, pml, doc):
    nm = Names(low)
    ndim = low.ndim
    axes = axes_of(ndim)
    axis_index = {ax: -ndim + i for i, ax in enumerate(axes)}
    use_cpml = pml == "cpml"

    # ---- field and model tables ----------------------------------------------
    specs = []
    for f in low.fields:
        aliases = f", aliases={tuple(f.aliases)!r}" if f.aliases else ""
        specs.append(f"FieldSpec({_q(str(f))}{aliases}, description={_q(f.description)}, "
                     f"supports_source={f.source}, supports_receiver={f.receiver})")
        specs.append(f"FieldSpec({_q(nm.prev(f))}, description={_q(f'Previous time level of `{f}`.')}, internal=True)")
    if use_cpml:
        for kind, namer in (("memory variable", nm.psi), ("auxiliary wavefield", nm.zeta)):
            for f, ax in nm.cpml:
                specs.append(f"FieldSpec({_q(namer(f, ax))}, description={_q(f'CPML {kind} of the {ax}-derivative of `{f}`.')}, "
                             f"internal=True, boundary_related=True, cpml_axis={_q(ax)})")
    names = [str(f) for f in low.fields] + [nm.prev(f) for f in low.fields]
    if use_cpml:
        names += [n(f, ax) for f, ax in nm.cpml for n in (nm.psi, nm.zeta)]
    names += [a for f in low.fields for a in f.aliases]
    names += [str(m) for m in low.models] + [a for m in low.models for a in m.aliases]
    names += [str(c) for c in low.coefficients]
    _check_names(names)
    models = _model_specs(low)

    src_fields = [str(f) for f in low.fields if f.source]
    rec_fields = [str(f) for f in low.fields if f.receiver]

    # ---- the step -------------------------------------------------------------
    sp = ", ".join(f"_h{ax}" for ax in axes)
    body = []
    temps = []  # the tensors the step computes and does not return
    state = []
    for f in low.fields:
        state += [str(f), nm.prev(f)]
    if use_cpml:
        state += [f"_{n(f, ax)}" for n in (nm.psi, nm.zeta) for f, ax in nm.cpml]
    body.append(f"{', '.join(state)} = wavefields")
    body += _model_lines(low)
    if low.d2 or low.high:
        body.append(f"{sp} = self._spacings_{ndim}d(h)")
        if use_cpml and low.d2:
            prof = ", ".join(f"_a{ax}, _b{ax}, _db{ax}" for ax in axes)
            body.append(f"{prof} = self.b")

    # The statements come in the order of the shipped equations: every
    # Laplacian, then every first derivative, then the CPML update of each axis
    # followed by its in-place sum. Autograd accumulates gradients and Inductor
    # fuses in the order the graph was built, so another order of the same
    # operations gives a gradient a few ulp away from theirs.
    for f in low.fields:
        if not any(g == f for g, _ in low.d2):
            continue
        laps = [f"_lap{ax}_{f}" for ax in axes]
        temps += laps
        body.append(f"{', '.join(laps)} = self.separable_d2_{ndim}d({f}, self.laplace_kernels, {sp})")
        if use_cpml:
            body.append(f"_src_{f} = ({f}, self.laplace_kernels, ({sp}))")
    if use_cpml:
        for f, ax in low.d2:
            temps.append(f"_d{ax}_{f}")
            body.append(f"_d{ax}_{f} = self.gradient({f}, h, {axis_index[ax]}, kernels=self.grad_kernels)")

    def d2_of(f, ax):
        return f"_d2{ax}_{f}" if use_cpml else f"_lap{ax}_{f}"

    # A group of several axes is summed into ``_w<k>``; a one-axis group is its
    # derivative.
    groups = [g for up in low.updates for g in up.groups]
    multi = [g for g in groups if len(g.axes) > 1]
    sums = {id(g): d2_of(g.field, g.axes[0]) for g in groups}
    sums.update({id(g): f"_w{k}" for k, g in enumerate(multi)})
    body += [f"{sums[id(g)]} = 0." for g in multi]
    temps += [sums[id(g)] for g in multi]
    for f, ax in low.d2:
        if use_cpml:
            i = axis_index[ax]
            psi, zeta = nm.psi(f, ax), nm.zeta(f, ax)
            temps.append(f"_d2{ax}_{f}")
            body.append(f"_d2{ax}_{f}, _{psi}_n, _{zeta}_n = cpml_axis_update(")
            body.append(f"    _lap{ax}_{f}, _d{ax}_{f}, _{psi}, _{zeta}, _a{ax}, _b{ax}, _db{ax}, h, {i},")
            body.append(f"    self.gradient, self.grad_kernels, _src_{f})")
        body += [f"{sums[id(g)]} += {d2_of(f, ax)}" for g in multi if g.field == f and ax in g.axes]

    # Even derivatives of order 4 and higher: centred, without CPML, a group of
    # several axes summed into ``_v<k>``.
    for f, ax, n in low.high:
        temps.append(_high_name(f, ax, n))
        body.append(f"{_high_name(f, ax, n)} = {_dn_call(f, ax, n, None, f'_h{ax}')}")
    high = [g for up in low.updates for g in up.high]
    hsums = {id(g): _high_name(g.field, g.axes[0], g.order) for g in high}
    for k, g in enumerate(g for g in high if len(g.axes) > 1):
        hsums[id(g)] = f"_v{k}"
        temps.append(f"_v{k}")
        body.append(f"_v{k} = {' + '.join(_high_name(g.field, ax, g.order) for ax in g.axes)}")

    for up in low.updates:
        f = up.field
        terms = [_coef_times(g.coef, sums[id(g)]) for g in up.groups]
        for c, g in up.local:
            terms.append(_coef_times(c, str(g)))
        terms += [_coef_times(g.coef, hsums[id(g)]) for g in up.high]
        if up.damping == 0:
            rhs = f"2 * {f} - {nm.prev(f)}" + "".join(f" + {t}" for t in terms)
        else:
            rhs = _damped_second(f, nm.prev(f), up.damping, terms)
        body.append(f"_{f}_next = {rhs}")

    out = []
    for f in low.fields:
        out += [f"_{f}_next", str(f)]
    if use_cpml:
        out += [f"_{n(f, ax)}_n" for n in (nm.psi, nm.zeta) for f, ax in nm.cpml]
    # The shipped equations compute these inside step_cpml(), which frees them
    # on return. Kept to the end of an uncompiled step, they would live through
    # the free surface's new outputs: 3-4 fields more peak memory than Acoustic,
    # 5-7 than Acoustic3D, in Full and checkpointed runs alike.
    if temps:
        body.append(f"del {', '.join(temps)}")
    body.append(f"return self._free_surface_out(({', '.join(out)},))")

    # Per-shot batched models (B, nz, nx): the eager layout inserts the channel
    # axis and the step is plain broadcasting; the propagator admits the layout
    # in 2-D only, as for the shipped Acoustic and Elastic.
    flags = [f"supports_batched_models = {ndim == 2}"]
    if free_surface is not None:
        flags.append(f"supports_per_edge_free_surface = {ndim == 2}")
        flags.append("supports_image_topography = True")

    text = "\n".join(f"    {line}" for line in doc).replace("\\", "\\\\").replace('"""', "'''")
    lines = [
        f'"""{name}: generated by sweep.symbolic from\n\n{text}\n"""',
        "import torch  # noqa: F401  (elementary functions in coefficients)",
        "",
        "from sweep.equations._cpml import cpml_axis_update  # noqa: F401",
        "from sweep.equations._invariants import once  # noqa: F401",
        "from sweep.equations.fields import FieldSpec, ModelSpec",
        "from sweep.symbolic._runtime import SymbolicSecondOrderEquation",
        "",
        "",
        f"class {name}(SymbolicSecondOrderEquation):",
        '    """Generated by sweep.symbolic; the PDE is in the module docstring."""',
        "",
        f"    NDIM = {ndim}",
        f"    SYMBOLIC_FREE_SURFACE = {free_surface!r}",
        f"    SYMBOLIC_PML = {pml!r}",
        f"    DEFAULT_SOURCE_FIELDS = {tuple(src_fields[:1])!r}",
        f"    DEFAULT_RECEIVER_FIELDS = {tuple(rec_fields[:1])!r}",
        *([f"    HIGH_ORDER_STENCILS = {tuple(sorted({(n, None) for _, _, n in low.high}))!r}"]
          if low.high else []),
        *([f"    DAMPING_TERMS = {damping_terms(low)!r}"] if low.damped else []),
        '    default_pml_type = "cpmlr"',
        "    takes_invariants = True",
        *[f"    {fl}" for fl in flags],
        "",
        "    MODEL_SPECS = (",
        *[f"        {m}," for m in models],
        "    )",
        "    FIELD_SPECS = (",
        *[f"        {s}," for s in specs],
        "    )",
        "",
        *_prepare_models(low),
        "    def func(self, wavefields, models, dt, h, b, invariants=None, **kwargs):",
        *[f"        {line}" for line in body],
        "",
    ]
    return "\n".join(lines)


def damping_terms(low):
    """The damping terms ``s * f.dt`` of a second-order system as text."""
    return tuple(f"{sympy.sstr(s * sympy.Symbol(f'{f}.dt'))}" for f, s in low.damped)


def _damped_second(f, prev, damping, terms):
    """``f_tt = R + s f_t`` with ``f_t`` centred, ``(f_next - f_prev) / (2 dt)``:
    ``f_next = (2 f - (1 + s dt/2) f_prev + dt**2 R) / (1 - s dt/2)``; the two
    factors are made once per rollout."""
    half = damping * _DT / 2
    a = _once(_printer.doprint(1 / (1 - half)))
    b = _once(_printer.doprint(1 + half))
    return f"{a} * (2 * {f} - {b} * {prev}" + "".join(f" + {t}" for t in terms) + ")"


# ---- first-order (staggered-grid) systems --------------------------------------

def _dname(d):
    return f"_d{d.axis}_{d.field}" if d.order == 1 else f"_d{d.order}{d.axis}_{d.field}"


def _mname(d):
    return f"m_{d.field}_{d.axis}"


def _plain_sum(derivs):
    names = [_dname(d) for d in derivs]
    return names[0] if len(names) == 1 else f"({' + '.join(names)})"


def _lap_name(field, ax):
    return f"_lap{ax}_{field}"


def _group_name(group, ax):
    """The derivative of a dissipative group along ``ax``: the separable second
    derivative, or a higher even one."""
    return _lap_name(group.field, ax) if group.order == 2 else _high_name(group.field, ax, group.order)


def _lap_sum(group):
    names = [_group_name(group, ax) for ax in group.axes]
    return names[0] if len(names) == 1 else f"({' + '.join(names)})"


def dissipative_terms(low):
    """The dissipative terms of a lowered first-order system, as text:
    ``nu*laplace(vx)`` for a group over every axis, ``kappa*T.dz2`` otherwise."""
    axes = axes_of(low.ndim)
    out = []
    for field, g in low.dissipative:
        if g.order == 2:
            op = Lap(field) if g.axes == axes else sympy.Add(*[D2(field, Str(ax)) for ax in g.axes])
        else:
            op = sympy.Add(*[Dn(field, Str(ax), g.order) for ax in g.axes])
        out.append(str(g.coef * op))
    return tuple(out)


def _first_update(up, shared, cache=True):
    """``f = f + dt * RHS``, its products associated as the shipped equations
    associate theirs: one coefficient goes with ``dt`` (``dt/rho * (...)``,
    ``rho*vp**2*dt * (...)``), several are summed inside ``dt * (...)``.
    ``shared``: the sums of derivatives computed once for the sub-step. The
    dissipative terms come last, each a coefficient times its second
    derivatives (``nu*(_lapz_vx + _lapx_vx)``). With ``cache``, a product of
    models and ``dt`` is made once per rollout (``dt/rho``, ``2*mu``; a bare
    model or coefficient is used as it is); the sub-steps of boundary saving,
    run at ``-dt``, make theirs in place."""
    f = str(up.field)

    def _sum(derivs):
        return shared.get(derivs) or _plain_sum(derivs)

    items = [(t.coef, _sum(t.derivs)) for t in up.terms] + [(g.coef, _lap_sum(g)) for g in up.d2]
    items += [(t.coef / 2, f"({_old_name(t.field)} + {t.field})") if t.average else (t.coef, str(t.field))
              for t in up.local]
    if up.damping != 0:
        return _damped_update(f, up.damping, items, cache)
    if len(items) == 1:
        c, s = items[0]
        sign = "+"
        if c.could_extract_minus_sign():
            c, sign = -c, "-"
        num, den = sympy.fraction(c)
        # (parenthesize: strict=False brackets an item of the level's own
        # precedence too -- a product in the denominator.)
        if c == 1:
            dtc = "dt"
        elif num == 1:
            dtc = f"dt/{_printer.parenthesize(den, PRECEDENCE['Mul'])}"
        else:
            dtc = f"{_printer.parenthesize(c, PRECEDENCE['Mul'], strict=True)}*dt"
        if cache and c != 1 and c.free_symbols:
            dtc = _once(dtc)
        return f"{f} = {f} {sign} {dtc} * {s}"
    return f"{f} = {f} + dt * ({_sum_items(items, cache)})"


def _old_name(field):
    """The value of ``field`` before its update, kept for a term that reads its mean."""
    return f"_old_{field}"


def _sum_items(items, cache):
    """``c0*s0 + c1*s1 - ...``: products of models made once per rollout."""
    parts = []
    for k, (c, s) in enumerate(items):
        neg = c.could_extract_minus_sign()
        if neg:
            c = -c
        if c == 1:
            term = s
        else:
            text = _printer.parenthesize(c, PRECEDENCE['Mul'], strict=True)
            if cache and c.free_symbols and not c.is_Symbol:
                text = _once(text)
            term = f"{text}*{s}"
        parts.append(("-" if neg else "") + term if k == 0 else (" - " if neg else " + ") + term)
    return "".join(parts)


_DT = sympy.Symbol("dt")


def _damped_update(f, damping, items, cache):
    """``f_t = s f + R`` with the trapezoidal rule, ``f = f + dt * (s (f + f_next) / 2 + R)``
    solved for ``f_next``: ``f = (1 + s dt/2) / (1 - s dt/2) * f + dt / (1 - s dt/2) * R``.
    Stable for any ``dt`` when ``s <= 0`` (a relaxation, however fast), second order
    in ``dt``; the two factors are made once per rollout."""
    half = damping * _DT / 2
    a = _printer.doprint((1 + half) / (1 - half))
    b = _printer.doprint(_DT / (1 - half))
    if cache:
        a, b = _once(a), _once(b)
    else:
        a, b = f"({a})", f"({b})"
    if not items:
        return f"{f} = {a} * {f}"
    return f"{f} = {a} * {f} + {b} * ({_sum_items(items, cache)})"


def _substep_lines(sub, use_cpml, counter, ndim, free=True, cache=True):
    """One sub-step: its derivatives, then per update its CPML corrections and
    the update. A sum of derivatives that several updates read (the divergence
    of the 3-D stresses) is computed once, before the first of them: written
    out per update, autograd would keep a copy per update for the backward.
    The temporaries are freed at the sub-step's end (``free``), as the shipped
    equations free theirs on returning from a sub-step function. The second
    derivatives of a dissipative term are taken just before their field's
    update, with the second-order equations' separable stencil and no CPML
    correction, and freed after it."""
    axes = axes_of(ndim)
    spacings = ", ".join(f"_h{ax}" for ax in axes)
    lines = []
    for d in sub.taken:
        side = "forward" if d.forward else "backward"
        call = (f"pd.{d.axis}_{side}({d.field})" if d.order == 1
                else _dn_call(d.field, d.axis, d.order, side, f"pd._spacing[{d.axis!r}]"))
        lines.append(f"{_dname(d)} = {call}")
    sums = [t.derivs for up in sub.updates for t in up.terms if len(t.derivs) > 1]
    names = {}
    for derivs in sums:
        if sums.count(derivs) > 1 and derivs not in names:
            names[derivs] = f"_s{next(counter)}"
    done, shared = set(), {}
    averaged = [t.field for up in sub.updates for t in up.local if t.average]
    for up in sub.updates:
        if up.field in averaged:
            lines.append(f"{_old_name(up.field)} = {up.field}")
        for d in (up.derivs if use_cpml else ()):
            if d in done or d.order > 1:          # no CPML above first order
                continue
            done.add(d)
            prof = f"_a{d.axis}h, _b{d.axis}h" if d.half else f"_a{d.axis}, _b{d.axis}"
            kernel = f"pd.k{d.axis}{'f' if d.forward else 'b'}"
            lines.append(f"{_dname(d)}, _{_mname(d)} = cpml_update({_dname(d)}, _{_mname(d)}, {prof}, "
                         f"({d.field}, {kernel}, pd._spacing[{d.axis!r}]))")
        for t in up.terms:
            if t.derivs in names and t.derivs not in shared:
                shared[t.derivs] = names[t.derivs]
                lines.append(f"{names[t.derivs]} = {' + '.join(_dname(d) for d in t.derivs)}")
        laps = [_lap_name(up.field, ax) for ax in axes] if any(g.order == 2 for g in up.d2) else []
        if laps:
            lines.append(f"{', '.join(laps)} = self.separable_d2_{ndim}d({up.field}, self.laplace_kernels, "
                         f"{spacings})")
        highs = [(g.order, ax) for g in up.d2 if g.order > 2 for ax in g.axes]
        for n, ax in highs:
            lines.append(f"{_high_name(up.field, ax, n)} = {_dn_call(up.field, ax, n, None, f'_h{ax}')}")
        lines.append(_first_update(up, shared, cache))
        dead = laps + [_high_name(up.field, ax, n) for n, ax in highs]
        if dead:
            lines.append(f"del {', '.join(dead)}")
    temps = ([_dname(d) for d in sub.taken] + list(names.values())
             + [_old_name(f) for f in dict.fromkeys(averaged)])
    if free and temps:
        lines.append(f"del {', '.join(temps)}")
    return lines


def generate_first_order(name, low, *, pml, doc):
    ndim = low.ndim
    axes = axes_of(ndim)
    use_cpml = pml == "cpml"
    mem = tuple(d for d in low.derivs if d.order == 1) if use_cpml else ()
    dissipative = dissipative_terms(low)
    damping = tuple(str(c * f) for f, c in low.damped)
    stencils = sorted({(d.order, "forward" if d.forward else "backward") for d in low.derivs if d.order > 1}
                      | {(g.order, None) for _, g in low.dissipative if g.order > 2}, key=repr)

    specs = []
    for f in low.fields:
        aliases = f", aliases={tuple(f.aliases)!r}" if f.aliases else ""
        specs.append(f"FieldSpec({_q(str(f))}{aliases}, description={_q(f.description)}, "
                     f"supports_source={f.source}, supports_receiver={f.receiver})")
    for d in mem:
        specs.append(f"FieldSpec({_q(_mname(d))}, description={_q(f'CPML memory variable of the {d.axis}-derivative of `{d.field}`.')}, "
                     f"internal=True, boundary_related=True, cpml_axis={_q(d.axis)})")
    names = [str(f) for f in low.fields] + [_mname(d) for d in mem]
    names += [a for f in low.fields for a in f.aliases]
    names += [str(m) for m in low.models] + [a for m in low.models for a in m.aliases]
    names += [str(c) for c in low.coefficients]
    _check_names(names)

    phys = [str(f) for f in low.fields]
    state = phys + [f"_{_mname(d)}" for d in mem]
    # A single field (a diffusion) unpacks as ``(T,) = wavefields``; without a
    # first derivative the step needs no ``pd``.
    body = [f"{_unpack(state)} = wavefields", *_model_lines(low), *(["pd = self.pd"] if low.derivs else [])]
    if dissipative:
        body.append(f"{', '.join(f'_h{ax}' for ax in axes)} = self._spacings_{ndim}d(h)")
    if mem:
        body.append(f"{', '.join(f'_a{ax}, _b{ax}, _a{ax}h, _b{ax}h' for ax in axes)} = self.b")
    counter = itertools.count()
    for sub in low.substeps:
        body.append(f"# {', '.join(str(u.field) for u in sub.updates)}")
        body += _substep_lines(sub, use_cpml, counter, ndim)
    body.append(f"return ({', '.join(state)},)")

    # A dissipative or damped system has no lossless sub-steps to run
    # backwards: the class refuses boundary saving instead
    # (SymbolicFirstOrderEquation).
    substeps = []
    if not dissipative and not damping:
        subs = []
        for k, sub in enumerate(low.substeps):
            subs += [f"def _sub{k}(wavefields, models, dt, h):",
                     f"    {_unpack(phys)} = wavefields[:{len(phys)}]",
                     *[f"    {line}" for line in _model_lines(low)],
                     *[f"    {line}" for line in _substep_lines(sub, False, itertools.count(), ndim, free=False,
                                                           cache=False)],
                     f"    return [{', '.join(phys)}, *wavefields[{len(phys)}:]]",
                     ""]
        subs.append(f"return [{', '.join(f'_sub{k}' for k in range(len(low.substeps)))}]")
        substeps = [
            "",
            "    def interior_substeps(self):",
            '        """The lossless leapfrog as sub-steps for eager boundary saving: each',
            "        updates its fields from the others only, so the same code at -dt",
            '        inverts it."""',
            "        pd = self.pd",
            "",
            *[f"        {line}" if line else "" for line in subs],
        ]

    src_fields = [str(f) for f in low.fields if f.source]
    rec_fields = [str(f) for f in low.fields if f.receiver]
    text = "\n".join(f"    {line}" for line in doc).replace("\\", "\\\\").replace('"""', "'''")
    lines = [
        f'"""{name}: generated by sweep.symbolic from\n\n{text}\n"""',
        "import torch  # noqa: F401  (elementary functions in coefficients)",
        "",
        "from sweep.equations._cpml import cpml_update  # noqa: F401",
        "from sweep.equations._invariants import once  # noqa: F401",
        "from sweep.equations.fields import FieldSpec, ModelSpec",
        "from sweep.symbolic._runtime import SymbolicFirstOrderEquation",
        "",
        "",
        f"class {name}(SymbolicFirstOrderEquation):",
        '    """Generated by sweep.symbolic; the PDE is in the module docstring."""',
        "",
        f"    NDIM = {ndim}",
        f"    SYMBOLIC_PML = {pml!r}",
        f"    DEFAULT_SOURCE_FIELDS = {tuple(src_fields[:1])!r}",
        f"    DEFAULT_RECEIVER_FIELDS = {tuple(rec_fields[:1])!r}",
        *([f"    DISSIPATIVE_TERMS = {dissipative!r}"] if dissipative else []),
        *([f"    DAMPING_TERMS = {damping!r}"] if damping else []),
        *([f"    HIGH_ORDER_STENCILS = {tuple(stencils)!r}"] if stencils else []),
        "    takes_invariants = True",
        f"    supports_batched_models = {ndim == 2}",
        "",
        "    MODEL_SPECS = (",
        *[f"        {m}," for m in _model_specs(low)],
        "    )",
        "    FIELD_SPECS = (",
        *[f"        {s}," for s in specs],
        "    )",
        "",
        *_prepare_models(low),
        "    def func(self, wavefields, models, dt, h, b, invariants=None, **kwargs):",
        *[f"        {line}" for line in body],
        *substeps,
        "",
    ]
    return "\n".join(lines)
