"""Write the eager equation class of a lowered PDE as Python source.

The step is written the way the hand-written equations write theirs
(``sweep.equations.acoustic`` / ``acoustic3d``): the same operators, the same
CPML call, the same order of operations, so a PDE that restates a shipped
equation reproduces it bit for bit.
"""
from __future__ import annotations

import sympy
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

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


def _coef_times(coef, rest):
    """``coef * dt**2 * rest`` with ``coef`` printed at multiplication precedence."""
    if coef == 1:
        return f"dt**2 * {rest}"
    return f"{_printer.parenthesize(coef, PRECEDENCE['Mul'])} * dt**2 * {rest}"


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
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f"name(s) {', '.join(clash)} used twice (fields, models, aliases "
                         "and the generated _prev / psi / zeta fields share one namespace)")

    models = []
    for m in low.models:
        aliases = f", aliases={tuple(m.aliases)!r}" if m.aliases else ""
        models.append(f"ModelSpec({_q(str(m))}{aliases}, description={_q(m.description)}, unit={_q(m.unit)})")

    src_fields = [str(f) for f in low.fields if f.source]
    rec_fields = [str(f) for f in low.fields if f.receiver]

    # ---- the step -------------------------------------------------------------
    sp = ", ".join(f"_h{ax}" for ax in axes)
    body = []
    state = []
    for f in low.fields:
        state += [str(f), nm.prev(f)]
    if use_cpml:
        state += [f"_{n(f, ax)}" for n in (nm.psi, nm.zeta) for f, ax in nm.cpml]
    body.append(f"{', '.join(state)} = wavefields")
    if low.models:
        mods = ", ".join(str(m) for m in low.models)
        body.append(f"({mods},) = models" if len(low.models) == 1 else f"{mods} = models")
    if low.d2:
        body.append(f"{sp} = self._spacings_{ndim}d(h)")
        if use_cpml:
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
        laps = ", ".join(f"_lap{ax}_{f}" for ax in axes)
        body.append(f"{laps} = self.separable_d2_{ndim}d({f}, self.laplace_kernels, {sp})")
        if use_cpml:
            body.append(f"_src_{f} = ({f}, self.laplace_kernels, ({sp}))")
    if use_cpml:
        for f, ax in low.d2:
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
    for f, ax in low.d2:
        if use_cpml:
            i = axis_index[ax]
            psi, zeta = nm.psi(f, ax), nm.zeta(f, ax)
            body.append(f"_d2{ax}_{f}, _{psi}_n, _{zeta}_n = cpml_axis_update(")
            body.append(f"    _lap{ax}_{f}, _d{ax}_{f}, _{psi}, _{zeta}, _a{ax}, _b{ax}, _db{ax}, h, {i},")
            body.append(f"    self.gradient, self.grad_kernels, _src_{f})")
        body += [f"{sums[id(g)]} += {d2_of(f, ax)}" for g in multi if g.field == f and ax in g.axes]

    for up in low.updates:
        f = up.field
        terms = [_coef_times(g.coef, sums[id(g)]) for g in up.groups]
        for c, g in up.local:
            terms.append(_coef_times(c, str(g)))
        rhs = f"2 * {f} - {nm.prev(f)}" + "".join(f" + {t}" for t in terms)
        body.append(f"_{f}_next = {rhs}")

    out = []
    for f in low.fields:
        out += [f"_{f}_next", str(f)]
    if use_cpml:
        out += [f"_{n(f, ax)}_n" for n in (nm.psi, nm.zeta) for f, ax in nm.cpml]
    body.append(f"return self._free_surface_out(({', '.join(out)},))")

    flags = []
    if free_surface is not None:
        flags.append(f"supports_per_edge_free_surface = {ndim == 2}")
        flags.append("supports_image_topography = True")

    text = "\n".join(f"    {line}" for line in doc).replace("\\", "\\\\").replace('"""', "'''")
    lines = [
        f'"""{name}: generated by sweep.symbolic from\n\n{text}\n"""',
        "import torch  # noqa: F401  (elementary functions in coefficients)",
        "",
        "from sweep.equations._cpml import cpml_axis_update  # noqa: F401",
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
        '    default_pml_type = "cpmlr"',
        *[f"    {fl}" for fl in flags],
        "",
        "    MODEL_SPECS = (",
        *[f"        {m}," for m in models],
        "    )",
        "    FIELD_SPECS = (",
        *[f"        {s}," for s in specs],
        "    )",
        "",
        "    def func(self, wavefields, models, dt, h, b, **kwargs):",
        *[f"        {line}" for line in body],
        "",
    ]
    return "\n".join(lines)
