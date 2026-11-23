"""Write a wave equation as a PDE and get an eager equation class for it.

    from sweep.symbolic import Field, Model, Eq, PDE, laplace

    vp = Model("vp", unit="m/s")
    p = Field("p", source=True, receiver=True)
    MyAcoustic = PDE([Eq(p.dt2, vp**2 * laplace(p))], name="MyAcoustic",
                     free_surface="zero").build(ndim=2)
    solver = PropTorch(MyAcoustic(spatial_order=8, device="cuda"), ..., impl="eager")

``build`` writes the class as Python source (``PDE.source``) and executes it;
``PDE.export`` saves that source as a module. The generated step uses the same
operators and CPML update as the shipped equations, so gradients (autograd),
``torch.compile``, eager boundary saving and ``cpml_strips`` work as they do
for those.

Supported so far:

* second-order-in-time fields whose right-hand side is linear in the fields,
  each term a model-only coefficient times a second derivative along one axis
  (``p.dz2``, ``p.dx2``, ``laplace(p)``) or times a field; any number of
  coupled fields; CPML on every second derivative; a pressure-release free
  surface (flat, per edge in 2-D, or image-method topography);
* first-order staggered-grid systems (``Field(time_order=1, stagger=...)``,
  ``Eq(vx.dt, ...)``): each right-hand side a model-only coefficient times
  first derivatives of the other fields, taken as staggered differences; CPML
  on every derivative; no free surface; plus dissipative terms, a model-only
  coefficient times second derivatives of the field itself
  (``nu * laplace(vx)``, ``kappa * T.dz2``), stepped forward in time (Euler)
  without CPML. A class with such a term refuses boundary saving;
* derivatives of order 3 and higher along one axis (``u.dx4``, ``v.dx3``,
  ``u.deriv("x", 4)``), without CPML: even orders in second-order equations;
  odd orders of the other fields and even orders of the field itself in
  first-order systems;
* ``Coefficient``: model-derived coefficients computed once per run.

Anything else raises when the PDE is constructed.
"""
from __future__ import annotations

import keyword
import linecache
import sys
import types

import sympy

from ._codegen import generate, generate_first_order
from ._expr import Coefficient, Eq, Field, Model, laplace, latex_names
from ._lower import LoweredFirst, lower

__all__ = ["Coefficient", "Eq", "Field", "Model", "PDE", "laplace"]

_FREE_SURFACES = (None, "zero")
_PMLS = ("cpml", "none")
# The generated module's own globals, which a class of the same name would shadow.
_MODULE_GLOBALS = {"torch", "cpml_axis_update", "cpml_update", "FieldSpec", "ModelSpec",
                   "SymbolicSecondOrderEquation", "SymbolicFirstOrderEquation"}


class PDE:
    """A system of wave equations, one per field.

    Args:
        equations: ``Eq`` objects, one per field (``Eq(p.dt2, ...)``, or
            ``Eq(vx.dt, ...)`` for a first-order system, whose fields are
            updated in the order the equations are written).
        name: name of the generated class (and of the registered equation
            with ``build(register=True)``).
        free_surface: ``None`` (no free surface: the propagator's
            ``free_surface=True`` and ``topography=`` raise) or ``"zero"``
            (pressure release: every field is zeroed above the surface;
            second-order fields only).
        pml: ``"cpml"`` (C-PML on every spatial derivative) or ``"none"``
            (no absorbing terms; the waves reflect at the grid edge).
        models: order of the propagator's ``models=``; defaults to the
            order the models were declared in. Must list exactly the models
            the equations use.
    """

    def __init__(self, equations, *, name, free_surface=None, pml="cpml", models=None):
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
            raise ValueError(f"name {name!r} must be a public Python identifier")
        if name in _MODULE_GLOBALS:
            raise ValueError(f"name {name!r} is taken by the generated module itself")
        if free_surface not in _FREE_SURFACES:
            raise ValueError(f"free_surface={free_surface!r}; expected one of {_FREE_SURFACES}")
        if pml not in _PMLS:
            raise ValueError(f"pml={pml!r}; expected one of {_PMLS}")
        equations = list(equations)
        if not equations or not all(isinstance(e, Eq) for e in equations):
            raise TypeError("equations must be a non-empty list of sweep.symbolic.Eq")
        self.equations = equations
        self.name = name
        self.free_surface = free_surface
        self.pml = pml
        self.models = None if models is None else tuple(models)
        # Lower once now, so that a PDE that cannot be built fails here. A
        # y-derivative is still checked against the dimension at build time.
        low = lower(self.equations, 3, self.models)
        if isinstance(low, LoweredFirst) and free_surface is not None:
            raise NotImplementedError(
                f"free_surface={free_surface!r}: a first-order system takes no free surface yet")
        if not any(f.source for f in low.fields):
            raise ValueError("no field has source=True")
        if not any(f.receiver for f in low.fields):
            raise ValueError("no field has receiver=True")

    def lower(self, ndim=2):
        return lower(self.equations, ndim, self.models)

    def source(self, ndim=2):
        """The generated module, as Python source."""
        low = self.lower(ndim)
        doc = [f"{e.lhs} = {e.rhs}" for e in self.equations]
        doc += [f"{c} = {c.expr}  (computed once per run)" for c in low.coefficients]
        doc += [f"free_surface={self.free_surface!r}, pml={self.pml!r}, ndim={ndim}"]
        if isinstance(low, LoweredFirst):
            return generate_first_order(self.name, low, pml=self.pml, doc=doc)
        return generate(self.name, low, free_surface=self.free_surface, pml=self.pml, doc=doc)

    def build(self, ndim=2, register=False):
        """Generate and return the equation class.

        Args:
            ndim: 2 or 3.
            register: also register the class under ``name``, so that
                ``get_equation(name)`` and ``sweep list equations`` find it.
        """
        src = self.source(ndim)
        # A module of its own, importable by name: Dynamo imports the module of
        # the step's globals to guard on them. The name is fixed per (name,
        # ndim), so a rebuild (a notebook cell run again) is a redefinition the
        # registry accepts; the globals it replaces are the same imports.
        modname = f"sweep.symbolic.generated.{self.name}_{ndim}d"
        filename = f"<{modname}>"
        # Tracebacks and Dynamo's messages read the source through linecache.
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        module = types.ModuleType(modname)
        module.__file__ = filename
        sys.modules[modname] = module
        exec(compile(src, filename, "exec"), module.__dict__)
        cls = getattr(module, self.name)
        cls.pde = self
        if register:
            from sweep.equations._registry import register_equation
            register_equation(self.name)(cls)
        return cls

    def export(self, path, ndim=2):
        """Write the generated module to ``path``."""
        with open(path, "w") as fh:
            fh.write(self.source(ndim))

    def latex(self, *, coefficients=True, symbols=None):
        """The system as LaTeX: an ``aligned`` environment, one equation per
        line in the order written, then the definition of each
        ``Coefficient`` the equations use (``coefficients=False`` leaves them
        out). Wrap it in ``$`` for a notebook (what the notebook display does)
        or paste it into a document.

        Args:
            symbols: LaTeX for declared names, ``{"vp": "c", "p": r"\\phi"}``
                (the declared object may be the key), overriding the defaults
                in :data:`sweep.symbolic._expr.LATEX_NAMES`: ``vp`` prints as
                ``v_{p}``, ``rho`` as ``\\rho``, ``lam`` and ``mu`` as the Lamé
                letters, ``vx`` as ``v_{x}``, ``sxz`` as ``\\sigma_{xz}``; a
                Greek name (``epsilon``, ``delta``, ``kappa``) as sympy prints
                it, any other name as written.
        """
        coefs = list(self.lower(3).coefficients) if coefficients else []
        exprs = [e.lhs for e in self.equations] + [e.rhs for e in self.equations]
        names = latex_names(exprs + coefs + [c.expr for c in coefs], symbols)
        rows = [f"{sympy.latex(e.lhs, symbol_names=names, long_frac_ratio=2)} &= {sympy.latex(e.rhs, symbol_names=names, long_frac_ratio=2)}"
                for e in self.equations]
        rows += [f"{sympy.latex(c, symbol_names=names, long_frac_ratio=2)} &= {sympy.latex(c.expr, symbol_names=names, long_frac_ratio=2)}"
                 for c in coefs]
        return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"

    def __repr__(self):
        return f"PDE({self.equations!r}, name={self.name!r})"

    def _repr_latex_(self):
        return f"${self.latex()}$"
