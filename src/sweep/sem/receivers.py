"""Receivers that record a spatial derivative of the fields (PropSEM ``receiver_type``).

A plain receiver interpolates a field at its point through its element's basis; these take the
derivative of that basis instead (:class:`~sweep.sem.propagator.GradientStamp`), so the eager
propagators and the CUDA record kernels record them as they record any field. An equation lists
the components it offers in ``spatial_receiver_types``, ``{name: recipe}``, a recipe being

* ``terms`` -- ``((field, axis, coef), ...)``: the sum of ``coef`` times the field's derivative
  along ``axis`` (0 is x, 1 is y in 3-D, the last is z; ``None`` takes the field itself);
* ``scale`` -- a material factor at the receiver: ``"inv_rho"`` (1/rho: a potential fluid's
  particle motion, ``u = grad(chi) / rho``) or ``"minus_kappa"`` (``-(lambda + 2 mu / 3)``: the
  pressure ``-kappa div u`` in a solid, SPECFEM's, plane strain in 2-D);
* ``time`` -- ``None``, ``"velocity"`` (d/dt), ``"acceleration"``, ``"neg_acceleration"`` or
  ``"integral"`` (the running sum times dt: the displacement a velocity-stress class's leapfrog
  integrates from its half-step velocity).

A fluid-solid equation gives ``{"fluid": recipe, "solid": recipe}``, and every receiver takes the
recipe of the medium its element holds (``None``: zero there).

The components: the strain ``e_ij = (d_i u_j + d_j u_i) / 2`` (``exx``, ``ezz``, ``exz`` in 2-D;
``exx``, ``eyy``, ``ezz``, ``exy``, ``exz``, ``eyz`` in 3-D), the rotation ``curl(u) / 2``
(``rot_y`` in 2-D, ``rot_x``, ``rot_y``, ``rot_z`` in 3-D; in the solver's axes, z down -- in
2-D SPECFEM2D's curl seismogram), their rates (``_t``: DAS records the strain rate) and the
pressure ``p``; in a potential fluid the particle displacement, velocity and acceleration
(``ux``, ``uz``; ``vx``, ``vz``; ``ax``, ``az``).
"""
from __future__ import annotations


def _deformation(u, time):
    """Strain and rotation components of the displacement-like fields ``u`` (2 or 3 names,
    x first), with the time operator ``time``."""
    if len(u) == 2:
        ux, uz = u
        out = {"exx": ((ux, 0, 1.0),), "ezz": ((uz, 1, 1.0),),
               "exz": ((ux, 1, 0.5), (uz, 0, 0.5)), "rot_y": ((ux, 1, 0.5), (uz, 0, -0.5))}
    else:
        ux, uy, uz = u
        out = {"exx": ((ux, 0, 1.0),), "eyy": ((uy, 1, 1.0),), "ezz": ((uz, 2, 1.0),),
               "exy": ((ux, 1, 0.5), (uy, 0, 0.5)), "exz": ((ux, 2, 0.5), (uz, 0, 0.5)),
               "eyz": ((uy, 2, 0.5), (uz, 1, 0.5)),
               "rot_x": ((uz, 1, 0.5), (uy, 2, -0.5)), "rot_y": ((ux, 2, 0.5), (uz, 0, -0.5)),
               "rot_z": ((uy, 0, 0.5), (ux, 1, -0.5))}
    return {name: dict(terms=terms, time=time) for name, terms in out.items()}


def solid_receivers(u, native="displacement"):
    """The strain, rotation and pressure components of a solid whose fields ``u`` are the
    displacement (``native="displacement"``) or, on a velocity-stress class, the velocity."""
    nd = len(u)
    div = tuple((f, k, 1.0) for k, f in enumerate(u))
    if native == "displacement":
        base, rate, p_time = None, "velocity", None
    else:
        base, rate, p_time = "integral", None, "integral"
    out = _deformation(u, base)
    out.update({name + "_t": dict(r, time=rate) for name, r in _deformation(u, base).items()})
    out["p"] = dict(terms=div, scale="minus_kappa", time=p_time)
    assert nd in (2, 3)
    return out


def fluid_receivers(chi, axes=("x", "z")):
    """A potential fluid's particle displacement, velocity and acceleration ``grad(chi) / rho``
    and its time derivatives."""
    out = {}
    for k, a in enumerate(axes):
        term = ((chi, k, 1.0),)
        out["u" + a] = dict(terms=term, scale="inv_rho", time=None)
        out["v" + a] = dict(terms=term, scale="inv_rho", time="velocity")
        out["a" + a] = dict(terms=term, scale="inv_rho", time="acceleration")
    return out


def recipe_fields(recipe):
    """The fields a recipe (or a fluid/solid pair of them) reads."""
    if recipe is None:
        return []
    if "terms" not in recipe:
        return [f for r in recipe.values() for f in recipe_fields(r)]
    return [f for f, _, _ in recipe["terms"]]


def recipe_axes(recipe):
    """The derivative axes a recipe (or a fluid/solid pair) needs."""
    if recipe is None:
        return set()
    if "terms" not in recipe:
        return set().union(*(recipe_axes(r) for r in recipe.values()))
    return {a for _, a, _ in recipe["terms"] if a is not None}
