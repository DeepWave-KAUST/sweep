"""Parameter families of the Biot coefficients and their kernels.

The solvers (FD :class:`sweep.equations.Biot`, SEM ``BiotVSSEM2DDeformed``) take the eight
coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` (:func:`sweep.equations.biot.biot_from_rock`),
so autograd gives the gradient with respect to those. The three families of Morency, Luo &
Tromp (2009) -- the ones SPECFEM2D writes out -- are other coordinates of the same medium:

* ``"primary"``: ``rho, rho_f, m, b, B, C, M, mu_fr`` with ``B = H - 4 mu / 3``;
* ``"density"`` (density-normalised): ``rho, rho_f, phi_c, B / rho, C / rho_f, M / m,
  mu / rho, b`` with ``phi_c = rho_f / m = phi / T``;
* ``"wavespeed"``: ``rho, rho_f, phi_c, c_PI, c_PII, c_S, r_I, b`` -- the inviscid fast-P,
  slow-P and S speeds and ``r_I``, the fluid-to-solid amplitude ratio of the fast P wave.

:func:`biot_to_family` / :func:`biot_from_family` map between them exactly, and
:func:`biot_family_kernels` turns a coefficient gradient into a family's kernels by the
exact chain rule (a vector-Jacobian product through :func:`biot_from_family`). SPECFEM2D's
printed wave-speed kernels are not that chain rule.

With an anisotropic permeability the drag is a tensor ``[[b_xx, b_xz], [b_xz, b_zz]]``
(:func:`biot_drag_tensor`); the coefficient lists then end in its three components, which
every family carries unchanged in place of ``b``.

:func:`biot_fluid` and :func:`biot_elastic` give the coefficients of a fluid and of a dry
elastic solid, the other two media of the SEM ``BiotVSSEM2DDeformed`` (a fluid has no solid
frame, ``rho = 0``; a dry solid no pore fluid, ``M = 0``). The families are those of a
porous medium only.
"""

from __future__ import annotations

import torch

__all__ = ["BIOT_FAMILIES", "biot_drag_tensor", "biot_fluid", "biot_elastic",
           "biot_to_family", "biot_from_family", "biot_family_kernels"]

BIOT_FAMILIES = {
    "primary": ["rho", "rho_f", "m", "b", "B", "C", "M", "mu_fr"],
    "density": ["rho", "rho_f", "phi_c", "B_bar", "C_bar", "M_bar", "mu_bar", "b"],
    "wavespeed": ["rho", "rho_f", "phi_c", "c_PI", "c_PII", "c_S", "r_I", "b"],
}


def biot_drag_tensor(eta, k_xx, k_xz, k_zz):
    """The drag ``eta kappa^-1`` of a permeability tensor ``[[k_xx, k_xz], [k_xz, k_zz]]``
    (m^2, x right and z down -- flip the sign of ``k_xz`` from a z-up convention such as
    SPECFEM2D's), as ``(b_xx, b_xz, b_zz)``. Tensors or floats; gradients flow."""
    det = k_xx * k_zz - k_xz * k_xz
    return eta * k_zz / det, -eta * k_xz / det, eta * k_xx / det


def biot_fluid(k_f, rho_f, drag_tensor=False):
    """The coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` of a fluid of bulk modulus
    ``k_f`` and density ``rho_f``: no solid frame (``rho = 0``), the fluid's own
    compliance and inertia in ``M`` and ``m``, nothing else. ``drag_tensor=True`` ends the
    list in ``(b_xx, b_xz, b_zz)`` instead of ``b``. Tensors or floats."""
    k_f, rho_f = torch.broadcast_tensors(_t(k_f), _t(rho_f))
    z = torch.zeros_like(k_f)
    return [z, z, z, k_f, z, z, rho_f] + [z] * (3 if drag_tensor else 1)


def biot_elastic(vp, vs, rho, drag_tensor=False):
    """The coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` of a dry elastic solid of P and
    S velocities ``vp``, ``vs`` and density ``rho``: ``H = rho vp^2``, ``mu = rho vs^2``,
    no pore fluid (``M = 0``) and nothing else. ``drag_tensor=True`` ends the list in
    ``(b_xx, b_xz, b_zz)`` instead of ``b``. Tensors or floats."""
    vp, vs, rho = torch.broadcast_tensors(_t(vp), _t(vs), _t(rho))
    z = torch.zeros_like(vp)
    return [rho * vp * vp, rho * vs * vs, z, z, rho, z, z] + [z] * (3 if drag_tensor else 1)


def _t(x):
    """A tensor; a Python number becomes a float64 one."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)


def _split(coef):
    coef = list(coef)
    if len(coef) not in (8, 10):
        raise ValueError(f"Biot coefficients are [H, mu, C, M, rho, rho_f, m, b] or end in "
                         f"(b_xx, b_xz, b_zz); got {len(coef)} entries")
    return coef[:7], coef[7:]


def _fast_p_ratio(h, mu, c, bm, rho, rho_f, m):
    g = (rho * c - rho_f * h) / m
    s0 = h - rho * bm / m
    return (s0 + torch.sqrt(s0 * s0 + 4 * (c - rho_f * bm / m) * g)) / (2 * g)


def biot_to_family(family, coef):
    """``coef`` (``[H, mu, C, M, rho, rho_f, m, b]``, or the drag tensor in place of ``b``)
    in the coordinates of ``family`` (see :data:`BIOT_FAMILIES`), as a list of tensors."""
    (h, mu, c, bm, rho, rho_f, m), drag = _split(_t(x) for x in coef)
    big_b = h - 4.0 / 3.0 * mu
    if family == "primary":
        return [rho, rho_f, m, *drag, big_b, c, bm, mu]
    if family == "density":
        return [rho, rho_f, rho_f / m, big_b / rho, c / rho_f, bm / m, mu / rho, *drag]
    if family == "wavespeed":
        a = rho * m - rho_f ** 2
        bq = h * m + bm * rho - 2 * c * rho_f
        disc = torch.sqrt(bq * bq - 4 * a * (h * bm - c * c))
        return [rho, rho_f, rho_f / m, torch.sqrt((bq + disc) / (2 * a)),
                torch.sqrt((bq - disc) / (2 * a)), torch.sqrt(mu / (rho - rho_f ** 2 / m)),
                _fast_p_ratio(h, mu, c, bm, rho, rho_f, m), *drag]
    raise ValueError(f"unknown family {family!r}; choose from {sorted(BIOT_FAMILIES)}")


def biot_from_family(family, params):
    """The inverse of :func:`biot_to_family`: ``[H, mu, C, M, rho, rho_f, m, b...]``."""
    params = [_t(x) for x in params]
    n = len(params)
    if n not in (8, 10):
        raise ValueError(f"a {family!r} parameter list has 8 entries (10 with a drag "
                         f"tensor); got {n}")
    if family == "primary":
        rho, rho_f, m = params[:3]
        drag = params[3:n - 4]
        big_b, c, bm, mufr = params[n - 4:]
        return [big_b + 4.0 / 3.0 * mufr, mufr, c, bm, rho, rho_f, m, *drag]
    drag = params[7:]
    if family == "density":
        rho, rho_f, phic, bbar, cbar, mbar, mubar = params[:7]
        m = rho_f / phic
        return [rho * (bbar + 4.0 / 3.0 * mubar), rho * mubar, rho_f * cbar, m * mbar,
                rho, rho_f, m, *drag]
    if family == "wavespeed":
        rho, rho_f, phic, cp1, cp2, cs, r = params[:7]
        m = rho_f / phic
        a1, a2 = cp1 * cp1, cp2 * cp2
        u1, u2 = rho * r + rho_f, rho_f * r + m
        nn = rho * r * r + 2 * rho_f * r + m
        return [a2 * rho + (a1 - a2) * u1 * u1 / nn, cs * cs * (rho - rho_f * rho_f / m),
                a2 * rho_f + (a1 - a2) * u1 * u2 / nn, a2 * m + (a1 - a2) * u2 * u2 / nn,
                rho, rho_f, m, *drag]
    raise ValueError(f"unknown family {family!r}; choose from {sorted(BIOT_FAMILIES)}")


def biot_family_kernels(family, coef, grads, relative=True):
    """``{name: kernel}`` of ``family`` from the gradient ``grads`` of a misfit with respect
    to ``coef`` (same layout: one tensor per coefficient, e.g. what ``loss.backward()``
    leaves on the models given to the propagator).

    ``relative=True`` gives ``y dJ/dy`` (Morency et al.'s ``K_y``, the misfit change per
    relative change of ``y``), ``False`` the plain ``dJ/dy``. Computed in float64 by the
    exact chain rule through :func:`biot_from_family`."""
    cf = [_t(c).detach().double() for c in coef]
    g = [_t(x).detach().double() for x in grads]
    if len(g) != len(cf):
        raise ValueError(f"{len(g)} gradients for {len(cf)} coefficients")
    y = [t.detach().clone().requires_grad_() for t in biot_to_family(family, cf)]
    back = biot_from_family(family, y)
    jac = torch.autograd.grad(back, y, grad_outputs=g, allow_unused=True)
    names = list(BIOT_FAMILIES[family])
    if len(cf) == 10:
        i = names.index("b")
        names[i:i + 1] = ["b_xx", "b_xz", "b_zz"]
    out = {}
    for name, yy, jj in zip(names, y, jac):
        jj = torch.zeros_like(yy) if jj is None else jj
        out[name] = (yy.detach() * jj) if relative else jj
    return out
