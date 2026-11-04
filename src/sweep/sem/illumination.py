"""Source and receiver illumination on the SEM: the diagonals PropTorch's
``compute_illumination`` gives, for gradient preconditioning and image
compensation.

With ``PropSEM.compute_illumination = True`` a call accumulates, at every GLL
node and over its time steps (summed over its shots),

    source illumination     S = sum_t |a(t)|^2,   a the wavefields' second time
                                                  derivative
    receiver illumination   R = sum_t |lambda(t)|^2,  lambda = M^-1 dL/df the
                                                  adjoint wavefield

where the wavefields ``f`` are an equation's kinematic fields -- those of the run
PropSEM makes: on the potential forms the wavelet itself drives chi, which is then
the pressure (the wavelet is the pressure source), so ``a = chi_tt`` is ``p_tt``
as on the pressure form -- the displacement of the
second-order forms (``a = u_tt``) and the velocity of the velocity-stress ones
(``a = v_t``) -- the quantity each form's gradient is built from, so ``S`` is
the pseudo-Hessian of Shin et al. (2001) and SPECFEM2D's first approximate
Hessian (``APPROXIMATE_HESS_KL``'s ``Hessian1``, ``b_accel . b_accel``). The
adjoint field is ``dL/df`` mapped back through the mass the step inverts.

The two come out shaped like the call's models, as the gradient does: each
element's GLL nodes take their node's value times the element's quadrature
weight ``w |J|`` -- the factor every model value's gradient carries -- and a
grid model takes the transpose of its interpolation to the GLL nodes. Then
``grad / (S + eps)`` is the kernel over the Hessian, with the quadrature and
the interpolation divided out. On a fluid-solid form the two domains'
fields add on the nodes of the interface; their units differ, so compensate
each domain apart.

Neither feeds the gradient: turning them on does not change it.
"""

from __future__ import annotations

from typing import Optional

import torch


def kinematic_fields(eq):
    """``[(field index, its previous level's index or None, inverse mass)]``
    of the wavefields an equation steps, or ``NotImplementedError``."""
    specs = eq.field_specs
    names = [s.name for s in specs]
    if "q_x" in names:
        raise NotImplementedError(
            f"compute_illumination is not defined for {type(eq).__name__}: the "
            "Biot solid and relative-fluid velocities share a 2x2 mass, which has no "
            "per-field adjoint wavefield")
    out = []
    for i, s in enumerate(specs):
        if not s.supports_receiver:
            continue
        prev = names.index(s.name + "_prev") if s.name + "_prev" in names else None
        if prev is None and not s.name.startswith("v_"):
            continue                           # a stress, a pressure: not a wavefield
        out.append((i, prev, _inv_mass(eq, s.name)))
    if not out:
        raise NotImplementedError(
            f"compute_illumination: {type(eq).__name__} has no displacement, potential "
            "or velocity field to illuminate with")
    return out


def _inv_mass(eq, name):
    sub = eq
    if hasattr(eq, "ac") and hasattr(eq, "el"):          # fluid-solid
        sub = eq.ac if name == "chi" else eq.el
    inv = getattr(sub, "_inv_M", None)
    if inv is None:
        raise NotImplementedError(
            f"compute_illumination: {type(eq).__name__} exposes no inverse mass for "
            f"its field {name!r}")
    return inv


def element_weights(mesh) -> torch.Tensor:
    """``w |J|`` at every element GLL node, ``(E, ngll[, ngll[, ngll]])``."""
    wj = getattr(mesh, "w_jac", None)
    if wj is not None:
        return wj
    w = mesh.w_ref[:, None] * mesh.w_ref[None, :]              # Quad4: constant |J|
    return (w * mesh.jacobian_det).expand(mesh.ibool.shape)


class IlluminationSink:
    """The two accumulators of one call, on the global nodes. ``after_step``
    takes each step's (old, new) state: it adds the source term and hooks the
    new wavefields, whose gradient in the backward is the adjoint."""

    def __init__(self, eq, dt: float, like: torch.Tensor):
        self.fields = kinematic_fields(eq)
        self.dt = float(dt)
        n = eq.mesh.n_global            # sums of squares over every step: fp64
        self.src = torch.zeros(n, dtype=torch.float64, device=like.device)
        self.rec = torch.zeros(n, dtype=torch.float64, device=like.device)
        self.rec_seen = False

    def after_step(self, old, new):
        dt = self.dt
        with torch.no_grad():
            for i, prev, _ in self.fields:
                if prev is None:
                    a = (new[i] - old[i]) / dt
                else:
                    a = (new[i] - 2.0 * old[i] + old[prev]) / (dt * dt)
                self.src += a.detach().double() ** 2
        for i, _, inv in self.fields:
            t = new[i]
            if t.requires_grad:
                t.register_hook(self._hook(inv))

    def _hook(self, inv):
        def add(g):
            with torch.no_grad():
                self.rec += (g.detach().double() * inv.double()) ** 2
            self.rec_seen = True
            return None
        return add


def to_model(acc: torch.Tensor, prop, gll_native: bool) -> torch.Tensor:
    """A node accumulator in the call's model shape (see the module doc)."""
    mesh = prop.mesh
    h = acc[mesh.ibool] * element_weights(mesh).to(acc.dtype)
    if gll_native:
        return h
    grid = torch.zeros(prop.shape, dtype=acc.dtype, device=acc.device, requires_grad=True)
    with torch.enable_grad():
        out = prop._interp_to_gll(grid)
        (g,) = torch.autograd.grad(out, grid, grad_outputs=h.reshape(out.shape))
    return g


class _Part:
    """One CUDA call's accumulators: the forward's, then the backward's."""

    def __init__(self, src):
        self.src, self.rec, self.rec_seen = src, None, False


class Illumination:
    """What ``PropSEM.source_illumination`` / ``receiver_illumination`` read:
    the call's accumulators, mapped to its model shape on first read."""

    def __init__(self, prop, gll_native: bool):
        self.prop, self.gll_native = prop, gll_native
        self.sinks: list = []
        self._src: Optional[torch.Tensor] = None
        self._rec: Optional[torch.Tensor] = None

    def sink(self, eq, dt, like):
        s = IlluminationSink(eq, dt, like)
        self.sinks.append(s)
        return s

    def add_nodes(self, src):
        """A node accumulator filled elsewhere (a CUDA driver's forward);
        returns the part whose ``rec`` its backward fills."""
        part = _Part(src)
        self.sinks.append(part)
        return part

    def source(self):
        if self._src is None and self.sinks:
            acc = sum(s.src for s in self.sinks if s.src is not None)
            self._src = to_model(acc, self.prop, self.gll_native)
        return self._src

    def receiver(self):
        if not any(s.rec_seen for s in self.sinks):
            return None
        if self._rec is None:
            acc = sum(s.rec for s in self.sinks if s.rec is not None)
            self._rec = to_model(acc, self.prop, self.gll_native)
        return self._rec
