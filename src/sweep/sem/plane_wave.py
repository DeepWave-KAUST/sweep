"""Plane-wave incidence on the 2-D displacement SEM, by the domain reduction method.

SPECFEM2D's ``initialfield``: a plane P or SV wave, or a Rayleigh wave, in a uniform
half-space -- with the reflected and converted waves of its free surface -- or, under an
absorbing top, in a whole space; and on the SH class a plane SH wave. SPECFEM2D starts the run with that field in the mesh and
keeps feeding it through its Stacey edges (Bielak's conditions), which the code ties to the
Stacey boundary; here it enters through a band of elements one element thick instead, the
domain reduction method (Bielak, Loukakis, Hisada & Yoshimura, 2003). With ``u0`` the plane
wave of the uniform background, ``b`` the nodes of the band's inner boundary and ``e`` its
other nodes, the band's elements load

    P_b = -K_be u0_e,        P_e = K_eb u0_b

(the mass is diagonal, so no mass term enters). Inside the inner boundary the run then
carries the whole field -- the plane wave and what the medium there scatters -- and from
the band out the scattered field alone, which the C-PML absorbs like any other outgoing
wave. The medium inside is free (layers, a basin, attenuation, a topography); outside, the
band, any margin and the absorbing layer are the uniform background, with a flat free
surface, and :func:`band_injection` refuses anything else.

``u0`` is computed in the frequency domain from the wavelet's samples, so a delay is the
band-limited one: the propagating waves are delayed copies of the wavelet; an SV wave past
its critical angle reflects an evanescent P wave, and a Rayleigh wave is two evanescent
waves, which decay with depth at a rate set by the frequency. One traction-free solve covers
every case, below the critical angle, at it and past it.

The loads are a fixed source: computed from the background, they carry no gradient to the
model (the wavelet's own gradient they do carry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .propagator import SEMPropagator

WAVES = ("P", "SV", "SH", "Rayleigh")

# the incident field on the band at t = 0, over its peak, beyond which the run would start
# with a jump (the wave is already there): refused
_QUIET = 1e-3


@dataclass(frozen=True)
class PlaneWave:
    """A plane wave for :class:`~sweep.sem.PropSEM`, passed in place of the source
    coordinates: ``prop(wavelet, PlaneWave("SV", 30.0), receivers, models=...)``.

    ``wave`` is ``"P"``, ``"SV"``, ``"SH"`` or ``"Rayleigh"`` and ``angle`` the incidence
    angle from the vertical in degrees, positive for a wave travelling towards +x (incident
    from the left), ``|angle| < 90``; a Rayleigh wave travels along the surface, ``angle`` 90
    (towards +x) or -90. P, SV and Rayleigh waves run on the P-SV classes, an SH wave on
    :class:`~sweep.sem.ElasticSHSEM2DDeformed`.

    The wavelet is the incident displacement at ``origin``: along the propagation direction
    for a P wave; for an SV wave along ``(cos a, sin a)`` (x, z down) when it travels towards
    +x, the mirror image of that when it travels towards -x; for an SH wave ``u_y``; for a
    Rayleigh wave the horizontal displacement on the surface. ``origin`` (model grid units, x first, as the
    sources) defaults to the point of the injection band the wave reaches first, so a
    wavelet that starts at rest starts the run at rest.

    ``box`` is the band's inner boundary, ``((x0, z0), (x1, z1))`` in model grid units, on
    element edges; the band is the ring of elements around it. Default: one element in from
    the model's edges, from the top down when the top is a free surface.
    """
    wave: str = "P"
    angle: float = 0.0
    origin: Optional[tuple] = None
    box: Optional[tuple] = None

    def __post_init__(self):
        if self.wave not in WAVES:
            raise ValueError(f"wave must be one of {WAVES}, got {self.wave!r}")
        a = float(self.angle)
        if self.wave == "Rayleigh":
            if abs(a) != 90.0:
                raise ValueError(
                    "a Rayleigh wave travels along the surface: angle = 90 (towards +x) or "
                    f"-90, got {self.angle!r}")
        elif not abs(a) < 90.0:
            raise ValueError(f"a plane {self.wave} wave needs |angle| < 90, got {self.angle!r}")

    @property
    def direction(self) -> tuple:
        """The incident wave's propagation direction ``(n_x, n_z)``, z down."""
        a = math.radians(float(self.angle))
        if self.wave == "Rayleigh":
            return (math.copysign(1.0, a), 0.0)
        return (math.sin(a), -math.cos(a))


# --------------------------------------------------------------------------- #
# The plane wave of a uniform background
# --------------------------------------------------------------------------- #
def rayleigh_speed(vp: float, vs: float) -> float:
    """The Rayleigh wave speed of a uniform half-space: the root ``c < vs`` of
    ``(2 - c^2/vs^2)^2 = 4 sqrt(1 - c^2/vp^2) sqrt(1 - c^2/vs^2)``."""
    r = (vs / vp) ** 2

    def h(x):                         # the dispersion function over x = c^2/vs^2, < 0 at 0+
        return ((2.0 - x) ** 2 - 4.0 * math.sqrt(1.0 - r * x) * math.sqrt(1.0 - x)) / x

    lo, hi = 1e-12, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if h(mid) < 0.0 else (lo, mid)
    return vs * math.sqrt(0.5 * (lo + hi))


def _plane_wave_terms(wave: str, angle: float, vp: Optional[float], vs: float, rho: float,
                      surface: bool):
    """``(p, terms)``: the horizontal slowness and the plane waves ``(a, q, reflected)`` of
    the field towards +x -- each with its displacement vector ``a = (a_x, a_z)`` (``(a_y,)``
    for SH; complex: a coefficient times a polarization) and vertical slowness ``q`` (z down),
    both for ``omega > 0``; an evanescent wave has ``q = -i kappa``. The reflected waves, for
    a free surface, from the traction-free condition there."""
    if wave == "SH":                  # sigma_yz = mu du_y/dz = 0: reflected whole, same sign
        p = math.sin(math.radians(abs(angle))) / vs
        q = math.sqrt(1.0 / (vs * vs) - p * p)
        return p, [((1.0,), complex(-q, 0.0), False)] + (
            [((1.0,), complex(q, 0.0), True)] if surface else [])
    lam, mu = rho * (vp * vp - 2.0 * vs * vs), rho * vs * vs

    def down(c, p):                   # downgoing vertical slowness, or -i kappa past it
        s = 1.0 / (c * c) - p * p
        return complex(math.sqrt(s), 0.0) if s >= 0.0 else complex(0.0, -math.sqrt(-s))

    pol = {"P": lambda p, q: (vp * p, vp * q), "S": lambda p, q: (-vs * q, vs * p)}

    def traction(a, p, q):            # (sigma_xz, sigma_zz) per unit amplitude, / (-i omega)
        return (mu * (q * a[0] + p * a[1]), lam * p * a[0] + (lam + 2.0 * mu) * q * a[1])

    scale = lambda a, c: (a[0] * c, a[1] * c)
    if wave == "Rayleigh":
        p = 1.0 / rayleigh_speed(vp, vs)
        qP, qS = down(vp, p), down(vs, p)
        aP, aS = pol["P"](p, qP), pol["S"](p, qS)
        RP = -traction(aS, p, qS)[0] / traction(aP, p, qP)[0]          # with R_S = 1
        X = RP * aP[0] + aS[0]                       # the surface's horizontal displacement
        return p, [(scale(aP, RP / X), qP, True), (scale(aS, 1.0 / X), qS, True)]
    c_inc = vp if wave == "P" else vs
    p = math.sin(math.radians(abs(angle))) / c_inc
    q_inc = complex(-math.sqrt(1.0 / (c_inc * c_inc) - p * p), 0.0)      # upgoing
    a_inc = pol["P" if wave == "P" else "S"](p, q_inc)
    terms = [(a_inc, q_inc, False)]
    if surface:
        qP, qS = down(vp, p), down(vs, p)
        aP, aS = pol["P"](p, qP), pol["S"](p, qS)
        tP, tS, ti = traction(aP, p, qP), traction(aS, p, qS), traction(a_inc, p, q_inc)
        det = tP[0] * tS[1] - tS[0] * tP[1]
        RP = (tS[0] * ti[1] - ti[0] * tS[1]) / det
        RS = (ti[0] * tP[1] - tP[0] * ti[1]) / det
        terms += [(scale(aP, RP), qP, True), (scale(aS, RS), qS, True)]
    return p, terms


class FreeField:
    """The plane wave ``u0`` of a uniform background (``vp``, ``vs``, ``rho``; ``vp`` unused
    by an SH wave) with its free surface at depth ``surface`` (metres, z down; ``None``: a
    whole space), the wavelet its incident displacement at ``origin = (x0, z0)`` (on the
    surface for a Rayleigh wave)."""

    def __init__(self, wave: PlaneWave, vp: Optional[float], vs: float, rho: float,
                 surface: Optional[float], origin: tuple):
        if wave.wave == "Rayleigh" and surface is None:
            raise ValueError("a Rayleigh wave needs a free surface: the top is absorbing")
        if wave.wave != "P" and not vs > 0.0:
            raise ValueError(f"an {wave.wave} wave needs a solid: vs = 0 in the background")
        self.wave, self.vs, self.rho = wave, float(vs), float(rho)
        self.vp = None if vp is None else float(vp)
        self.surface = None if surface is None else float(surface)
        self.origin = (float(origin[0]), float(origin[1]))
        if wave.wave == "Rayleigh":
            self.origin = (self.origin[0], self.surface)
        self.p, self.terms = _plane_wave_terms(wave.wave, wave.angle, self.vp, self.vs,
                                               self.rho, surface is not None)
        # the waves past their critical angle (and a Rayleigh wave's): a pi/2 phase in
        # their coefficients, which has no zero-frequency limit
        self.evanescent = any(q.imag != 0.0 for _, q, _ in self.terms)
        self.ncomp = len(self.terms[0][0])        # (u_x, u_z), or SH's u_y

    def _delays(self, x, z):
        """Per term ``(delay, decay)``, ``(n,)`` each: the wave is the wavelet ``delay``
        later, times ``exp(-omega decay)`` (evanescent)."""
        x0, z0 = self.origin
        if self.wave.angle < 0.0:                      # the mirror image of the +x field
            x = 2.0 * x0 - x
        out = []
        for _, q, refl in self.terms:
            if not refl:
                out.append((self.p * (x - x0) + q.real * (z - z0), None))
                continue
            at_top = self.p * (x - x0) + self.terms[0][1].real * (self.surface - z0) \
                if self.wave.wave != "Rayleigh" else self.p * (x - x0)
            dz = z - self.surface
            if q.imag == 0.0:
                out.append((at_top + q.real * dz, None))
            else:
                out.append((at_top, -q.imag * dz))
        return out

    def displacement(self, x, z, wavelet: torch.Tensor, dt: float, nt: int,
                     t0: float = 0.0) -> torch.Tensor:
        """``u0`` ``(n, 2, nt)`` (``(n, 1, nt)`` for SH; float64) at the points ``x, z``
        (metres, z down) at ``t = t0 + k dt``; ``wavelet`` is sampled at ``k dt`` and taken
        as zero outside its record."""
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1).cpu()
        z = torch.as_tensor(z, dtype=torch.float64).reshape(-1).cpu()
        w = wavelet.to(torch.float64)
        dev = w.device
        if self.surface is not None and self.evanescent and bool(
                (z < self.surface - 1e-9 * max(1.0, abs(self.surface))).any()):
            raise ValueError("an evanescent wave is not defined above the free surface")
        delays = self._delays(x, z)
        span = max(float(d.abs().max()) for d, _ in delays) if x.numel() else 0.0
        need = w.shape[-1] + nt + 2 * math.ceil((span + abs(t0)) / dt) + 64
        N = 1 << max(1, (2 * need - 1).bit_length())
        W = torch.fft.rfft(w, n=N)
        if self.evanescent:
            dc = float(W[0].abs()) if W.dim() == 1 else float(W[..., 0].abs().max())
            if dc > 1e-6 * float(w.abs().sum()):
                raise ValueError(
                    f"a {'Rayleigh wave' if self.wave.wave == 'Rayleigh' else 'wave past its critical angle'} "
                    "has a pi/2 phase, undefined at zero frequency: the wavelet must have "
                    "zero mean")
        omega = (2.0 * math.pi * torch.fft.rfftfreq(N, d=dt, dtype=torch.float64)).to(dev)
        # the mirror image flips u_x; u_y and u_z keep their sign
        sign = -1.0 if self.wave.angle < 0.0 and self.ncomp == 2 else 1.0
        chunk = max(1, (1 << 22) // omega.numel())         # ~64 MB per (points, freq) array
        out = torch.empty(x.numel(), self.ncomp, nt, dtype=w.dtype, device=dev)
        for s in range(0, x.numel(), chunk):
            sl = slice(s, s + chunk)
            acc = [0.0] * self.ncomp
            for (a, _, _), (d, k) in zip(self.terms, delays):
                d = d[sl].to(dev)
                ph = -omega[None, :] * (d[:, None] - t0)
                e = torch.polar(torch.ones_like(ph) if k is None else
                                torch.exp(-omega[None, :] * k[sl].to(dev)[:, None]), ph)
                for c in range(self.ncomp):
                    acc[c] = acc[c] + complex(a[c]) * e
            for c in range(self.ncomp):
                u = torch.fft.irfft(acc[c] * W, n=N)[..., :nt]
                out[sl, c] = sign * u if c == 0 else u
        return out


# --------------------------------------------------------------------------- #
# The band and its loads
# --------------------------------------------------------------------------- #
def _check_uniform(name, v, rows):
    vals = v.detach()[rows]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo > 1e-6 * max(abs(hi), abs(lo), 1e-30):
        raise ValueError(
            f"plane-wave incidence needs the uniform background outside the injection "
            f"boundary: {name} ranges over [{lo:.6g}, {hi:.6g}] there (the band, any margin "
            "and the absorbing layer); move the box out, or the heterogeneity in")
    return float(vals.reshape(-1)[0])


@dataclass
class Band:
    """The injection band on one equation's mesh: ``interior`` and ``elements`` the elements
    inside its inner boundary and of the band, ``nodes`` the band's nodes (global ids),
    ``on_boundary`` which of them lie on the inner boundary, ``free`` the background's plane
    wave."""
    interior: torch.Tensor
    elements: torch.Tensor
    nodes: torch.Tensor
    on_boundary: torch.Tensor
    free: FreeField


@dataclass
class BandInjection:
    """The plane wave's loads on a :class:`Band`: ``loads`` ``(n, k, nt)``, the forces at
    the band's nodes (``k`` = 2, or SH's 1)."""
    band: Band
    loads: torch.Tensor

    def propagator(self, eq, dt, nt, receivers, receiver_type=None):
        """The :class:`SEMPropagator` that runs it: one unit force per node and
        component, each with its own time series (the per-source scatter)."""
        nodes = self.band.nodes.to(self.loads.device)
        k = self.loads.shape[1]
        one = torch.ones(nodes.numel(), 1)
        zero = torch.zeros_like(one)
        return NodalLoads(eq=eq, dt=dt, nt=nt, sources=[], receivers=receivers,
                          source_kinds=[], source_payloads=[], receiver_type=receiver_type,
                          nodal_G=torch.cat([nodes] * k).reshape(-1, 1),
                          nodal_W=tuple(torch.cat([one if j == c else zero for j in range(k)])
                                        for c in range(k)))

    def wavelet(self, dtype) -> torch.Tensor:
        """The loads as the per-source wavelet ``(k n, nt)``: the x components, then z (the
        one component, SH's)."""
        return torch.cat(list(self.loads.unbind(1))).to(dtype)


@dataclass
class NodalLoads(SEMPropagator):
    """A :class:`SEMPropagator` whose sources are unit forces at mesh nodes, one time series
    each: ``nodal_G`` ``(P, 1)`` their nodes, ``nodal_W`` per source field ``(P, 1)`` their
    weights; the wavelet is ``(P, nt)``."""
    nodal_G: Optional[torch.Tensor] = None
    nodal_W: tuple = ()

    def _check_wavelet(self, w):
        P = self.nodal_G.shape[0]
        if w.dim() != 2 or tuple(w.shape) != (P, self.nt):
            raise ValueError(f"the nodal loads take a ({P}, {self.nt}) series; "
                             f"got {tuple(w.shape)}")
        return True

    def _build_per_source_patterns(self, device, dtype):
        return (self.nodal_G.to(device),
                [w.to(device=device, dtype=dtype) for w in self.nodal_W])


def _elements(nex, ez, ex):
    """Element ids of the row/column ranges' product (rows ``ez``, columns ``ex``)."""
    return (torch.as_tensor(ez)[:, None] * nex + torch.as_tensor(ex)[None, :]).reshape(-1)


def _band_residual(eq, elems, U):
    """``K U`` element-local ``(E_s, ngll, ngll, k, T)`` on the elements ``elems``, from
    the element-gathered displacement ``U`` ``(E_s, ngll, ngll, k, T)``: the bare stiffness
    of :class:`~sweep.sem.ElasticSEM2DDeformed` (``_grads`` and ``_divergence_e``; ``k = 2``)
    or of :class:`~sweep.sem.ElasticSHSEM2DDeformed` (``stiffness_apply``; ``k = 1``) with a
    trailing time axis, on the detached moduli."""
    D = eq._D.detach().to(U.dtype)
    if U.shape[-2] == 1:              # SH: the flux mu grad(u_y)
        g = {k: getattr(eq, k).detach()[elems].to(U.dtype)[..., None, None]
             for k in ("_ixx", "_ixz", "_iex", "_iez", "_w_J", "_inv_rho")}
        dU_dxi = torch.einsum("bg,eagkt->eabkt", D, U)
        dU_deta = torch.einsum("ag,egbkt->eabkt", D, U)
        q_x = g["_inv_rho"] * (g["_ixx"] * dU_dxi + g["_iex"] * dU_deta)
        q_z = g["_inv_rho"] * (g["_ixz"] * dU_dxi + g["_iez"] * dU_deta)
        F_beta = g["_w_J"] * (q_x * g["_ixx"] + q_z * g["_ixz"])
        F_alpha = g["_w_J"] * (q_x * g["_iex"] + q_z * g["_iez"])
        return (torch.einsum("db,eadkt->eabkt", D, F_beta)
                + torch.einsum("ga,egbkt->eabkt", D, F_alpha))
    g = {k: getattr(eq, k).detach()[elems].to(U.dtype)[..., None]
         for k in ("_ixx", "_ixz", "_iex", "_iez", "_w_J", "_lam", "_mu", "_lam2mu")}
    dU_dxi = torch.einsum("bg,eagkt->eabkt", D, U)
    dU_deta = torch.einsum("ag,egbkt->eabkt", D, U)
    dU_dx = g["_ixx"][..., None, :] * dU_dxi + g["_iex"][..., None, :] * dU_deta
    dU_dz = g["_ixz"][..., None, :] * dU_dxi + g["_iez"][..., None, :] * dU_deta
    dux_dx, duz_dx = dU_dx[..., 0, :], dU_dx[..., 1, :]
    dux_dz, duz_dz = dU_dz[..., 0, :], dU_dz[..., 1, :]
    lam, mu, lam2mu = g["_lam"], g["_mu"], g["_lam2mu"]
    s_xx = lam2mu * dux_dx + lam * duz_dz
    s_zz = lam * dux_dx + lam2mu * duz_dz
    s_xz = mu * (dux_dz + duz_dx)
    w, ixx, ixz, iex, iez = g["_w_J"], g["_ixx"], g["_ixz"], g["_iex"], g["_iez"]
    F_beta = torch.stack([w * (s_xx * ixx + s_xz * ixz), w * (s_xz * ixx + s_zz * ixz)], -2)
    F_alpha = torch.stack([w * (s_xx * iex + s_xz * iez), w * (s_xz * iex + s_zz * iez)], -2)
    return (torch.einsum("db,eadkt->eabkt", D, F_beta)
            + torch.einsum("ga,egbkt->eabkt", D, F_alpha))


def injection_band(eq, wave: PlaneWave, *, rows: tuple, cols: tuple, nex: int,
                   top_free: bool, origin=None) -> Band:
    """The band of ``wave`` on ``eq``'s mesh (``nex`` element columns, element ``e = ez *
    nex + ex``): ``rows = (ez0, ez1)`` and ``cols = (ex0, ex1)`` the elements inside the
    inner boundary, the band the ring of elements around them (none above a box that reaches
    the top); ``top_free``: the mesh top is a free surface; ``origin`` the wavelet's point
    (metres, z down), default the band's node the incident wave reaches first. Refuses a
    background that is not uniform, elastic and, under a free top, flat-topped outside the
    inner boundary, and a band in the absorbing layer."""
    from .equations.elastic2d_deformed import ElasticSEM2DDeformed
    from .equations.elastic_sh2d import ElasticSHSEM2DDeformed
    sh = isinstance(eq, ElasticSHSEM2DDeformed)
    if (not (sh or isinstance(eq, ElasticSEM2DDeformed))
            or getattr(eq, "domain", None) is not None):
        raise NotImplementedError(
            f"plane-wave incidence is wired for the 2-D displacement classes "
            f"(ElasticSEM2DDeformed, ViscoElasticSEM2DDeformed, ElasticSHSEM2DDeformed); "
            f"got {type(eq).__name__}")
    if sh != (wave.wave == "SH"):
        raise ValueError(f"an {'SH' if sh else 'in-plane'} class takes "
                         f"{'an SH wave' if sh else 'P, SV and Rayleigh waves'}; got "
                         f"{wave.wave} on {type(eq).__name__}")
    mesh = eq.mesh
    E, nez = mesh.n_elements, mesh.n_elements // nex
    (ez0, ez1), (ex0, ex1) = rows, cols
    if not (0 <= ez0 < ez1 and 0 < ex0 < ex1 < nex and ez1 < nez):
        raise ValueError(f"the box (element rows {rows}, columns {cols}) leaves no band "
                         f"inside the {nez} x {nex} element mesh")
    if ez0 == 0 and not top_free:
        raise ValueError("the box reaches the top, which is not a free surface: stop it an "
                         "element below the absorbing layer")
    interior = _elements(nex, torch.arange(ez0, ez1), torch.arange(ex0, ex1))
    ring_rows = torch.arange(ez0 - (1 if ez0 > 0 else 0), ez1 + 1)
    whole = _elements(nex, ring_rows, torch.arange(ex0 - 1, ex1 + 1))
    inside = torch.zeros(E, dtype=torch.bool)
    inside[interior] = True
    band = whole[~inside[whole]]
    if eq.cpml is not None:
        # an element of the layer has a damped node off its edges
        if sh:                        # the potential form's damping, per node
            cp = eq.cpml
            d_x, d_z = ((cp.sigma_x, cp.sigma_z) if cp.sigma_x is not None else (cp.b_x, cp.b_z))
            damped = ((d_x != 0) | (d_z != 0))[mesh.ibool].cpu()
        else:
            damped = eq.cpml.node_in_pml.cpu()
        hit = damped[:, 1:-1, 1:-1].reshape(E, -1).any(1)
        if bool(hit[band].any()):
            raise ValueError("the injection band reaches into the absorbing layer: move the "
                             "box in by an element")

    outside = (~inside).nonzero().reshape(-1)
    if sh:                            # built as the potential form: vp = vs, rho = 1 / mu
        vp = None
        vs = _check_uniform("vs", eq.vp_e.reshape(E, -1).cpu(), outside)
        rho = 1.0 / (_check_uniform("rho", eq.rho_e.reshape(E, -1).cpu(), outside) * vs * vs)
    else:
        vp = _check_uniform("vp", eq.vp_e.reshape(E, -1).cpu(), outside)
        vs = _check_uniform("vs", eq.vs_e.reshape(E, -1).cpu(), outside)
        rho = _check_uniform("rho", eq.rho_e.reshape(E, -1).cpu(), outside)
    for w in (*getattr(eq, "_w1", ()), *getattr(eq, "_w2", ())):
        if bool((w.detach().reshape(E, -1).cpu()[outside] != 0.0).any()):
            raise ValueError("plane-wave incidence needs an elastic background: Q must be "
                             "infinite outside the injection boundary")
    surface = None
    if top_free:
        top = _elements(nex, torch.tensor([0]), torch.arange(nex))
        top = top[~inside[top]]
        z_top = mesh.z_e.detach().cpu().double()[top][:, 0, :]
        surface = float(z_top.reshape(-1)[0])
        if float((z_top - surface).abs().max()) > 1e-5 * float(mesh.z_e.detach().abs().max() + 1):
            raise ValueError("plane-wave incidence needs a flat free surface outside the "
                             "injection boundary (the plane wave's half-space); the "
                             "topography there is not flat")

    if wave.wave == "Rayleigh" and surface is None:
        raise ValueError("a Rayleigh wave needs a free surface: the top is absorbing")
    ib = mesh.ibool.cpu()
    nodes = torch.unique(ib[band])
    in_nodes = torch.zeros(mesh.n_global, dtype=torch.bool)
    in_nodes[ib[interior].reshape(-1)] = True
    xg, zg = mesh.x_global.detach().cpu().double(), mesh.z_global.detach().cpu().double()
    if origin is None:
        nx_, nz_ = wave.direction
        first = nodes[torch.argmin(nx_ * xg[nodes] + nz_ * zg[nodes])]
        # a Rayleigh wave's reference is on the surface above that node
        origin = (float(xg[first]), surface if wave.wave == "Rayleigh" else float(zg[first]))
    return Band(interior=interior, elements=band, nodes=nodes, on_boundary=in_nodes[nodes],
                free=FreeField(wave, vp, vs, rho, surface, origin))


def band_injection(eq, band: Band, wavelet: torch.Tensor, dt: float, nt: int,
                   chunk: int = 128) -> BandInjection:
    """The loads of ``band.free`` on ``band`` for the wavelet sampled at ``k dt``, over
    ``nt`` steps, in float64 on the equation's device."""
    mesh, free, nodes = eq.mesh, band.free, band.nodes
    xg, zg = mesh.x_global.detach().cpu().double(), mesh.z_global.detach().cpu().double()
    dev = eq.vp_e.device
    w64 = wavelet.to(device=dev, dtype=torch.float64)
    u0 = free.displacement(xg[nodes], zg[nodes], w64, dt, nt)       # (n, k, nt)
    # a run starts at rest: the wave must not be on the band yet
    start = free.displacement(xg[nodes], zg[nodes], w64.detach(), dt, 2, t0=-dt).abs().amax()
    peak = u0.detach().abs().amax()
    if float(start) > _QUIET * float(peak):
        why = (" -- an evanescent wave's motion carries the wavelet's Hilbert transform, "
               "whose tails run ahead of it: give it a few periods more" if free.evanescent
               else "")
        raise ValueError(
            f"the incident wave is already on the injection band at t = 0 "
            f"(|u0| {float(start):.3g} there, {float(peak):.3g} at its peak): delay the "
            "wavelet, or leave origin to its default (where the wave reaches the band "
            f"first){why}")

    loc = torch.full((mesh.n_global,), -1, dtype=torch.long)
    loc[nodes] = torch.arange(nodes.numel())
    elems = band.elements.to(dev)
    lb = loc.to(dev)[mesh.ibool[elems]]                        # (E_b, ngll, ngll) local ids
    is_b = band.on_boundary.to(dev)
    mask_e = (~is_b)[lb].to(torch.float64)[..., None, None]
    mask_b = 1.0 - mask_e
    flat, k = lb.reshape(-1), u0.shape[1]
    loads = torch.zeros_like(u0)
    for s in range(0, nt, chunk):
        sl = slice(s, min(s + chunk, nt))
        U = u0[:, :, sl][lb]                                     # (E_b, ngll, ngll, k, T)
        T = U.shape[-1]
        Ke = torch.zeros(nodes.numel(), k, T, dtype=u0.dtype, device=dev).index_add(
            0, flat, _band_residual(eq, elems, U * mask_e).reshape(-1, k, T))   # K (u0 on e)
        Kb = torch.zeros_like(Ke).index_add(
            0, flat, _band_residual(eq, elems, U * mask_b).reshape(-1, k, T))   # K (u0 on b)
        loads[:, :, sl] = torch.where(is_b[:, None, None], -Ke, Kb)
    return BandInjection(band=band, loads=loads)
