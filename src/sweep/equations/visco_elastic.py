"""2-D visco-elastic wave equation: generalized standard linear solid (GSLS).

Same rheology as SPECFEM2D (and hence SeisFlows): ``L`` Zener bodies in
parallel whose relaxation times ``tau_sigma_l`` are FIXED by the frequency band,
giving a nearly constant Q over that band.  Staggered-grid velocity-stress form
after Robertsson, Blanch & Symes (1994).  Each mechanism has its own strength
``tau_l(Q)``: the least-squares solution of ``Q Im M(w) - Re M(w) = 0`` over
the band (Emmerich & Korn 1987), which is LINEAR in ``tau_l`` (an
``L x L`` system per grid point, a rational function of ``1/Q``), so Q is an
ordinary differentiable model; the band fit is within ~5% for Q >= 10 and ~9%
at Q = 3 (the first-order single-tau method of Blanch et al. 1995 is 40% off at
Q = 10).
"""
import itertools

import numpy as np

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec, record_multi
from .elastic import Elastic
from .fields import FieldSpec, ModelSpec
from .slot_table import Slot, SlotTable
from ._registry import register_equation
from ._elastic_step_core import (
    elastic_velocity_substep,
    elastic_velocity_gradients,
    elastic_shear_cpml,
    _fs_sides,
)
from ._free_surface import overwrite_surface_row

# Mechanism count the compiled kernels are built for (register arrays); the
# constant-Q fit already refuses n_sls >= 5 (negative strengths).
MAX_SLS = 4


def gsls_relaxation_times(n_sls, f_min, f_max):
    """Stress relaxation times ``tau_sigma_l = 1 / (2 pi f_l)`` with the
    relaxation frequencies ``f_l`` log-spaced over ``[f_min, f_max]``."""
    if not (0 < f_min < f_max):
        raise ValueError(f"need 0 < f_min < f_max, got ({f_min}, {f_max})")
    if n_sls == 1:
        f = np.array([np.sqrt(f_min * f_max)])
    else:
        f = np.geomspace(f_min, f_max, n_sls)
    return 1.0 / (2.0 * np.pi * f)


def _poly_det(mat):
    """Determinant of a small matrix of polynomials (ascending coefficient
    arrays), by the Leibniz formula -- exact up to float64 rounding."""
    from numpy.polynomial import polynomial as P
    n = len(mat)
    out = np.zeros(1)
    for perm in itertools.permutations(range(n)):
        sign = 1.0
        seen = list(perm)
        for i in range(n):              # parity by cycle decomposition
            while seen[i] != i:
                j = seen[i]
                seen[i], seen[j] = seen[j], seen[i]
                sign = -sign
        term = np.array([sign])
        for i in range(n):
            term = P.polymul(term, mat[i][perm[i]])
        out = P.polyadd(out, term)
    return out


def _horner(coeffs, x):
    """Evaluate the ascending-coefficient polynomial ``coeffs`` at ``x``."""
    acc = float(coeffs[-1]) + 0.0 * x
    for c in coeffs[-2::-1]:
        acc = acc * x + float(c)
    return acc


_GSLS_STRENGTHS = None


def _gsls_strength_function():
    global _GSLS_STRENGTHS
    if _GSLS_STRENGTHS is not None:
        return _GSLS_STRENGTHS
    import torch

    class _GSLSStrengths(torch.autograd.Function):
        """``tau_l = q N_l(q) / D(q)`` elementwise with ``q = 1/Q`` (float64
        internally).  Saves only ``Q``; the backward is the analytic
        derivative, so the per-cell ``L x L`` solve never materialises."""

        @staticmethod
        def forward(ctx, Q, fit):
            ctx.fit = fit
            ctx.save_for_backward(Q)
            q = 1.0 / Q.double()
            d = _horner(fit.D, q)
            return tuple((q * _horner(n, q) / d).to(Q.dtype) for n in fit.N)

        @staticmethod
        def backward(ctx, *grads):
            (Q,) = ctx.saved_tensors
            fit = ctx.fit
            q = 1.0 / Q.double()
            d = _horner(fit.D, q)
            dd = _horner(fit.dD, q)
            gq = torch.zeros_like(q)
            for g, n_c, dn_c in zip(grads, fit.N, fit.dN):
                if g is None:
                    continue
                n = _horner(n_c, q)
                dtau = (n + q * _horner(dn_c, q)) / d - q * n * dd / (d * d)
                gq = gq + g.double() * dtau
            return (-(q * q) * gq).to(Q.dtype), None

    _GSLS_STRENGTHS = _GSLSStrengths
    return _GSLS_STRENGTHS


class GSLSFit:
    """Per-mechanism strengths ``tau_l(Q)`` for fixed ``tau_sigma_l``.

    With ``a_l(w) = (w ts_l)^2 / (1 + (w ts_l)^2)`` and ``b_l(w) = w ts_l /
    (1 + (w ts_l)^2)``, ``M(w)/M_R = 1 + sum_l tau_l (a_l + i b_l)``.  Constant
    Q means ``sum_l tau_l (Q b_l - a_l) = 1`` on the band; the least-squares
    ``tau`` solves ``(BB - (BA + AB) q + AA q^2) tau = q (sb - sa q)`` with
    ``q = 1/Q`` (sampled at ``n_freq`` log-spaced frequencies).  By Cramer's
    rule ``tau_l = q N_l(q) / D(q)`` with polynomial ``D`` (degree ``2L``) and
    ``N_l``; :meth:`tau` evaluates that form elementwise (``q = 0`` gives
    ``tau = 0`` exactly), :meth:`tau_np` the direct solve (reference)."""

    def __init__(self, tau_sigma, f_min, f_max, n_freq=200):
        from numpy.polynomial import polynomial as P
        self.tau_sigma = np.asarray(tau_sigma, dtype=np.float64)
        w = 2.0 * np.pi * np.geomspace(f_min, f_max, n_freq)[:, None]
        wt = w * self.tau_sigma
        a = wt ** 2 / (1.0 + wt ** 2)
        b = wt / (1.0 + wt ** 2)
        self.BB, self.AA = b.T @ b, a.T @ a
        self.BA = b.T @ a + a.T @ b
        self.sb, self.sa = b.sum(0), a.sum(0)
        L = len(self.tau_sigma)
        mat = [[np.array([self.BB[i, j], -self.BA[i, j], self.AA[i, j]]) for j in range(L)]
               for i in range(L)]
        rhs = [np.array([self.sb[i], -self.sa[i]]) for i in range(L)]
        self.D = _poly_det(mat)
        self.N = [_poly_det([[rhs[i] if j == l else mat[i][j] for j in range(L)]
                             for i in range(L)]) for l in range(L)]
        self.dD = P.polyder(self.D)
        self.dN = [P.polyder(n) for n in self.N]

    def min_valid_q(self, q_grid=None):
        """Smallest Q above which every ``tau_l(Q) > 0`` (a physical,
        dissipative relaxation spectrum).  Raises if even the high-Q limit
        has a non-positive strength (too many mechanisms for the band)."""
        w_inf = np.linalg.solve(self.BB, self.sb)        # tau * Q as Q -> inf
        if np.any(w_inf <= 0):
            raise ValueError(
                f"GSLS fit with {len(self.tau_sigma)} mechanisms gives non-positive "
                f"relaxation strengths {w_inf} over this band; use fewer mechanisms "
                "(n_sls <= 4) or a wider band.")
        q_grid = np.geomspace(1.01, 1e6, 600) if q_grid is None else q_grid
        bad = [Q for Q in q_grid if np.any(self.tau_np(Q) <= 0)]
        return 1.0 if not bad else 1.02 * max(bad)

    def tau_np(self, Q):
        q = 1.0 / float(Q)
        M = self.BB - self.BA * q + self.AA * q * q
        return np.linalg.solve(M, q * (self.sb - self.sa * q))

    def tau(self, Q):
        """Torch: ``Q`` of any shape -> list of ``L`` tensors ``tau_l`` (same
        shape/dtype as Q), differentiable in Q."""
        return list(_gsls_strength_function().apply(Q, self))

    def modulus_ratio(self, tau, freq):
        """Numpy ``M(w)/M_R`` for strengths ``tau`` (length L) at ``freq``."""
        w = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)[..., None]
        ts = self.tau_sigma
        return 1.0 + np.sum(np.asarray(tau) * 1j * w * ts / (1.0 + 1j * w * ts), axis=-1)

    def q_of_frequency(self, Q, freq):
        """Exact ``Q(f)`` realised for a target ``Q`` (numpy; checks the fit)."""
        m = self.modulus_ratio(self.tau_np(Q), freq)
        return m.real / m.imag


def _memory_field_specs(n_sls):
    return tuple(
        FieldSpec(f"r_{c}_{l}", description=f"GSLS memory variable {c}, mechanism {l}.", internal=True)
        for l in range(n_sls) for c in ("xx", "zz", "xz")
    )


def _field_specs(n_sls):
    """``[vx, vz, sxx, szz, sxz] + memory variables + 10 CPML memories`` --
    the compiled wavefield order (physical and memory slots are full-grid,
    the CPML slots are per-axis slabs), so field indices agree on both
    backends."""
    return Elastic.FIELD_SPECS[:5] + _memory_field_specs(n_sls) + Elastic.FIELD_SPECS[5:]


def _slot_table(n_sls):
    """``visco_elastic2d`` bind order: elastic2d's five physical fields, the
    ``3 n_sls`` memory variables (full grid, no CPML role), then elastic2d's
    ten CPML memories in their slab order.  No reconstruction list: boundary
    saving is refused."""
    return SlotTable(
        slots=tuple([
            Slot("vx", "vel"), Slot("vz", "vel"),
            Slot("sxx", "stress"), Slot("szz", "stress"), Slot("sxz", "stress"),
        ] + [Slot(f"r_{c}_{l}", "phys") for l in range(n_sls) for c in ("xx", "zz", "xz")]
          + [Slot(f"m_{p}{a}", "pml_mem", a)
             for p in ("vx", "vz", "sxx", "szz", "sxz") for a in ("x", "z")]),
        aux_storage="slab",
    )


@register_equation()
class ViscoElastic(FirstOrderEquation):
    """First-order 2-D visco-elastic wave equation (GSLS, nearly constant Q).

    Velocity-stress staggered grid as :class:`~sweep.equations.elastic.Elastic`
    plus ``3 L`` memory variables (``r_xx_l``, ``r_zz_l``, ``r_xz_l``, one
    triple per standard linear solid).  With relaxed P and shear moduli
    ``pi_R``, ``mu_R``, strengths ``tp_l = tau_l(Qp)``, ``ts_l = tau_l(Qs)``
    and ``theta = dvx/dx + dvz/dz``::

        dsxx/dt = pi_U theta - 2 mu_U dvz/dz + sum_l r_xx_l
        dszz/dt = pi_U theta - 2 mu_U dvx/dx + sum_l r_zz_l
        dsxz/dt = mu_U (dvx/dz + dvz/dx)    + sum_l r_xz_l
        dr_xx_l/dt = -(r_xx_l + pi_R tp_l theta - 2 mu_R ts_l dvz/dz) / tau_sigma_l
        (r_zz_l, r_xz_l analogous)

    with unrelaxed moduli ``pi_U = pi_R (1 + sum_l tp_l)``, ``mu_U = mu_R (1 +
    sum_l ts_l)``, i.e. ``M(w) = M_R (1 + sum_l tau_l i w tau_sigma_l / (1 + i w
    tau_sigma_l))``.  The memory variables live at integer time levels and are
    advanced with the trapezoidal rule (unconditionally stable for any
    ``dt / tau_sigma_l``); the stress picks up their time average.

    **Models** are ``vp, vs, rho, Qp, Qs``; ``vp``/``vs`` are phase velocities
    at the reference frequency ``f_ref`` (SPECFEM's
    ``READ_VELOCITIES_AT_f0 = .true.``), converted exactly to the relaxed
    moduli through the GSLS complex modulus at ``f_ref``.  All five models are
    differentiable.  ``Qp = Qs = inf`` reduces bit-exactly to ``Elastic``.

    The ``tau_sigma_l`` are fixed by ``(n_sls, f_band)`` at construction
    (relaxation frequencies log-spaced over the band) and are not model
    parameters; ``tau_l(Q)`` is the least-squares constant-Q fit (see
    :class:`GSLSFit`).

    **Stability:** the explicit stress update runs at the UNRELAXED
    velocities, which exceed ``vp``/``vs`` (e.g. ~8% at Q = 10 for a 12x band),
    so choose ``dt`` from the CFL condition on
    ``sqrt((lam_U + 2 mu_U) / rho)`` (:meth:`unrelaxed_velocities`).  ``Q`` must
    exceed ``self.q_min`` (about 1.3 for the default 3 mechanisms), below which
    the fit would need a negative relaxation strength; ``n_sls >= 5`` is
    refused for the same reason.

    **Free surface** (flat, per edge): the surface-row normal strain rate is
    solved so that the normal traction stays exactly zero INCLUDING the memory
    variables; in the elastic limit this is Robertsson's modified coefficient
    ``4 mu (lam + mu) / (lam + 2 mu)`` used by ``Elastic``.

    **Backends:** eager and compiled CUDA (``impl='c'``; forward, and the
    ``full``, chunk- and recursive-checkpoint backwards with a hand-derived
    exact adjoint).  Not supported: boundary saving (the attenuation is
    dissipative, so reverse reconstruction is unstable; ``impl='c'`` defaults
    to ``full``), topography / APM, domain decomposition, per-shot batched
    models.

    References:
        Robertsson, J. O., Blanch, J. O. and Symes, W. W., 1994, Viscoelastic
        finite-difference modeling: Geophysics, 59(9), 1444-1456,
        doi:10.1190/1.1443701 -- the staggered-grid memory-variable scheme.
        Emmerich, H. and Korn, M., 1987, Incorporation of attenuation into
        time-domain computations of seismic wave fields: Geophysics, 52(9),
        1252-1264, doi:10.1190/1.1442386 -- the constant-Q fit (linear least
        squares for the strengths at fixed relaxation frequencies).
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="P-wave phase velocity at f_ref.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="S-wave phase velocity at f_ref.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
        ModelSpec("Qp", description="P-wave quality factor (inf = no attenuation)."),
        ModelSpec("Qs", description="S-wave quality factor (inf = no attenuation)."),
    )
    FIELD_SPECS = _field_specs(3)

    # Derives from FirstOrderEquation, NOT Elastic: code that resolves an
    # equation through its MRO (e.g. the domain-decomposition whitelist) must
    # not mistake this for Elastic and run the elastic CUDA kernels.  The
    # elastic pieces are reused by composition instead.
    C_NAME = "visco_elastic2d"
    default_pml_type = "cpmls"   # staggered-grid CPML (8 profiles)
    supports_per_edge_free_surface = True
    supports_per_edge_free_surface_c = True
    # No interior_substeps: the attenuation is dissipative, so boundary saving
    # (reverse reconstruction) is refused and impl='c' defaults to 'full'.
    supports_boundary_saving_c = False
    # The compiled kernels consume the PREPARED set (see prepare_models);
    # autograd chains their gradients back to (vp, vs, rho, Qp, Qs).
    prepare_models_for_c = True

    def __init__(self, spatial_order=4, device='cpu', backend='torch',
                 f_ref=10.0, f_band=None, n_sls=3):
        """
        Args:
            spatial_order: FD order (as ``Elastic``).
            device: Device for the gradient kernels.
            backend: ``'torch'`` only.
            f_ref: Reference frequency (Hz) at which ``vp`` / ``vs`` are given.
            f_band: ``(f_min, f_max)`` over which Q is fitted constant. Defaults
                to SPECFEM's band: a factor 12 centred (geometrically) on f_ref.
                For FWI set it to the inverted band.
            n_sls: Number of standard linear solids (SPECFEM default 3).
        """
        if backend != 'torch':
            raise NotImplementedError("ViscoElastic supports backend='torch' only")
        if not 1 <= int(n_sls) <= MAX_SLS:
            raise ValueError(f"n_sls must be in [1, {MAX_SLS}], got {n_sls}")
        super().__init__(spatial_order, device, backend)
        if f_band is None:
            r = np.sqrt(12.0)
            f_band = (f_ref / r, f_ref * r)
        self.f_ref = float(f_ref)
        self.f_band = (float(f_band[0]), float(f_band[1]))
        self.n_sls = int(n_sls)
        self.tau_sigma = gsls_relaxation_times(self.n_sls, *self.f_band)
        # As Python floats for the step: a numpy array read there is traced as
        # a tensor, and float() of it is a data-dependent graph break.
        self._tau_sigma_floats = tuple(float(v) for v in self.tau_sigma)
        self.fit = GSLSFit(self.tau_sigma, *self.f_band)
        self.q_min = self.fit.min_valid_q()
        if self.n_sls != 3:
            specs = _field_specs(self.n_sls)
            self.field_specs = specs
            self.wavefields = [s.name for s in specs]

    @property
    def default_source_fields(self):
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vz"]

    _fs_zero_traction = Elastic._fs_zero_traction

    # ------------------------------------------------------------------ models
    def _moduli_factors(self, taus):
        """For strengths ``taus`` (list of L tensors) return ``(s_U, s_R)``: the
        factors mapping ``rho v^2`` (v = phase velocity at f_ref) to the
        unrelaxed and relaxed moduli.

        ``m = M(w_ref)/M_R = a + i b``; the phase velocity is
        ``v = sqrt(M_R/rho) / Re(m^-1/2)`` so ``M_R = rho v^2 cos^2(phi/2)/|m|``."""
        import torch
        wt = 2.0 * np.pi * self.f_ref * self.tau_sigma
        ca = wt ** 2 / (1.0 + wt ** 2)
        cb = wt / (1.0 + wt ** 2)
        a = 1.0 + sum(float(ca[l]) * t for l, t in enumerate(taus))
        b = sum(float(cb[l]) * t for l, t in enumerate(taus))
        absm = torch.sqrt(a * a + b * b)
        cos2 = 0.5 * (1.0 + a / absm)            # cos^2(phi/2), phi = atan2(b, a)
        s_R = cos2 / absm
        s_U = s_R * (1.0 + sum(taus))
        return s_U, s_R

    def prepare_models(self, models):
        """``(vp, vs, rho, Qp, Qs)`` -> ``[lam_U, mu_U, rho, lam_U + 2 mu_U,
        pi_R tp_0 .. pi_R tp_{L-1}, mu_R ts_0 .. mu_R ts_{L-1}]`` -- the step
        coefficients shared by the eager step and the CUDA kernels (rho third,
        where elastic2d's kernels keep it).  The
        unrelaxed Lame set is exactly ``Elastic._lame`` of the unrelaxed
        velocities, so at ``Q = inf`` (``tau_l = 0`` -> factors exactly 1) the
        elastic part is bit-identical to ``Elastic``.

        impl='c' hands ``(B, 1, nz, nx)`` repeats of one shared model (this
        equation has no per-shot models): the coefficients are computed once
        and repeated, and the repeat's backward sums the per-shot gradients."""
        if len(models) != 5:
            return list(models)
        vp, vs, rho, Qp, Qs = models
        batch = vp.shape[0] if vp.ndim == 4 else None
        if batch is not None:
            vp, vs, rho, Qp, Qs = (m[:1] for m in (vp, vs, rho, Qp, Qs))
        q_lo = float(min(Qp.detach().min(), Qs.detach().min()))
        if not q_lo > self.q_min:
            raise ValueError(
                f"ViscoElastic: min(Qp, Qs) = {q_lo:.3g} is below {self.q_min:.3g}, "
                "where the constant-Q fit yields a negative relaxation strength "
                f"(n_sls={self.n_sls}, f_band={self.f_band}). Bound Q from below.")
        tp = self.fit.tau(Qp)
        ts = self.fit.tau(Qs)
        sU_p, sR_p = self._moduli_factors(tp)
        sU_s, sR_s = self._moduli_factors(ts)
        _, _, rho, lam, mu, lam2mu = Elastic._lame((vp * sU_p.sqrt(), vs * sU_s.sqrt(), rho))
        pi_R = rho * vp * vp * sR_p
        mu_R = rho * vs * vs * sR_s
        out = [lam, mu, rho, lam2mu] + [pi_R * t for t in tp] + [mu_R * t for t in ts]
        if batch is not None:
            out = [t.repeat(batch, 1, 1, 1) for t in out]
        return out

    def unrelaxed_velocities(self, models):
        """``(vp_U, vs_U)`` for ``(vp, vs, rho, Qp, Qs)``: the velocities the
        explicit stress update actually runs at (above ``vp`` / ``vs`` for
        finite Q, equal at ``Q = inf``) -- take ``dt`` from the CFL condition
        on ``vp_U``."""
        lam, mu, rho, lam2mu = self.prepare_models(list(models))[:4]
        return (lam2mu / rho).sqrt(), (mu / rho).sqrt()

    def relaxation_coefficients(self, dt):
        """Trapezoidal memory-update weights ``r+ = a_l r - c_l F``:
        ``a_l = (1 - dt/2ts) / (1 + dt/2ts)``, ``c_l = (dt/ts) / (1 + dt/2ts)``.
        ``dt`` may be the eager step's float32 0-d tensor (possibly inside a
        compiled step: kept symbolic); :meth:`c_eq_aux` evaluates the same
        expressions so both backends see identical float32 weights."""
        a_l, c_l = [], []
        for ts in self._tau_sigma_floats:
            half = 0.5 * dt / ts
            a_l.append((1.0 - half) / (1.0 + half))
            c_l.append((dt / ts) / (1.0 + half))
        return a_l, c_l

    # -------------------------------------------------------------------- step
    def func(self, wavefields, models, dt, h, b, **kwargs):
        if getattr(self, "_topo_rows_runtime", None) is not None:
            raise NotImplementedError("ViscoElastic does not support topography yet")
        models = self.prepare_models(models)
        L = self.n_sls
        lam, mu, rho, lam2mu = models[:4]
        p_tau = models[4:4 + L]
        m_tau = models[4 + L:4 + 2 * L]
        base = list(wavefields[:5]) + list(wavefields[5 + 3 * L:])
        rmem = list(wavefields[5:5 + 3 * L])

        free_surface = getattr(self, "free_surface", False)
        fs_faces = getattr(self, "fs_faces", None)
        kw = dict(
            lame_lambda=lam, lame_mu=mu, mu_xz=mu, rho_x=rho, rho_z=rho,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=free_surface, topo_rows=None, fs_faces=fs_faces,
            lame_lambda_2mu=lam2mu,
        )
        (vx, vz, sxx, szz, sxz,
         m_vxx, m_vxz, m_vzx, m_vzz,
         m_txxx, m_txxz, m_tzzx, m_tzzz,
         m_txzx, m_txzz) = elastic_velocity_substep(*base, **kw)

        exx, ezz, vx_z, vz_x, m_vxx, m_vzz = elastic_velocity_gradients(
            vx, vz, m_vxx, m_vzz,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=free_surface, topo_rows=None, fs_faces=fs_faces,
        )
        vx_z, vz_x, m_vxz, m_vzx = elastic_shear_cpml(vx_z, vz_x, m_vxz, m_vzx, self.b)
        exz = vx_z + vz_x

        a_l, c_l = self.relaxation_coefficients(dt)
        r_xx = rmem[0::3]
        r_zz = rmem[1::3]
        r_xz = rmem[2::3]

        # ---- free surface: solve the surface normal strain rate so that the
        # normal traction stays zero through the full (memory-variable) update:
        # 0 = dt (lam2mu e + lam e_t) + dt/2 sum_l (r+_l + r_l),
        # r+_l = a_l r_l - c_l (P_l (e_t + e) - 2 M_l e_t).
        # z faces first (from the raw dvx/dx), then x faces (from the -- at a
        # z-x corner already solved -- dvz/dz); the CUDA adjoint transposes
        # exactly this order.
        z_sides, x_sides = _fs_sides(fs_faces, free_surface)
        if z_sides or x_sides:
            halo = self.pd.coes.shape[0]
            den = lam2mu - 0.5 * sum(c_l[l] * p_tau[l] for l in range(L))
            cross = 0.5 * sum(c_l[l] * (p_tau[l] - 2.0 * m_tau[l]) for l in range(L))
            if z_sides:     # szz = 0: unknown dvz/dz
                hist = sum(0.5 * (1.0 + a_l[l]) * r_zz[l] for l in range(L))
                e = -(lam * exx + hist - cross * exx) / den
                for side in z_sides:
                    ezz = overwrite_surface_row(ezz, e, halo, axis=-2, side=side)
            if x_sides:     # sxx = 0: unknown dvx/dx
                hist = sum(0.5 * (1.0 + a_l[l]) * r_xx[l] for l in range(L))
                e = -(lam * ezz + hist - cross * ezz) / den
                for side in x_sides:
                    exx = overwrite_surface_row(exx, e, halo, axis=-1, side=side)

        theta = exx + ezz
        sum_xx = sum_zz = sum_xz = 0.0
        new_r = []
        for l in range(L):
            tp = p_tau[l] * theta
            rxx = a_l[l] * r_xx[l] - c_l[l] * (tp - 2.0 * m_tau[l] * ezz)
            rzz = a_l[l] * r_zz[l] - c_l[l] * (tp - 2.0 * m_tau[l] * exx)
            rxz = a_l[l] * r_xz[l] - c_l[l] * (m_tau[l] * exz)
            sum_xx = sum_xx + (rxx + r_xx[l])
            sum_zz = sum_zz + (rzz + r_zz[l])
            sum_xz = sum_xz + (rxz + r_xz[l])
            new_r += [rxx, rzz, rxz]

        szz = szz + dt * (lam2mu * ezz + lam * exx)
        sxx = sxx + dt * (lam2mu * exx + lam * ezz)
        sxz = sxz + dt * mu * exz
        szz = szz + (0.5 * dt) * sum_zz
        sxx = sxx + (0.5 * dt) * sum_xx
        sxz = sxz + (0.5 * dt) * sum_xz

        if free_surface:
            szz, sxz, sxx = self._fs_zero_traction(szz, sxz, sxx, None)

        return (
            vx, vz, sxx, szz, sxz,
            *new_r,
            m_vxx, m_vxz, m_vzx, m_vzz,
            m_txxx, m_txxz, m_tzzx, m_tzzz,
            m_txzx, m_txzz,
        )

    # ----------------------------------------------------------------- CUDA
    def c_eq_aux(self, prop):
        """``eq_aux`` for the CUDA kernels: one float32 CPU tensor
        ``[a_0 .. a_{L-1}, c_0 .. c_{L-1}]``, evaluated exactly as the eager
        step evaluates them (float32 ``dt``)."""
        import torch
        dt = torch.tensor(float(prop._dt), dtype=torch.float32)
        a_l, c_l = self.relaxation_coefficients(dt)
        return (torch.stack(a_l + c_l).contiguous(),)

    @property
    def cuda_layout(self):
        n_mem = 3 * self.n_sls
        elastic = ("x", "z", "x", "z", "x", "z", "x", "z", "x", "z")

        def strip(grid):         # the surface strip: z-low/-high rows, x-low/-high columns
            return 2 * grid[-1] + 2 * grid[-2]

        def save_all_shape(B, nt, grid):
            # [vx | vz | strip] per step: the gradient reads the solved surface
            # strain rates of the forward (visco_elastic2d driver_traits).
            return (nt, 2 * B * grid[-2] * grid[-1] + B * strip(grid))

        def backward_workspace_shapes(B, nt, grid, mode):
            # [0-7] elastic2d's q*/p* adjoint scratch; then the staggered
            # skeleton's slots (sg_driver.cuh SgCarrierSlots, N_VEL = 3):
            # full = one read-only zero grid (v(nt)), checkpoint modes = the
            # vx / vz / strip carriers for v(t) and v(t+1).
            n = 8 + (6 if mode in ("ckpt", "recursive") else 1 if mode == "full" else 0)
            return [[B, 1, *grid]] * n

        def checkpoint_replay_shapes(B, nt, grid, seg, mode):
            # [0] the replay's strip scratch; chunk mode adds one chunk's
            # vx / vz / strip histories (row 0 = v(start), row k = v(start+k)).
            shapes = [(B, strip(grid))]
            if mode == "ckpt":
                shapes += [(seg + 1, B, 1, *grid)] * 2 + [(seg + 1, B, strip(grid))]
            return shapes

        return CUDALayoutSpec(
            record_shape=record_multi(),
            save_all_shape=save_all_shape,
            # vx, vz, sxx, szz, sxz + the memory variables (full grid); the ten
            # CPML memories (C++ bind order as Elastic) in per-axis slabs.
            base_nvar=5 + n_mem,
            pml_nvar=10,
            last_two_nvar=1,
            last_two_storage_nvar=5,
            backward_workspace_nvar=8,
            backward_workspace_shapes=backward_workspace_shapes,
            checkpoint_replay_shapes=checkpoint_replay_shapes,
            pml_slot_axes=elastic,
            checkpoint_slot_axes=(None,) * (5 + n_mem) + elastic,
            adjoint_pml_slab=True,
            slots=_slot_table(self.n_sls),
            grads_out_has_wavelet=False,
            illum_nvar=0,
        )
