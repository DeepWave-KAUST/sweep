"""The nonlinear constant-Q fit of a generalized standard linear solid: SPECFEM2D's
``USE_SOLVOPT``, where the relaxation frequencies are fitted with the strengths, for
each Q.

:class:`~sweep.equations.visco_elastic.GSLSFit` keeps the relaxation frequencies
``theta_l = 1 / tau_sigma_l`` log-spaced over the band and solves the linear least
squares for the strengths ``w_l`` (``M(w) / M_R = 1 + sum_l w_l i w / (theta_l + i w)``).
The same residual, ``Q Im m(w_k) - Re m(w_k)`` at ``n_freq`` frequencies log-spaced
over the band with its ends, is minimised here over both, with ``theta_l`` and ``w_l``
kept positive -- a dissipative spectrum for every Q, where the linear fit's strengths
turn negative below its ``q_min``. Over SPECFEM2D's band (``f_ref / 10 .. 10 f_ref``) at
``Q = 30`` the largest error of ``Q(f)`` drops from 5.1% (the linear fit at SPECFEM2D's
``2 n_sls - 1`` frequencies) to 3.3% with 3 mechanisms and from 3.4% to 0.7% with 4; at
``Q = 3`` with 4, from 7.0% to 1.0%.

SPECFEM2D runs SolvOpt (Shor's r-algorithm) from the linear fit at each point and
stops at about four significant digits; this runs Levenberg-Marquardt to full
precision -- the same optimum, its residual a little lower. The fit is scale-free
in frequency, so one table per (``n_sls``, band ratio, ``n_freq``) serves every band
of that ratio: the optimum at the Chebyshev nodes of ``q = 1 / Q`` in ``[0, 1]``,
followed upward from ``q = 0`` (each node the better of the previous node's optimum
and SPECFEM2D's linear start), and ``log theta_l(q)`` and ``w_l(q) / q`` as their
Chebyshev series. A model's Q then maps to ``(tau_sigma_l, w_l)`` by polynomials:
elementwise, differentiable, and ``Q = inf`` gives ``w_l = 0`` exactly. The table is
checked at the midpoints between its nodes against fresh fits, and refused where the
optimum is not a smooth function of Q there (one mechanism: its optimum jumps
between two branches as Q falls).
"""

from __future__ import annotations

import numpy as np

# Chebyshev degree of the tables, and the smallest Q they cover (q = 1/Q <= 1)
_DEGREE = 48
Q_MIN = 1.0
# the table's residual at a midpoint may exceed a fresh fit's by this (relative)
_MID_TOL = 1e-8
_TABLES: dict = {}


def _design(y, q, w, n):
    y = np.clip(y, -60.0, 60.0)            # a wild trial step stays finite
    th, v = np.exp(y[:n]), np.exp(y[n:])
    d = th[None, :] ** 2 + w[:, None] ** 2
    a = w[:, None] * (th[None, :] - w[:, None] * q) / d
    return th, v, d, a


def _residuals(y, q, w, n):
    _, v, _, a = _design(y, q, w, n)
    return a @ v - 1.0


def _jacobian(y, q, w, n):
    th, v, d, a = _design(y, q, w, n)
    w2 = w[:, None]
    j_th = v * w2 * th * (w2 * w2 - th * th + 2.0 * th * w2 * q) / (d * d)
    return np.concatenate([j_th, a * v], axis=1)


def _cost(y, q, w, n):
    r = _residuals(y, q, w, n)
    return 0.5 * float(r @ r)


def _linear_start(q, w, n, ratio):
    """SPECFEM2D's start: ``theta_l`` log-spaced over the band with its ends, the
    strengths from the linear least squares at ``2 n - 1`` frequencies."""
    th = ratio ** (np.arange(n) / (n - 1) - 0.5)
    m = 2 * n - 1
    wm = ratio ** (np.arange(m) / (m - 1) - 0.5)
    a = wm[:, None] * (th[None, :] - wm[:, None] * q) / (th[None, :] ** 2 + wm[:, None] ** 2)
    v = np.linalg.solve(a.T @ a, a.T @ np.ones(m))
    return np.concatenate([np.log(th), np.log(np.abs(v))])


def _solve(q, w, n, y0):
    from scipy.optimize import least_squares
    r = least_squares(_residuals, y0, jac=_jacobian, args=(q, w, n), method="lm",
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200 * (2 * n + 1))
    y = r.x
    order = np.argsort(y[:n])                  # mechanisms in ascending frequency
    return np.concatenate([y[:n][order], y[n:][order]])


def _best(q, w, n, ratio, warm):
    cands = [_solve(q, w, n, _linear_start(q, w, n, ratio))]
    if warm is not None:
        cands.append(_solve(q, w, n, warm))
    return min(cands, key=lambda y: _cost(y, q, w, n))


def _table(n, ratio, n_freq):
    key = (n, float(ratio), int(n_freq))
    if key in _TABLES:
        return _TABLES[key]
    from numpy.polynomial import chebyshev as C
    w = ratio ** (np.arange(n_freq) / (n_freq - 1) - 0.5)   # frequencies over the band centre
    t = np.cos(np.pi * (np.arange(_DEGREE + 1) + 0.5) / (_DEGREE + 1))[::-1]
    ys, y = [], None
    for tj in t:
        y = _best(0.5 * (tj + 1.0) / Q_MIN, w, n, ratio, y)
        ys.append(y)
    coef = C.chebfit(t, np.array(ys), _DEGREE)
    worst = 0.0
    for tm in 0.5 * (t[1:] + t[:-1]):
        q = 0.5 * (tm + 1.0) / Q_MIN
        yi = C.chebval(tm, coef)
        c_tab, c_fit = _cost(yi, q, w, n), _cost(_best(q, w, n, ratio, yi), q, w, n)
        worst = max(worst, (c_tab - c_fit) / c_fit)
    if not worst <= _MID_TOL:
        raise ValueError(
            f"the nonlinear constant-Q fit with {n} mechanisms over a band ratio of "
            f"{ratio:.4g} at {n_freq} frequencies is not a smooth function of Q (a "
            f"table midpoint is {worst:.1e} above a fresh fit): use fit='linear', or "
            "another number of mechanisms or band")
    _TABLES[key] = coef
    return coef


class GSLSNonlinearFit:
    """Relaxation times and strengths of ``n_sls`` standard linear solids fitted for a
    constant Q over ``[f_min, f_max]`` by the nonlinear least squares at ``n_freq``
    log-spaced frequencies (default ``4 n_sls``, SPECFEM2D's). See the module
    docstring. :meth:`params` maps a tensor of Q elementwise (differentiable),
    :meth:`params_np` fits one Q directly (the reference)."""

    def __init__(self, n_sls, f_min, f_max, n_freq=None):
        n = int(n_sls)
        if n < 2:
            raise ValueError(
                "fit='nonlinear' needs n_sls >= 2: with one mechanism the optimum jumps "
                "between two branches as Q falls, so it is no function of Q to "
                "differentiate. Use fit='linear'.")
        if not 0 < f_min < f_max:
            raise ValueError(f"need 0 < f_min < f_max, got ({f_min}, {f_max})")
        self.n_sls = n
        self.f_band = (float(f_min), float(f_max))
        self.n_freq = 4 * n if n_freq is None else int(n_freq)
        if self.n_freq < 2 * n:
            raise ValueError(f"n_freq = {self.n_freq} frequencies cannot fit the "
                             f"{2 * n} parameters of {n} mechanisms")
        self.ratio = self.f_band[1] / self.f_band[0]
        self.omega_c = 2.0 * np.pi * np.sqrt(self.f_band[0] * self.f_band[1])
        self.q_min = Q_MIN
        self._coef = _table(n, self.ratio, self.n_freq)

    def params(self, Q):
        """Torch: ``Q`` of any shape -> ``(tau_sigma, w)``, two lists of ``n_sls``
        tensors shaped and typed as ``Q``, differentiable in ``Q``."""
        import torch
        n = self.n_sls
        q = 1.0 / Q.double()
        t = 2.0 * Q_MIN * q - 1.0
        coef = torch.as_tensor(self._coef, dtype=torch.float64, device=Q.device)
        t2 = (2.0 * t).unsqueeze(-1)
        b1 = torch.zeros(t.shape + (2 * n,), dtype=torch.float64, device=Q.device)
        b2 = torch.zeros_like(b1)
        for c in coef.flip(0)[:-1]:                        # Clenshaw
            b1, b2 = t2 * b1 - b2 + c, b1
        y = t.unsqueeze(-1) * b1 - b2 + coef[0]
        tau = [(1.0 / (self.omega_c * torch.exp(y[..., l]))).to(Q.dtype) for l in range(n)]
        w = [(q * torch.exp(y[..., n + l])).to(Q.dtype) for l in range(n)]
        return tau, w

    def params_np(self, Q):
        """Numpy, one ``Q``: ``(tau_sigma, w)`` of a direct fit from SPECFEM2D's start."""
        n, q = self.n_sls, 1.0 / float(Q)
        w = self.ratio ** (np.arange(self.n_freq) / (self.n_freq - 1) - 0.5)
        y = _best(q, w, n, self.ratio, None)
        return 1.0 / (self.omega_c * np.exp(y[:n])), q * np.exp(y[n:])

    @staticmethod
    def modulus_ratio(tau_sigma, w, freq):
        """Numpy ``M(f) / M_R`` of the relaxation times ``tau_sigma`` and strengths
        ``w`` (length ``n_sls`` each) at ``freq``."""
        wt = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)[..., None] * np.asarray(tau_sigma)
        return 1.0 + np.sum(np.asarray(w) * 1j * wt / (1.0 + 1j * wt), axis=-1)

    def q_of_frequency(self, Q, freq):
        """``Q(f)`` the fit realises for a target ``Q`` (numpy)."""
        m = self.modulus_ratio(*self.params_np(Q), freq)
        return m.real / m.imag
