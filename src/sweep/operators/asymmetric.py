"""Asymmetric collocated first-derivative operators.

Implements the finite-difference operators of

    Bartana, A., Xu, J. & Rickett, J. (SLB) -- *Solving the elastodynamic
    equations in curvilinear coordinates using asymmetric operators*.

An **asymmetric operator** is a first-derivative stencil with the *same*
number of coefficients on both sides of the output point but whose pairs
are not anti-symmetric (``c_-l != -c_l``).  With the paper's plane-wave
convention (``p_{i+l} = p_i exp(-i k l dx)``) the stencil symbol is

    D(theta) = sum_l c_l exp(-i l theta),        theta = k dx,

and the target is the exact derivative ``-i theta``.  The anti-symmetric
part of ``c`` produces the imaginary part of the symbol, the symmetric
part the real part.  A central operator has a purely imaginary symbol
that *vanishes* at Nyquist -- that null is exactly the odd-even
decoupling of a collocated grid.  The asymmetric operator keeps the
imaginary part fitted to ``-theta`` over the resolved band, holds the
real part at zero in band, and forces it to ``-2`` at Nyquist, so the
symbol never vanishes and the decoupled mode disappears.

Coefficients come from a least-squares fit in the wavenumber domain
(paper eq. 6)

    E = min integral_0^{kmax} | sum_l c_l exp(-i l theta) + i theta |^2 dtheta

subject to an equality constraint on the real part at Nyquist.  With
``L = 8`` (17 coefficients) and ``kmax = 0.56 k_Nyquist`` this reproduces
the paper's Figure 1: in band the imaginary part follows ``-theta`` to
~1e-5 and the real part stays below 1e-4, then decays to exactly ``-2``
at Nyquist.

Two exact moment constraints (``sum_l c_l = 0`` and ``sum_l c_l l = 1``)
are imposed on top of the fit.  They perturb the interior operator by
less than 1e-4 but guarantee that a constant field differentiates to
exactly zero -- a pure least-squares fit leaves a small residual there,
which in a conservative velocity-stress scheme acts as a spurious source
that accumulates over a long simulation.

Operator pairing
----------------
A collocated velocity-stress scheme applies a first-derivative operator
twice per step (velocity -> stress, stress -> velocity).  Using the
*same* asymmetric stencil in both places is unconditionally unstable: the
composite symbol ``(E - ik)^2`` is complex, so one leapfrog root always
leaves the unit circle.  The paper's remark that "the operator in
equations (3) is symmetric (self-adjoint)" pins down the correct pairing:
the divergence operator is the negative transpose of the gradient
operator, ``c*_l = -c_{-l}``.  The composite symbol is then
``-(E^2 + k^2)`` -- real, negative and non-vanishing at Nyquist -- and
the semi-discrete scheme conserves energy.

Free surface
------------
The paper puts the top row of the grid *on* the free surface and uses no
ghost points, so near the surface the centred stencil has to be
truncated.  Boundary stencils come from the same least-squares machinery
with the offsets restricted to ``[-r, min(L, r + surface_reach)]`` at
depth ``r``: as centred as the surface allows, tapering to a plain
forward difference on the surface row itself.  Their Nyquist real part is
constrained to ``boundary_nyquist_real`` for the gradient and its
negation for the divergence, mirroring the interior pair.

That closure is *consistent* but not exactly energy conserving, and it
cannot be made both: an identity-norm summation-by-parts closure does not
exist for an asymmetric interior operator (the row conditions on the
operator and the column conditions on its transpose are jointly
infeasible for any finite boundary support).  The residual supports a
near-static, high-wavenumber, stress-dominated mode trapped in the first
``L`` rows -- an hourglass mode -- which grows at about ``1.0006`` per
step.  Two remedies are provided and measured in
:mod:`sweep.equations.elastic_curvilinear`: a tapered Kreiss-Oliger
filter on the stresses in the surface band (the default, cheap and local
to the unresolved wavenumbers), and ``adjoint_boundary_weight``, which
blends the divergence block toward the exact negative transpose
(unconditionally neutral, but no longer a consistent derivative right
where the surface wave lives).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Least-squares coefficients
# --------------------------------------------------------------------------

def asymmetric_coefficients(offsets, kmax_frac=0.56, nyquist_real=None,
                            moments=(0, 1), n_quad=1024):
    """Least-squares first-derivative coefficients for arbitrary offsets.

    Fits ``sum_l c_l exp(-i l theta) ~ -i theta`` over
    ``theta in [0, kmax_frac * pi]``, real and imaginary parts
    simultaneously (paper eq. 6).  Coefficients are dimensionless --
    divide by the grid spacing to get the derivative.

    Parameters
    ----------
    offsets : sequence of int
        Stencil offsets ``l``.  ``range(-L, L+1)`` gives the interior
        operator; a truncated range gives a boundary operator.
    kmax_frac : float
        Upper edge of the fit band as a fraction of the Nyquist
        wavenumber.
    nyquist_real : float or None
        Equality constraint on the real part of the symbol at Nyquist,
        ``sum_l c_l (-1)^l = nyquist_real``.  This is what makes the
        operator asymmetric: without it the symmetric part of a centred
        offset set comes out identically zero and the operator degenerates
        to plain optimised central differences.  Ignored (as
        over-determined) for stencils with fewer than three points.
    moments : sequence of int
        Taylor moments constrained exactly; ``(0, 1)`` is plain
        consistency (annihilates constants, differentiates a ramp to 1).
    n_quad : int
        Number of quadrature nodes in the fit band.

    Returns
    -------
    numpy.ndarray
        ``float64`` coefficients aligned with ``offsets``.
    """
    off = np.asarray(offsets, dtype=np.float64)
    if off.size == 0:
        raise ValueError("offsets must be non-empty")
    theta = np.linspace(1e-9, float(kmax_frac) * np.pi, int(n_quad))

    # Residual split into real / imaginary rows:
    #   real: sum_l c_l cos(l theta)          -> 0
    #   imag: -sum_l c_l sin(l theta) + theta -> 0
    a = np.vstack([np.cos(np.outer(theta, off)),
                   -np.sin(np.outer(theta, off))])
    rhs = np.concatenate([np.zeros_like(theta), -theta])

    con, con_rhs = [], []
    for p in moments:
        con.append(off ** p)
        con_rhs.append(1.0 if p == 1 else 0.0)
    if nyquist_real is not None and off.size >= 3:
        con.append(np.cos(np.pi * off))
        con_rhs.append(float(nyquist_real))
    con = np.array(con, dtype=np.float64)
    con_rhs = np.array(con_rhs, dtype=np.float64)

    # Equality-constrained least squares through the KKT system.
    n, m = off.size, con.shape[0]
    kkt = np.zeros((n + m, n + m), dtype=np.float64)
    kkt[:n, :n] = a.T @ a + 1e-12 * np.eye(n)
    kkt[:n, n:] = con.T
    kkt[n:, :n] = con
    return np.linalg.solve(kkt, np.concatenate([a.T @ rhs, con_rhs]))[:n]


def symbol(coef, offsets, theta):
    """Wavenumber-domain symbol ``sum_l c_l exp(-i l theta)``."""
    e = np.exp(-1j * np.outer(np.asarray(theta, dtype=np.float64),
                              np.asarray(offsets, dtype=np.float64)))
    return e @ np.asarray(coef, dtype=np.float64)


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

def _project_moments(block, moments=(0, 1)):
    """Nearest consistent operator to ``block`` (row-wise, Frobenius).

    The exact negative transpose of the gradient closure is energy
    conserving but not a consistent derivative -- its boundary rows have
    row sums up to 0.67 instead of 0, so a constant stress field would
    accelerate the top rows.  Projecting each row onto the two moment
    conditions restores consistency and moves the block only a fifth as
    far from the exact adjoint as the plain consistent closure sits
    (Frobenius distance 0.32 versus 1.55), which is enough to turn the
    growing boundary mode into a marginal one.
    """
    out = np.array(block, dtype=np.float64, copy=True)
    nr, nc = out.shape
    for r in range(nr):
        l = np.arange(nc) - r
        A = np.vstack([l.astype(np.float64) ** m for m in moments])
        c = np.array([1.0 if m == 1 else 0.0 for m in moments])
        lam = np.linalg.solve(A @ A.T, c - A @ out[r])
        out[r] = out[r] + A.T @ lam
    return out


class CollocatedAsymmetricOperator:
    """Collocated asymmetric first-derivative operator and its adjoint.

    Parameters
    ----------
    half_width : int
        ``L``; the interior stencil spans offsets ``-L .. L``
        (``2L + 1`` coefficients).  The paper uses ``L = 8``.
    kmax_frac : float
        Fit band edge as a fraction of Nyquist (paper: ``0.56``).
    nyquist_real : float
        Real part of the interior symbol at Nyquist (paper: ``-2``).
        Setting it to ``0`` recovers a plain optimised central operator
        and brings the odd-even decoupling back -- useful as a control.
    surface_reach : int
        How far past "as centred as possible" a boundary stencil reaches:
        at depth ``r`` the offsets are ``[-r, min(L, r + surface_reach)]``.
    adjoint_boundary_weight : float
        Blend between two irreconcilable requirements on the divergence
        block at the free surface.  ``0`` uses stencils that are
        *consistent* first derivatives but break the discrete
        integration-by-parts identity, so the closure can pump energy;
        ``1`` uses the exact negative transpose of the gradient block,
        which conserves energy exactly but is not a consistent derivative
        on the first rows.  Intermediate values trade one against the
        other; the smallest weight that makes the one-step amplification
        matrix non-expansive is the useful setting.
    boundary_nyquist_real : float or None
        Nyquist real-part constraint for the boundary stencils.  The
        gradient block uses this value and the divergence block its
        negation -- the same relation the interior pair satisfies
        automatically (``c*_l = -c_{-l}`` flips the even part of the
        symbol).  Two things have to hold at once and they pull in
        opposite directions: the value must be far enough from zero that
        the boundary rows keep a non-vanishing symbol at Nyquist (or the
        vertical odd-even mode reappears as a *trapped* null mode in the
        first ``L`` rows, which nothing damps), and small enough that the
        closure stays spectrally neutral.  The neutral window for
        ``surface_reach=1`` runs to about ``-0.9``; ``-0.5`` sits in the
        middle of it.
    """

    def __init__(self, half_width=8, kmax_frac=0.56, nyquist_real=-2.0,
                 surface_reach=1, boundary_nyquist_real=-0.5,
                 adjoint_boundary_weight=0.0, project_adjoint=False):
        L = int(half_width)
        if L < 1:
            raise ValueError(f"half_width must be >= 1, got {L}")
        self.L = L
        self.kmax_frac = float(kmax_frac)
        self.nyquist_real = float(nyquist_real)
        self.surface_reach = int(surface_reach)
        self.boundary_nyquist_real = (
            None if boundary_nyquist_real is None
            else float(boundary_nyquist_real))
        self.adjoint_boundary_weight = float(adjoint_boundary_weight)
        self.project_adjoint = bool(project_adjoint)

        self.offsets = np.arange(-L, L + 1)
        self.coef = asymmetric_coefficients(
            self.offsets, self.kmax_frac, self.nyquist_real)
        # Divergence stencil: -D^T  =>  c*_l = -c_{-l}
        self.coef_adj = -self.coef[::-1].copy()

        # Boundary blocks: rows 0 .. L-1 below the surface, one for each
        # operator.  They differ only in the sign of the Nyquist
        # constraint, mirroring the interior pair.
        self.n_top = L
        self.n_win = 2 * L
        self.top_block = self._boundary_block(adjoint=False)
        # The divergence block spans twice as many output rows: the exact
        # negative transpose of the gradient block reaches that far.
        self.n_top_adj = 2 * L
        self.n_win_adj = 3 * L
        self.top_block_adj = self._divergence_block()
        # The blocks' non-zero entries per output row, as Python floats: the
        # compiled step writes them out as slices (see _splice_top).
        self._block_taps = {
            key: tuple(tuple((int(w), float(block[r, w])) for w in np.flatnonzero(block[r]))
                       for r in range(block.shape[0]))
            for key, block in (("top", self.top_block), ("top_adj", self.top_block_adj))}

        self._torch_cache = {}

    # -- construction ------------------------------------------------------

    def _boundary_block(self, adjoint):
        block = np.zeros((self.n_top, self.n_win), dtype=np.float64)
        for r in range(self.n_top):
            off, coef = self.boundary_stencil(r, adjoint=adjoint)
            for l, c in zip(off, coef):
                block[r, r + int(l)] = c
        return block

    def _dense(self, block, adjoint, m=None):
        """Dense matrix of the operator over the first ``m`` rows below the
        surface (row 0 == free surface)."""
        L = self.L
        m = m or (5 * L + 2)
        d = np.zeros((m, m), dtype=np.float64)
        base = self.coef_adj if adjoint else self.coef
        for r in range(m):
            if r >= self.n_top:
                for l, c in zip(self.offsets, base):
                    j = r + int(l)
                    if 0 <= j < m:
                        d[r, j] = c
            else:
                d[r, :self.n_win] = block[r, :]
        return d

    def _divergence_block(self):
        """Divergence block over rows ``0 .. 2L-1``, columns ``0 .. 3L-1``.

        Blends the consistent boundary stencils with the exact negative
        transpose of the gradient operator (see
        ``adjoint_boundary_weight``).  The gradient's surface row is
        dropped from the transpose because the free-surface condition
        replaces the stress rate there, so it never contributes to the
        energy balance.
        """
        L, w = self.L, self.adjoint_boundary_weight
        cons = self._dense(self._boundary_block(adjoint=True), adjoint=True)
        cons = cons[:self.n_top_adj, :self.n_win_adj]
        if w == 0.0 and not self.project_adjoint:
            return cons
        grad = self._dense(self.top_block, adjoint=False)
        grad[0, :] = 0.0
        adj = (-grad.T)[:self.n_top_adj, :self.n_win_adj]
        blend = (1.0 - w) * cons + w * adj
        if not self.project_adjoint:
            return blend
        return _project_moments(blend)

    def boundary_stencil(self, r, adjoint=False):
        """Offsets and coefficients for the row ``r`` cells below the surface."""
        hi = min(self.L, max(r + self.surface_reach, 1))
        off = np.arange(-r, hi + 1)
        # A narrow stencil cannot honour a 0.56-Nyquist fit; scale the band
        # with the stencil width, saturating where the full band is
        # attainable.
        n_full = min(2 * self.L + 1, 9)
        kmax = self.kmax_frac * min(1.0, off.size / float(n_full))
        nyq = self.boundary_nyquist_real
        if nyq is not None and adjoint:
            nyq = -nyq
        return off, asymmetric_coefficients(off, kmax, nyq)

    # -- diagnostics -------------------------------------------------------

    def max_symbol(self, n=4096):
        """Largest ``|symbol|`` of the interior operator over ``[0, pi]``.

        Sets the CFL limit: the 2-D semi-discrete eigenvalue bound is
        ``dt * v * max_symbol * sqrt(1/dx**2 + 1/dz**2) <= 2``.
        """
        theta = np.linspace(0.0, np.pi, int(n))
        return float(np.abs(symbol(self.coef, self.offsets, theta)).max())

    def cfl_dt(self, max_velocity, spacings):
        """Largest stable time step for the given velocity and spacings."""
        inv2 = sum(1.0 / float(s) ** 2 for s in spacings)
        return 2.0 / (float(max_velocity) * self.max_symbol() * np.sqrt(inv2))

    def boundary_report(self, adjoint=False):
        """Per-row summary of the boundary stencils (used by the tests)."""
        rows = []
        theta = np.linspace(1e-6, np.pi, 1024)
        for r in range(self.n_top):
            off, coef = self.boundary_stencil(r, adjoint=adjoint)
            s = symbol(coef, off, theta)
            rows.append({
                "row": r,
                "offsets": (int(off[0]), int(off[-1])),
                "npts": int(off.size),
                "sum": float(coef.sum()),
                "moment1": float(np.dot(coef, off)),
                "max_abs_symbol": float(np.abs(s).max()),
                "nyquist_symbol": float(np.abs(s[-1])),
                "max_abs_coef": float(np.abs(coef).max()),
            })
        return rows

    # -- torch application -------------------------------------------------

    def kernels(self, tensor_like):
        """The stencils as conv kernels on ``tensor_like``'s device and dtype:
        ``kz``/``kx`` (gradient), ``kz_adj``/``kx_adj`` (divergence), and the
        free-surface blocks ``top``/``top_adj``.  Built once per device and
        dtype, with their taps registered for the compiled step's slice form
        (:func:`sweep.operators.torch.register_stencil`)."""
        import torch
        from sweep.operators.torch import register_stencil

        # One string key: under torch.compile a tensor fetched from a dict
        # keyed by a tuple holding a ``torch.dtype`` reaches Dynamo without a
        # source, its registered taps cannot be read, and the graph breaks at
        # every derivative (torch 2.9).
        key = f"{tensor_like.device}|{tensor_like.dtype}"
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached

        def _t(a, shape):
            return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                                   device=tensor_like.device,
                                   dtype=tensor_like.dtype).reshape(shape)

        k = 2 * self.L + 1
        cached = {
            "kz": register_stencil(_t(self.coef, (1, 1, k, 1))),
            "kz_adj": register_stencil(_t(self.coef_adj, (1, 1, k, 1))),
            "kx": register_stencil(_t(self.coef, (1, 1, 1, k))),
            "kx_adj": register_stencil(_t(self.coef_adj, (1, 1, 1, k))),
            "top": _t(self.top_block, self.top_block.shape),
            "top_adj": _t(self.top_block_adj, self.top_block_adj.shape),
        }
        self._torch_cache[key] = cached
        return cached

    _torch = kernels

    @staticmethod
    def _conv(u, kernel):
        # ``apply_kernels`` is the zero-padded conv, or its slice form inside
        # a compiled step.
        from sweep.operators.torch import apply_kernels

        return apply_kernels(u, kernel)

    def _splice_top(self, out, u, block, surf, n_top, n_win, taps=None,
                    kernel=None):
        """Replace ``out[surf : surf+n_top]`` with ``block @ u[surf : surf+n_win]``.

        Inside a compiled step (``kernel``, the conv the interior rows came
        from, has its taps registered) the block -- ``taps``, its non-zero
        entries per row -- is written out row by row as shifted slices of
        ``u``, which fuse with the rest of the step; the einsum is a matmul
        Inductor cannot fuse."""
        import torch
        from sweep.operators.torch import _compiled_taps

        win = u[..., surf:surf + n_win, :]
        if win.shape[-2] != n_win:
            raise ValueError(
                f"grid too shallow for the free-surface stencil block: need "
                f"{n_win} rows below the surface, got {win.shape[-2]}")
        if taps is not None and _compiled_taps(kernel) is not None:
            rows = []
            for row_taps in taps[:n_top]:
                row = None
                for w, c in row_taps:
                    term = win[..., w:w + 1, :] * c
                    row = term if row is None else row + term
                rows.append(row)
            top = torch.cat(rows, dim=-2)
        else:
            top = torch.einsum("rw,bcwx->bcrx", block, win)
        return torch.cat([out[..., :surf, :], top,
                          out[..., surf + n_top:, :]], dim=-2)

    def _odd_extend(self, u, surf):
        """Reflect ``u`` anti-symmetrically about the surface row.

        Exact for the eta-flux ``J T_3j sigma_ij``, which is the surface
        traction and therefore vanishes on the surface row: the odd
        continuation reproduces its Taylor series through the quadratic
        term, so the full-width interior stencil stays usable right up to
        the surface.  This is the classical traction-image trick, which on
        a boundary-fitted grid costs nothing because the surface is the
        flat ``eta = 0`` row.
        """
        import torch

        if surf != self.L:
            raise ValueError(
                f"mirror closure expects the surface row at the stencil halo "
                f"({self.L}), got {surf}")
        ghost = -u[..., surf + 1:surf + self.L + 1, :].flip(-2)
        return torch.cat([ghost, u[..., surf:, :]], dim=-2)

    def _even_extend(self, u, surf):
        """Reflect ``u`` symmetrically about the surface row."""
        import torch

        if surf != self.L:
            raise ValueError(
                f"mirror closure expects the surface row at the stencil halo "
                f"({self.L}), got {surf}")
        ghost = u[..., surf + 1:surf + self.L + 1, :].flip(-2)
        return torch.cat([ghost, u[..., surf:, :]], dim=-2)

    def _apply_z(self, u, kernel, surf, mode, block_key="top",
                 n_top=None, n_win=None):
        cache_free = surf is None or mode == "zero"
        if cache_free:
            # ``zero``: the fields are held at zero above the surface, so the
            # plain convolution already implements "stencil points above the
            # free surface contribute zero".
            return self._conv(u, kernel)
        if mode == "shifted":
            return self._splice_top(
                self._conv(u, kernel), u, self._torch(u)[block_key], int(surf),
                n_top if n_top is not None else self.n_top,
                n_win if n_win is not None else self.n_win,
                taps=self._block_taps[block_key], kernel=kernel)
        if mode == "mirror_odd":
            return self._conv(self._odd_extend(u, int(surf)), kernel)
        if mode == "mirror_even":
            return self._conv(self._even_extend(u, int(surf)), kernel)
        raise ValueError(f"unknown surface closure {mode!r}")

    def d_x(self, u):
        """Gradient operator along the last (x) axis."""
        return self._conv(u, self._torch(u)["kx"])

    def d_x_adj(self, u):
        """Divergence operator along the last (x) axis (``-D_x^T``)."""
        return self._conv(u, self._torch(u)["kx_adj"])

    def d_z(self, u, surface_row=None, mode="shifted"):
        """Gradient operator along axis -2 (z / eta).

        ``surface_row`` is the index of the free-surface row in the padded
        runtime array; ``None`` disables the boundary treatment (no free
        surface, so the stencil halo is an ordinary PML halo).  ``mode``
        selects the closure -- see :meth:`_apply_z`.
        """
        return self._apply_z(u, self._torch(u)["kz"], surface_row, mode)

    def d_z_adj(self, u, surface_row=None, mode="shifted"):
        """Divergence operator along axis -2 (``-D_z^T`` in the interior)."""
        return self._apply_z(u, self._torch(u)["kz_adj"], surface_row, mode,
                             block_key="top_adj", n_top=self.n_top_adj,
                             n_win=self.n_win_adj)
