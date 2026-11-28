"""Pseudoelastic pure P-mode wave equation (Sun & Alkhalifah 2021), 2-D, eager only."""
from __future__ import annotations

import torch

from ._elastic_step_core import elastic_stress_substep, elastic_velocity_substep
from ._registry import register_equation
from .elastic import Elastic


@register_equation(aliases=("PureP",))
class PseudoelasticP(Elastic):
    """Pseudoelastic pure P-mode wave equation of Sun & Alkhalifah (2021): the
    2-D isotropic elastic velocity-stress equations in which the particle
    velocity is projected onto its curl-free (P) part after every velocity
    update.

    Writing the normalised gradient :math:`\\tilde\\nabla = \\nabla/\\sqrt{-\\Delta}`,
    the paper's first-order form (their eq. 8) is

    .. math::

        \\partial_t P = \\tilde\\nabla\\cdot\\big[(\\nabla\\cdot\\boldsymbol\\tau + \\mathbf f)/\\rho\\big],
        \\qquad
        \\partial_t \\boldsymbol\\tau = \\mathbf C : \\nabla(\\tilde\\nabla P),

    i.e. the velocity that drives the stresses is :math:`\\mathbf v = \\tilde\\nabla P
    = \\tilde\\nabla\\tilde\\nabla\\cdot\\mathbf v`, the irrotational part of the
    elastic velocity.  The equation carries no S wave and no P-S or S-P
    conversion, so it runs on the P-wave grid (``vs`` does not set the grid
    spacing), while the PP reflections keep the full elastic amplitude
    variation with angle (``vs`` enters through the shear modulus in the stress
    update), including post-critical reflections -- unlike a first-order
    (Born) correction of the acoustic equation.

    Implementation: :class:`Elastic`'s staggered-grid step (same stencils,
    CPML, sources, receivers and the image-method free surface) with the
    projection inserted between the velocity and the stress sub-steps.  The
    projection is spectral (two 2-D FFTs each way over the whole padded grid)
    and is built from the symbol of the staggered FD derivative, so a discrete
    FD gradient field is left exactly unchanged: with ``vs = 0`` the equation
    reproduces ``Elastic(vs = 0)`` to round-off.  With ``free_surface=True`` the
    grid is mirrored about the surface row inside the projection (``vx`` odd,
    ``vz`` even), which keeps the sea surface traction-free; this requires water
    (``vs = 0``) along the surface row, as in marine acquisition.  The
    projection reads the whole grid, so boundary saving is refused and there is
    no compiled (``impl='c'``) path; use ``impl='eager'`` with the default
    checkpointing.  Only the top face may be a free surface, and only flat.

    Reference: B. Sun and T. Alkhalifah, 2021, *Pseudoelastic pure P-mode wave
    equation*, Geophysics 86(6), T469-T485,
    [10.1190/geo2021-0084.1](https://doi.org/10.1190/geo2021-0084.1).
    """
    C_NAME = None                                 # eager only
    supports_per_edge_free_surface = False        # top-only mirror in the projection
    supports_image_topography = False
    supports_boundary_saving_eager = False        # the projection reads the whole grid
    supports_boundary_saving_c = False
    supports_batched_models = False

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Build the pure P-mode operator.

        Args:
            spatial_order: order of the staggered FD derivatives of the
                elastic step (even; default 4).  The projection uses the
                symbol of this stencil.
            device: device of the stencil kernels (``'cpu'`` or ``'cuda'``).
            backend: ``'torch'`` only.
        """
        if backend != 'torch':
            raise NotImplementedError("PseudoelasticP runs on the torch eager backend only.")
        super().__init__(spatial_order, device, backend)
        self._proj_cache = {}

    # ------------------------------------------------------------------ projection
    def _projector(self, nz, nx, hz, hx, device):
        """The orthogonal projector onto the range of the discrete staggered
        gradient ``D = (D+x, D+z)`` in the wavenumber domain,
        ``P = D (D^H D)^-1 D^H``: the discrete ``grad~ div~``.  ``D`` is the
        exact symbol of the propagator's stencil (the FFT of its impulse
        response, staggering shift included), so an FD gradient field
        ``(D+x p, D+z p)`` is invariant.  Returns ``(Pxx, Pxz, Pzx, Pzz)``,
        complex tensors of shape (nz, nx)."""
        key = (nz, nx, float(hz), float(hx), str(device))
        hit = self._proj_cache.get(key)
        if hit is not None:
            return hit
        with torch.no_grad():
            delta = torch.zeros(1, 1, nz, nx, device=device)
            cz, cx = nz // 2, nx // 2
            delta[0, 0, cz, cx] = 1.0
            Dx = torch.fft.fft2(torch.roll(self.pd.x_forward(delta)[0, 0], (-cz, -cx), (0, 1)).double())
            Dz = torch.fft.fft2(torch.roll(self.pd.z_forward(delta)[0, 0], (-cz, -cx), (0, 1)).double())
            D2 = (Dx.abs() ** 2 + Dz.abs() ** 2)
            D2[0, 0] = 1.0
            Pxx = Dx * Dx.conj() / D2
            Pxz = Dx * Dz.conj() / D2
            Pzx = Dz * Dx.conj() / D2
            Pzz = Dz * Dz.conj() / D2
            for a in (Pxx, Pxz, Pzx, Pzz):
                a[0, 0] = 0.0                                # no mean velocity
            out = tuple(a.to(torch.complex64) for a in (Pxx, Pxz, Pzx, Pzz))
        self._proj_cache[key] = out
        return out

    def _project(self, vx, vz, hz, hx):
        """``(vx, vz) -> grad~ div~ (vx, vz)`` on the padded grid; with the
        image-method free surface the grid is first mirrored about the surface
        row (``vx`` odd, ``vz`` even about the plane of the normal stresses)."""
        fs = bool(getattr(self, "free_surface", False))
        if fs:
            s = int(self.pd.coes.shape[0])                 # surface row (normal stresses)
            bx, bz = vx[..., s:, :], vz[..., s:, :]
            L = bx.shape[-2]
            vx_in = torch.cat([-bx[..., 1:, :].flip(-2), bx], -2)       # 2L-1 rows
            vz_in = torch.cat([bz[..., :-1, :].flip(-2), bz], -2)       # 2L-1 rows
        else:
            vx_in, vz_in = vx, vz
        nz, nx = vx_in.shape[-2:]
        Pxx, Pxz, Pzx, Pzz = self._projector(nz, nx, hz, hx, vx.device)
        Fx = torch.fft.fft2(vx_in)
        Fz = torch.fft.fft2(vz_in)
        vx_out = torch.fft.ifft2(Pxx * Fx + Pxz * Fz).real
        vz_out = torch.fft.ifft2(Pzx * Fx + Pzz * Fz).real
        if fs:
            i0 = L - 1 - s                                  # physical row 0 in the mirrored grid
            vx_out = vx_out[..., i0: i0 + vx.shape[-2], :]
            vz_out = vz_out[..., i0: i0 + vz.shape[-2], :]
        return vx_out, vz_out

    # ------------------------------------------------------------------ step
    def prepare_models(self, models):
        vp, vs, rho, lam, mu, lam2mu = self._lame(models)
        if bool(getattr(self, "free_surface", False)):
            s = int(self.pd.coes.shape[0])
            if bool(torch.any(vs[..., s, :] != 0)):
                raise ValueError(
                    "PseudoelasticP: free_surface=True needs water (vs = 0) along the surface row; "
                    "the mirror used by the projection is exact only for a fluid top.")
        return [vp, vs, rho, lam, mu, lam2mu]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu = self._lame(models)
        free_surface = getattr(self, "free_surface", False)
        kw = dict(
            lame_lambda=lame_lambda, lame_mu=lame_mu, mu_xz=lame_mu, rho_x=rho, rho_z=rho,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=free_surface, fs_faces=getattr(self, "fs_faces", None),
            lame_lambda_2mu=lame_lambda_2mu, invariants=kwargs.get("invariants"),
        )
        state = list(elastic_velocity_substep(*wavefields, **kw))
        hz, hx = self._spacings_2d(h)
        state[0], state[1] = self._project(state[0], state[1], hz, hx)
        out = list(elastic_stress_substep(*state, **kw))
        if free_surface:
            out[3], out[4], out[2] = self._fs_zero_traction(out[3], out[4], out[2])
        return tuple(out)

    def interior_substeps(self):
        raise NotImplementedError(
            "PseudoelasticP has no reversible interior sub-steps: the P projection reads the whole grid. "
            "Boundary saving is refused; use the default checkpointing (memory=Ckpt) or Full.")
