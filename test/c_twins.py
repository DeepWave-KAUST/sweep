"""Eager twins of the impl='c' forwards of the second-order acoustic family,
for checking that a c backward is the exact transpose of its c forward
(c_consistency_suite.py): if the c record matches the twin to fp32 round-off,
the c gradients must match autograd through the twin.

What a twin changes against the plain eager equation:

* ``halo0``: the outer M (halo) cells of every output field are zeroed after
  each step -- the c kernels never update them, eager's zero-padded stencils do.
* ``product_rule``: the CPML term d(a psi)/dx taken as da/dx * psi + a * dpsi/dx
  (the product rule of acoustic3d, acoustic_vrz2d/3d and acoustic_lsrtm2d/3d),
  where eager differentiates a * psi as one field (cpml_axis_update) like the
  2-D acoustic kernel.  Both are consistent CPMLs; they differ at O(h^2) inside
  the absorbing layer.  da/dx is the first-difference stencil of the 1-D
  profile, as the kernels' gradient<2, Order, X>(cpml.ax, ...).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sweep.scalars import fd_coefficients


def halo0(base):
    """``base`` with the outer M cells of every output field zeroed each step."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            out = list(base.func(self, wavefields, models, dt, h, b, **kw))
            M = self.so // 2
            nd = self.ndim
            ref = out[0]
            keep = torch.zeros(ref.shape[-nd:], dtype=torch.bool, device=ref.device)
            keep[tuple(slice(M, s - M) for s in ref.shape[-nd:])] = True
            for k, o in enumerate(out):
                if isinstance(o, torch.Tensor) and tuple(o.shape[-nd:]) == tuple(ref.shape[-nd:]):
                    out[k] = torch.where(keep, o, torch.zeros_like(o))
            return tuple(out)

    V.__name__ = base.__name__
    return V


def profile_gradient(p, so, h):
    """Central first difference of a 1-D CPML profile held broadcastable
    (one axis of length n, the others 1), zero beyond its ends."""
    shape = p.shape
    v = p.reshape(-1)
    M = so // 2
    c = torch.tensor(fd_coefficients(1, so), dtype=v.dtype, device=v.device)
    vp = F.pad(v[None, None], (M, M))[0, 0]
    n = v.numel()
    d = torch.zeros_like(v)
    for k in range(1, M + 1):
        d = d + c[k - 1] * (vp[M + k:M + k + n] - vp[M - k:M - k + n])
    return (d / h).reshape(shape)


def fd_grad(f, axis, so, h):
    """Order-``so`` central first difference of ``f`` along ``axis`` (negative,
    counted from the end), zero beyond the array -- the kernels' gradient<>
    stencil.  Eager Acoustic3D has no first-derivative kernels and falls back
    to torch.gradient (second order), so the twins never use self.gradient."""
    M = so // 2
    c = fd_coefficients(1, so)
    n = f.shape[axis]
    pad = [0, 0] * (-axis - 1) + [M, M]
    fp = F.pad(f, pad)
    out = torch.zeros_like(f)
    for k in range(1, M + 1):
        out = out + float(c[k - 1]) * (fp.narrow(axis, M + k, n) - fp.narrow(axis, M - k, n))
    return out / h


def pr_axis(lap, du, psi, zeta, a, b, db, da, dpsi):
    """One axis of the product-rule CPML: (contribution to w_sum, psi_next, zeta_next)."""
    tmp = ((1 + b) * lap + db * du) + (da * psi + a * dpsi)
    return (1 + b) * tmp + a * zeta, b * du + a * psi, b * tmp + a * zeta


def product_rule_vrz2d(base):
    """AcousticVRZ with the CPML in acoustic_vrz2d's product-rule form."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            u, u_pre, psix, psiz, zetax, zetaz = wavefields
            vp, z = models
            hz, hx = self._spacings_2d(h)
            lap_z, lap_x = self.separable_d2_2d(u, self.laplace_kernels, hz, hx)
            az, bz, dbzdz, ax, bx, dbxdx = self.b
            g = lambda f, axis: fd_grad(f, axis, self.so, hx if axis == -1 else hz)
            dudx, dudz = g(u, -1), g(u, -2)
            inv_z = 1.0 / z
            dbdx = g(vp, -1) * inv_z + vp * g(inv_z, -1)
            dbdz = g(vp, -2) * inv_z + vp * g(inv_z, -2)
            cx, psixn, zetaxn = pr_axis(lap_x, dudx, psix, zetax, ax, bx, dbxdx,
                                        profile_gradient(ax, self.so, hx), g(psix, -1))
            cz, psizn, zetazn = pr_axis(lap_z, dudz, psiz, zetaz, az, bz, dbzdz,
                                        profile_gradient(az, self.so, hz), g(psiz, -2))
            rhs = (vp * z) * ((vp * inv_z) * (cx + cz) + dbdx * (dudx + psixn) + dbdz * (dudz + psizn))
            return 2 * u - u_pre + dt * dt * rhs, u, psixn, psizn, zetaxn, zetazn

    V.__name__ = base.__name__
    return V


def product_rule_vrz3d(base):
    """AcousticVRZ3D with the CPML in acoustic_vrz3d's product-rule form."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            u, u_pre, psix, psiy, psiz, zetax, zetay, zetaz = wavefields
            vp, z = models
            hz, hy, hx = self._spacings_3d(h)
            lap_z, lap_y, lap_x = self.separable_d2_3d(u, self.laplace_kernels, hz, hy, hx)
            az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = self.b
            g = lambda f, axis: fd_grad(f, axis, self.so, {-1: hx, -2: hy, -3: hz}[axis])
            dudx, dudy, dudz = g(u, -1), g(u, -2), g(u, -3)
            inv_z = 1.0 / z
            dbdx = g(vp, -1) * inv_z + vp * g(inv_z, -1)
            dbdy = g(vp, -2) * inv_z + vp * g(inv_z, -2)
            dbdz = g(vp, -3) * inv_z + vp * g(inv_z, -3)
            cx, psixn, zetaxn = pr_axis(lap_x, dudx, psix, zetax, ax, bx, dbxdx,
                                        profile_gradient(ax, self.so, hx), g(psix, -1))
            cy, psiyn, zetayn = pr_axis(lap_y, dudy, psiy, zetay, ay, by, dbydy,
                                        profile_gradient(ay, self.so, hy), g(psiy, -2))
            cz, psizn, zetazn = pr_axis(lap_z, dudz, psiz, zetaz, az, bz, dbzdz,
                                        profile_gradient(az, self.so, hz), g(psiz, -3))
            rhs = (vp * z) * ((vp * inv_z) * (cx + cy + cz) + dbdx * (dudx + psixn)
                             + dbdy * (dudy + psiyn) + dbdz * (dudz + psizn))
            return 2 * u - u_pre + dt * dt * rhs, u, psixn, psiyn, psizn, zetaxn, zetayn, zetazn

    V.__name__ = base.__name__
    return V


def product_rule_acoustic3d(base):
    """Acoustic3D with the CPML in acoustic3d's product-rule form."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            u, u_pre, psix, psiy, psiz, zetax, zetay, zetaz = wavefields
            (vp,) = models
            hz, hy, hx = self._spacings_3d(h)
            lap_z, lap_y, lap_x = self.separable_d2_3d(u, self.laplace_kernels, hz, hy, hx)
            az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = self.b
            g = lambda f, axis: fd_grad(f, axis, self.so, {-1: hx, -2: hy, -3: hz}[axis])
            cx, psixn, zetaxn = pr_axis(lap_x, g(u, -1), psix, zetax, ax, bx, dbxdx,
                                        profile_gradient(ax, self.so, hx), g(psix, -1))
            cy, psiyn, zetayn = pr_axis(lap_y, g(u, -2), psiy, zetay, ay, by, dbydy,
                                        profile_gradient(ay, self.so, hy), g(psiy, -2))
            cz, psizn, zetazn = pr_axis(lap_z, g(u, -3), psiz, zetaz, az, bz, dbzdz,
                                        profile_gradient(az, self.so, hz), g(psiz, -3))
            return (2 * u - u_pre + vp ** 2 * dt ** 2 * (cx + cy + cz), u,
                    psixn, psiyn, psizn, zetaxn, zetayn, zetazn)

    V.__name__ = base.__name__
    return V


def product_rule_lsrtm2d(base):
    """AcousticLSRTM with both fields' CPML in acoustic_lsrtm2d's product-rule
    form (acoustic_cpml_update_2d): sc' = ... + dt^2 (v^2 W(sc) + mp v^2 W(bg))."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            (u, u_pre, psix, psiz, zetax, zetaz,
             su, su_pre, spsix, spsiz, szetax, szetaz) = wavefields
            vp, mp = models
            hz, hx = self._spacings_2d(h)
            az, bz, dbzdz, ax, bx, dbxdx = self.b
            dax, daz = profile_gradient(ax, self.so, hx), profile_gradient(az, self.so, hz)
            g = lambda f, axis: fd_grad(f, axis, self.so, hx if axis == -1 else hz)

            def field(w, w_psix, w_psiz, w_zetax, w_zetaz):
                lap_z, lap_x = self.separable_d2_2d(w, self.laplace_kernels, hz, hx)
                cx, pxn, zxn = pr_axis(lap_x, g(w, -1), w_psix, w_zetax, ax, bx, dbxdx, dax, g(w_psix, -1))
                cz, pzn, zzn = pr_axis(lap_z, g(w, -2), w_psiz, w_zetaz, az, bz, dbzdz, daz, g(w_psiz, -2))
                return cx + cz, pxn, pzn, zxn, zzn

            w_bg, psixn, psizn, zetaxn, zetazn = field(u, psix, psiz, zetax, zetaz)
            w_sc, spsixn, spsizn, szetaxn, szetazn = field(su, spsix, spsiz, szetax, szetaz)
            bg_utt = vp * vp * w_bg
            sc_utt = vp * vp * w_sc
            return (2 * u - u_pre + dt * dt * bg_utt, u, psixn, psizn, zetaxn, zetazn,
                    2 * su - su_pre + dt * dt * (sc_utt + mp * bg_utt), su, spsixn, spsizn, szetaxn, szetazn)

    V.__name__ = base.__name__
    return V


def product_rule_lsrtm3d(base):
    """AcousticLSRTM3D with both fields' CPML in acoustic_lsrtm3d's product-rule
    form (acoustic_cpml_update_3d): sc' = ... + dt^2 (v^2 W(sc) + mp v^2 W(bg))."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            (u, u_pre, psix, psiy, psiz, zetax, zetay, zetaz,
             su, su_pre, spsix, spsiy, spsiz, szetax, szetay, szetaz) = wavefields
            vp, mp = models
            hz, hy, hx = self._spacings_3d(h)
            az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = self.b
            dax = profile_gradient(ax, self.so, hx)
            day = profile_gradient(ay, self.so, hy)
            daz = profile_gradient(az, self.so, hz)
            g = lambda f, axis: fd_grad(f, axis, self.so, {-1: hx, -2: hy, -3: hz}[axis])

            def field(w, w_psix, w_psiy, w_psiz, w_zetax, w_zetay, w_zetaz):
                lap_z, lap_y, lap_x = self.separable_d2_3d(w, self.laplace_kernels, hz, hy, hx)
                cx, pxn, zxn = pr_axis(lap_x, g(w, -1), w_psix, w_zetax, ax, bx, dbxdx, dax, g(w_psix, -1))
                cy, pyn, zyn = pr_axis(lap_y, g(w, -2), w_psiy, w_zetay, ay, by, dbydy, day, g(w_psiy, -2))
                cz, pzn, zzn = pr_axis(lap_z, g(w, -3), w_psiz, w_zetaz, az, bz, dbzdz, daz, g(w_psiz, -3))
                return cx + cy + cz, pxn, pyn, pzn, zxn, zyn, zzn

            w_bg, psixn, psiyn, psizn, zetaxn, zetayn, zetazn = field(u, psix, psiy, psiz, zetax, zetay, zetaz)
            w_sc, spsixn, spsiyn, spsizn, szetaxn, szetayn, szetazn = field(
                su, spsix, spsiy, spsiz, szetax, szetay, szetaz)
            bg_utt = vp * vp * w_bg
            sc_utt = vp * vp * w_sc
            return (2 * u - u_pre + dt * dt * bg_utt, u, psixn, psiyn, psizn, zetaxn, zetayn, zetazn,
                    2 * su - su_pre + dt * dt * (sc_utt + mp * bg_utt), su,
                    spsixn, spsiyn, spsizn, szetaxn, szetayn, szetazn)

    V.__name__ = base.__name__
    return V
