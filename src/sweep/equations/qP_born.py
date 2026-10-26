"""Born (LSRTM) equations of the pure-qP VTI / TTI operators of ``qP_vti`` /
``qP_tti`` (Liang et al. 2023, doi:10.1190/geo2022-0292.1, eqs 14/20, A-7).

Derivation (docs/user-guide/qp_born.md).  The operator of eq. 14 is linear in
P, with the anisotropy carried by S_k, a function of the wavenumber direction
only.  Write it as  P_tt = v^2 F[P]  and perturb v -> v + dv, P -> P0 + dP:

    P0_tt = v^2 F[P0] + f                         (background)
    dP_tt = v^2 F[dP] + m v^2 F[P0],   m = 2 dv / v   (scattered, first order)

so the scattered field obeys the SAME qP operator as the background -- its own
S_k, i.e. the direction of its own wavenumber -- driven by the reflectivity m
times the background's  v^2 F[P0] (= P0_tt away from the source).  The
space-domain operator replaces S_k by S_n(grad P) of the field it acts on, so
the background takes S_n(grad P0) and the scattered field S_n(grad dP).

That is NOT the linearisation of the discretised space-domain operator in P:
there S_n(grad(P0 + dP)) would contribute S_n(grad P0) + dS_n(grad P0)[grad dP],
which propagates the scattered field with the background's direction, and S_n
is homogeneous of degree 0 in P -- after the background wavefront has passed
(P0 ~ 0) that linearisation would carry dP with S = 0, an elliptical medium.
The Born operator here is the first-order model perturbation of eq. 14 instead.
It is homogeneous of degree one in m (scaling m scales the data) but, through
S_n(grad dP), not exactly additive; the gradient (autograd, impl='eager') is the
exact one of this forward map.

Fields: the background (h1 ... zetaz) and the scattered field (sh1 ... szetaz),
each with its own CPML memories.  Source on h1, receivers on sh1 by default, as
:class:`~sweep.equations.AcousticBorn`.
"""
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec, history_fields, record_single
from .fields import FieldSpec, ModelSpec
from ._registry import register_equation
from . import qP_tti as _tti
from . import qP_vti as _vti

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None
import torch


def _field_specs():
    bg = (
        FieldSpec("h1", aliases=("pressure", "p", "background"), description="Background qP pressure-like wavefield.", supports_source=True),
        FieldSpec("h2", aliases=("pressure_prev", "background_prev"), description="Previous-step background wavefield.", internal=True),
        FieldSpec("psix", description="Background CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="Background CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="Background CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="Background CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )
    sc = (
        FieldSpec("sh1", aliases=("scattered", "scattered_pressure", "data"), description="Scattered (Born) qP wavefield used for LSRTM data prediction.", supports_receiver=True),
        FieldSpec("sh2", aliases=("scattered_prev",), description="Previous-step scattered wavefield.", internal=True),
        FieldSpec("spsix", description="Scattered-wave CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("spsiz", description="Scattered-wave CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("szetax", description="Scattered-wave CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("szetaz", description="Scattered-wave CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )
    return bg + sc


def qp_born_cuda_layout():
    """``cuda_layout`` of the compiled Born pair (csrc/cuda/equations/qp2d,
    born_traits.cuh): two qP fields; the backward returns the mp gradient.

    * full: per step the scattered state (p + its four CPML memories) and the
      background update r0, ``(nt, 6, B, nz, nx)``;
    * boundary saving: both fields' CPML frames (10 fields), both rebuilt in
      the deep interior.
    """
    return CUDALayoutSpec(
        record_shape=record_single(),
        save_all_shape=history_fields(6),
        # background then scattered: u triple, psix, psiz, zetax, zetaz, psixn, psizn
        base_nvar=6,
        pml_nvar=12,
        last_two_nvar=2,
        last_two_storage_nvar=2,
        boundary_save_nvar=10,
        boundary_frame=True,
        bs_reconstruction_nvar=14,
        backward_workspace_nvar=4,
        grads_out_has_wavelet=True,
        # the wavelet gradient needs the background adjoint (refused, see
        # c_refuse_grads)
        grads_out_wavelet_written=False,
    )


class _QPBorn(SecondOrderEquation):
    """Shared step of the two Born classes."""

    # Same reasons as AcousticVTI / AcousticTTI.
    supports_free_surface = False
    supports_boundary_saving_eager = False
    C_HAS_RECURSIVE_CKPT = False

    default_pml_type = "cpmlr"
    FIELD_SPECS = _field_specs()

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2, s_gradient="adjoint"):
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
        super().init_grad_kernels()
        if backend == "jax" and jnp is None:
            raise ImportError(f"{type(self).__name__} with backend='jax' requires jax to be installed.")
        self.op = {"torch": torch, "jax": jnp}[backend]
        # as AcousticVTI: 'adjoint' (default) propagates each adjoint field with
        # its own S_n -- for LSRTM the back-propagated residual, which 'frozen'
        # and 'exact' would carry with the scattered field's direction (S = 0
        # where mp = 0, i.e. elliptically, at the first iteration)
        self.s_gradient = _vti.check_s_gradient(s_gradient, backend)

    @property
    def cuda_layout(self):
        return qp_born_cuda_layout()

    def c_eq_aux(self, prop):
        return _vti.s_gradient_aux(self)

    def check_stability(self, models, dt, spacing):
        """Warn once when ``dt`` breaks the scheme's stability condition
        (:func:`~sweep.equations.qP_vti.qp_stability_ratio`); ``PropTorch`` calls it before every forward."""
        _vti.check_qp_stability(self, models, dt, spacing)

    def c_refuse_grads(self, wavelet_grad, model_grad_mask, boundary_saving=False, boundary_dtype=None,
                       checkpoint=False):
        """impl='c' returns the reflectivity gradient only: the background
        models' and the wavelet's need the background adjoint."""
        _vti.refuse_boundary_saving(self, boundary_saving, boundary_dtype)
        asked = [n for n, g in zip(self.models, model_grad_mask) if g and n != "mp"]
        if wavelet_grad:
            asked.insert(0, "wavelet")
        if asked:
            raise NotImplementedError(
                f"impl='c' {type(self).__name__} computes the reflectivity (mp) gradient only; "
                f"{asked} require grad. Detach them (the LSRTM case), or use impl='eager', "
                "whose autograd returns every gradient.")

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["sh1"]

    def _rhs(self, fields, aniso, dt, h, b):
        raise NotImplementedError

    def func(self, wavefields, models, dt, h, b, **kwargs):
        *aniso, mp = models
        bg, sc = wavefields[:6], wavefields[6:]
        if self.s_gradient == "adjoint":
            fns = self._adjoint_fns(dt, h, b)
            rhs0, aux0 = _vti.adjoint_mode_step(*fns, bg, aniso, h, self.b, self.gradient, self.grad_kernels)
            rhs1, aux1 = _vti.adjoint_mode_step(*fns, sc, aniso, h, self.b, self.gradient, self.grad_kernels)
        else:
            rhs0, aux0 = self._rhs(bg, aniso, dt, h, b)
            rhs1, aux1 = self._rhs(sc, aniso, dt, h, b)
        u_next = 2 * bg[0] - bg[1] + rhs0
        su_next = 2 * sc[0] - sc[1] + rhs1 + mp * rhs0
        return (u_next, bg[0], *aux0, su_next, sc[0], *aux1)


@register_equation()
class AcousticVTIBorn(_QPBorn):
    """Born / LSRTM equation of the pure-qP VTI operator (:class:`AcousticVTI`).

    A background qP field ``h1`` in the smooth ``vp`` / ``epsilon`` /
    ``delta`` and a scattered qP field ``sh1`` driven by the reflectivity
    ``mp`` (``2 dv / v``) times the background's ``v^2 F[p0]``; each field
    evaluates the anisotropy operator S_n on its own gradient (the Born
    approximation of the wavenumber-domain eq. 14, see the module docstring).
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Background VTI reference velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon parameter."),
        ModelSpec("delta", description="Thomsen delta parameter."),
        ModelSpec("mp", aliases=("reflectivity", "ref"), description="Velocity reflectivity 2 dv / v of the Born scattering."),
    )

    C_NAME = "qp_vti_born2d"
    _adjoint_fns = _vti.AcousticVTI._adjoint_fns

    def _rhs(self, f, aniso, dt, h, b):
        vp, epsilon, delta = aniso
        hz, hx = self._spacings_2d(h)
        lap_z, lap_x = self.separable_d2_2d(f[0], self.laplace_kernels, hz, hx)
        return _vti.rhs_cpml(f[0], f[2], f[3], f[4], f[5], vp, epsilon, delta, dt, h, b, self.b,
                             lap_x, lap_z, self.gradient, self.grad_kernels,
                             s_frozen=self.s_gradient == "frozen")


@register_equation()
class AcousticTTIBorn(_QPBorn):
    """Born / LSRTM equation of the pure-qP TTI operator (:class:`AcousticTTI`).

    As :class:`AcousticVTIBorn`, with the tilt ``theta`` (radians).
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Background TTI reference velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon parameter."),
        ModelSpec("delta", description="Thomsen delta parameter."),
        ModelSpec("theta", description="Tilt angle parameter.", unit="rad"),
        ModelSpec("mp", aliases=("reflectivity", "ref"), description="Velocity reflectivity 2 dv / v of the Born scattering."),
    )

    C_NAME = "qp_tti_born2d"
    _adjoint_fns = _tti.AcousticTTI._adjoint_fns

    def _rhs(self, f, aniso, dt, h, b):
        vp, epsilon, delta, theta = aniso
        hz, hx = self._spacings_2d(h)
        lap_z, lap_x = self.separable_d2_2d(f[0], self.laplace_kernels, hz, hx)
        return _tti.rhs_cpml(f[0], f[2], f[3], f[4], f[5], vp, epsilon, delta, theta, dt, h, b, self.b,
                             lap_x, lap_z, self.gradient, self.op, self.grad_kernels,
                             s_frozen=self.s_gradient == "frozen")
