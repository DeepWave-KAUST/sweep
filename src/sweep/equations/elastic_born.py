"""Born (single-scattering) modelling for the 2-D first-order elastic equation.

The model is split into a background ``(vp, vs, rho)`` and a perturbation
``(dvp, dvs, drho)``; the wavefield into the background field of
:class:`~sweep.equations.elastic.Elastic` and a scattered field driven by the
perturbation acting on it (Li, Guo, Li & Alkhalifah 2019, eq. 5, here in
velocity-stress form)::

    rho  d(dv)/dt   = div(dsigma) - drho dv0/dt
    d(dsigma)/dt    = C : eps(dv) + dC : eps(v0)

with ``dC`` the Lamé perturbation ``(dlambda, dmu)`` of ``(dvp, dvs, drho)``
linearised at the background.  The scattered step is the exact Jacobian of
the discrete :class:`Elastic` step -- same staggered stencil, CPML, free
surface and Robertsson surface coefficient -- so the scattered record equals
the directional derivative of the Elastic record along the perturbation.

Reference: Y. Li, Q. Guo, Z. Li and T. Alkhalifah, 2019, *Elastic reflection
waveform inversion with variable density*, Geophysics 84(4), R553-R567,
[10.1190/geo2017-0722.1](https://doi.org/10.1190/geo2017-0722.1).
"""

import dataclasses

from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec
from ._elastic_step_core import (
    _fs_sides,
    elastic_velocity_substep,
    elastic_stress_substep,
)
from ._free_surface import FS_MOD_SXX, overwrite_at_topo, overwrite_surface_row
from ._registry import register_equation
from .elastic import Elastic

_NF = len(Elastic.FIELD_SPECS)   # 15: 5 physical fields + 10 CPML memories


def _scattered_spec(spec):
    """The scattered-field twin of one :class:`Elastic` field spec."""
    if spec.internal:
        return FieldSpec(
            "s" + spec.name,
            description=f"Scattered-field {spec.description[0].lower()}{spec.description[1:]}",
            internal=True, boundary_related=spec.boundary_related, cpml_axis=spec.cpml_axis,
        )
    return FieldSpec(
        "s" + spec.name,
        aliases=("scattered_" + spec.name,),
        description=f"Scattered {spec.description[0].lower()}{spec.description[1:]}",
        supports_source=False, supports_receiver=spec.supports_receiver,
    )


@register_equation()
class ElasticBorn(FirstOrderEquation):
    """Born modelling for the 2-D first-order elastic equation (velocity-stress,
    staggered grid).

    Two coupled copies of :class:`Elastic`: the background field
    ``(vx, vz, sxx, szz, sxz)`` propagating through ``(vp, vs, rho)``, and the
    scattered field ``(svx, svz, ssxx, sszz, ssxz)`` driven by the perturbation
    ``(dvp, dvs, drho)`` acting on the background field.  Each field has its
    own ten CPML memories.  Defaults: source on the background stresses
    (explosion), receivers on the scattered velocities -- the layout of elastic
    least-squares RTM and reflection waveform inversion.

    The perturbations are absolute (same units as the background).  Their
    gradients are the elastic LSRTM images; the background gradients are the
    reflection-waveform-inversion (RWI) ones of Li et al. (2019).

    With ``impl='c'`` the backward is the :class:`Elastic` one: the scattered
    field's adjoint imaged against the background field, which is exactly the
    gradient of ``(dvp, dvs, drho)``, in every memory mode.  The gradients of
    the background ``(vp, vs, rho)`` and of the wavelet are ``impl='eager'``
    only (autograd); the compiled path refuses them.

    Reference: Y. Li, Q. Guo, Z. Li and T. Alkhalifah, 2019, *Elastic
    reflection waveform inversion with variable density*, Geophysics 84(4),
    R553-R567, [10.1190/geo2017-0722.1](https://doi.org/10.1190/geo2017-0722.1).
    """

    MODEL_SPECS = Elastic.MODEL_SPECS + (
        ModelSpec("dvp", aliases=("vp_perturbation",), description="P-wave velocity perturbation.", unit="m/s"),
        ModelSpec("dvs", aliases=("vs_perturbation",), description="S-wave velocity perturbation.", unit="m/s"),
        ModelSpec("drho", aliases=("rho_perturbation", "density_perturbation"), description="Density perturbation.", unit="kg/m^3"),
    )
    FIELD_SPECS = Elastic.FIELD_SPECS + tuple(_scattered_spec(s) for s in Elastic.FIELD_SPECS)

    C_NAME = "elastic_born2d"

    default_pml_type = "cpmls"
    supports_per_edge_free_surface = True
    supports_per_edge_free_surface_c = True
    supports_bs_free_surface = True
    supports_batched_models = True
    supports_image_topography = True
    supports_image_topography_c = True

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Build the elastic Born equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative
                operator, as :class:`Elastic`. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Defaults to ``'cpu'``.
            backend: ``'torch'`` or ``'jax'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend)

    @property
    def default_source_fields(self):
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["svx", "svz"]

    @staticmethod
    def _born_models(models):
        """Normalise the model list to the 10-tuple
        ``(vp, vs, rho, λ, μ, λ+2μ, drho, δλ, δμ, δλ+2δμ)``.

        Accepts the six raw models ``(vp, vs, rho, dvp, dvs, drho)`` or the
        prepared 10-tuple (returned as-is)."""
        n = len(models)
        if n == 10:
            return tuple(models)
        if n != 6:
            raise ValueError(f"ElasticBorn models must be a 6- or 10-tuple, got {n}")
        vp, vs, rho, dvp, dvs, drho = models
        lam = rho * (vp ** 2 - 2 * vs ** 2)
        mu = rho * vs ** 2
        dlam = drho * (vp ** 2 - 2 * vs ** 2) + rho * (2 * vp * dvp - 4 * vs * dvs)
        dmu = drho * vs ** 2 + 2 * rho * vs * dvs
        return vp, vs, rho, lam, mu, lam + 2 * mu, drho, dlam, dmu, dlam + 2 * dmu

    def prepare_models(self, models):
        return list(self._born_models(models))

    def _kw(self, models, dt, h, b, pml):
        _, _, rho, lam, mu, lam2mu = self._born_models(models)[:6]
        return dict(
            lame_lambda=lam, lame_mu=mu, mu_xz=mu, rho_x=rho, rho_z=rho,
            dt=dt, h=h, b=b, pd=self.pd, pml=pml,
            free_surface=bool(getattr(self, "free_surface", False)),
            topo_rows=getattr(self, "_topo_rows_runtime", None),
            fs_faces=getattr(self, "fs_faces", None),
            lame_lambda_2mu=lam2mu,
        )

    def _velocity_substep(self, wf, models, kw):
        """Both velocity updates; the scattered one gets ``-drho dv0/dt``,
        written as ``-(drho/rho) (v0_new - v0_old)`` -- the exact derivative
        of the discrete update in rho, and its own inverse at ``-dt``."""
        bg, sc = wf[:_NF], wf[_NF:]
        bg1 = elastic_velocity_substep(*bg, **kw)
        sc1 = list(elastic_velocity_substep(*sc, **kw))
        rho, drho = models[2], models[6]
        r = drho / rho
        sc1[0] = sc1[0] - r * (bg1[0] - bg[0])
        sc1[1] = sc1[1] - r * (bg1[1] - bg[1])
        return [*bg1, *sc1]

    def _stress_substep(self, wf, models, kw):
        """Both stress updates; the scattered one gets ``dt dC : eps(v0)``
        with the background's CPML-corrected strain rates.  At a free surface
        the Robertsson surface row of sigma_xx (sigma_zz on x faces) uses
        ``c = 4 μ (λ+μ)/(λ+2μ)``; its scattered source is ``dt dc vx_x``."""
        bg, sc = wf[:_NF], wf[_NF:]
        *bg2, (vx_x, vz_z, vx_z, vz_x) = elastic_stress_substep(*bg, **kw, return_gradients=True)
        sc2 = list(elastic_stress_substep(*sc, **kw))
        dt = kw["dt"]
        _, _, _, lam, mu, lam2mu, _, dlam, dmu, dlam2mu = models
        sxx_s, szz_s = sc2[2], sc2[3]
        sc2[2] = sxx_s + dt * (dlam2mu * vx_x + dlam * vz_z)
        sc2[3] = szz_s + dt * (dlam2mu * vz_z + dlam * vx_x)
        sc2[4] = sc2[4] + dt * dmu * (vx_z + vz_x)

        topo = kw["topo_rows"]
        has_topo = kw["free_surface"] and topo is not None
        z_sides, x_sides = _fs_sides(kw["fs_faces"], kw["free_surface"])
        if FS_MOD_SXX and (has_topo or z_sides or x_sides):
            # d/dλ, d/dμ of c = 4 μ (λ+μ)/(λ+2μ)
            den = lam2mu * lam2mu
            dc = (4 * mu * mu * dlam + 4 * (lam * lam + 2 * lam * mu + 2 * mu * mu) * dmu) / den
            halo = self.pd.coes.shape[0]
            if has_topo or z_sides:
                surf = sxx_s + dt * dc * vx_x
                if has_topo:
                    sc2[2] = overwrite_at_topo(sc2[2], surf, topo, axis=-2)
                else:
                    for side in z_sides:
                        sc2[2] = overwrite_surface_row(sc2[2], surf, halo, axis=-2, side=side)
            if x_sides:
                surf = szz_s + dt * dc * vz_z
                for side in x_sides:
                    sc2[3] = overwrite_surface_row(sc2[3], surf, halo, axis=-1, side=side)

        out = [*bg2, *sc2]
        if kw["free_surface"]:
            for o in (0, _NF):
                out[o + 3], out[o + 4], out[o + 2] = self._fs_zero_traction(
                    out[o + 3], out[o + 4], out[o + 2], topo)
        return out

    _fs_zero_traction = Elastic._fs_zero_traction

    def func(self, wavefields, models, dt, h, b, **kwargs):
        models = self._born_models(models)
        kw = self._kw(models, dt, h, b, self.b)
        wf = self._velocity_substep(list(wavefields), models, kw)
        return tuple(self._stress_substep(wf, models, kw))

    def interior_substeps(self):
        """The two half-steps of :meth:`func` (background and scattered
        together) with ``pml`` zeroed and the free-surface BC kept, for the
        eager boundary-saving reverse driver: composed in reverse order at
        ``-dt`` they invert the step exactly."""
        pml_zero = (0.0,) * 8

        def sub_v(wf, models, dt, h):
            models = self._born_models(models)
            return self._velocity_substep(list(wf), models, self._kw(models, dt, h, None, pml_zero))

        def sub_s(wf, models, dt, h):
            models = self._born_models(models)
            return self._stress_substep(list(wf), models, self._kw(models, dt, h, None, pml_zero))

        return [sub_v, sub_s]

    @property
    def cuda_layout(self):
        """:class:`Elastic`'s layout with the scattered field added to the
        forward: bind order [bg vx..sxz][sc vx..sxz][bg 10 memories][sc 10
        memories] (csrc/cuda/equations/elastic_born2d/driver_traits.cuh).
        Everything the backward binds -- adjoint fields, checkpoint snapshots
        and replay state, boundary strips, last state, history -- is the
        background's, in Elastic's layout: the backward is elastic2d's."""
        el = Elastic.cuda_layout.fget(self)
        return dataclasses.replace(
            el,
            base_nvar=10,
            pml_nvar=20,
            pml_slot_axes=el.pml_slot_axes * 2,
            backward_state_slots=tuple(range(5)) + tuple(range(10, 20)),
            checkpoint_nvar=15,
            boundary_save_nvar=5,
            bs_reconstruction_nvar=el.reconstruction_nvar,
            slots=None,
            # not stepped: domain decomposition does not drive this driver
            stepped=False,
            dd_backward_phases=False,
        )

    def c_refuse_grads(self, wavelet_grad, model_grad_mask, boundary_saving=False,
                       boundary_dtype=None, checkpoint=False):
        """impl='c' returns the (dvp, dvs, drho) gradients only."""
        background = [name for name, g in zip(("vp", "vs", "rho"), model_grad_mask) if g]
        if wavelet_grad or background:
            what = ", ".join(background + (["the wavelet"] if wavelet_grad else []))
            raise NotImplementedError(
                f"impl='c' ElasticBorn returns the gradients of dvp, dvs and drho only, not of "
                f"{what} (the background adjoint of the Born system). Detach them, or use "
                "impl='eager', whose autograd returns every gradient.")
