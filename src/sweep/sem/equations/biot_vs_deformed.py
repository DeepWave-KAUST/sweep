"""1st-order velocity-stress Biot poroelastic SEM on a Quad9 deformed mesh.

The low-frequency Biot equations in the form of the FD :class:`sweep.equations.Biot`
(solid velocity ``v``, Darcy flux ``q = phi (v_f - v)``, total stress ``sigma``, pore
pressure ``p``; Biot 1962, Carcione *Wave Fields in Real Media* ch. 7)::

    rho  dv/dt + rho_f dq/dt         = div sigma
    rho_f dv/dt + m dq/dt + B q      = -grad p
    d sigma/dt = (H - 2 mu) div v I + mu (grad v + grad v^T) + C div q I
    d p/dt     = -C div v - M div q

with the coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` of
:func:`sweep.equations.biot.biot_from_rock`; ``B = b I`` here, a tensor ``eta kappa^-1`` in
:class:`BiotAnisoPermSEM2DDeformed`.

Discretisation
--------------
The solid is :class:`ElasticVSSEM2DDeformed`'s: ``v_x, v_z`` continuous (CG) nodal
fields, the total stress element-local, the momentum equation in weak form with a lumped
mass. The pore fluid is the pressure form of the acoustic SEM: ``p`` is a CG nodal field
whose equation is taken in weak form, and the Darcy flux ``q_x, q_z`` is element-local,
driven by the strong gradient of ``p``::

    M_p dp/dt = -A(w alpha div v) + D^T W q,      M_p = A(w / M),  alpha = C / M,
    sigma     = sigma' - alpha p I,               d sigma'/dt = lambda_d div v I + mu (grad v + grad v^T)

(``A`` assembles, ``D^T W`` is the weak divergence), so that a pore fluid on its own
(``rho_f = C = 0``) is exactly the acoustic pressure SEM. The other placement -- ``q``
continuous and ``p`` element-local, as in SPECFEM2D's poroelastic solver -- leaves the
relative flow, which has no shear stiffness, on a continuous field: a point source then
excites grid-scale modes that run along the grid lines through it, and the pore pressure
there grows like ``1 / dh`` under refinement. The two couplings (``alpha p`` in the
momentum, ``alpha div v`` in the pressure) are transposes of each other, as are ``D^T W q``
and the strong ``grad p``, so the semi-discrete system conserves the Biot energy up to the
viscous drag.

The two momentum equations meet at every node: the solid's (one, CG) and the element
copies of the flux there (one per element sharing the node). Eliminating ``dv/dt`` leaves,
per node, ``A dq/dt + W_B q = r`` over the copies (and both components when ``B`` is a
tensor), with ``A = diag(w m) - w rho_f (w rho_f)^T / M_rho``; with the forces frozen over
the step this is integrated exactly, ``q <- E q + Psi r`` (``E`` and ``Psi`` from one
matrix exponential per node, computed in :meth:`init_state` for the run's ``dt``), and
then ``v <- v + (dt F - sum w rho_f (q_new - q)) / M_rho``. So any permeability is stable,
the time step is that of the inviscid fast P wave (see
:func:`sweep.equations.biot.biot_fast_p_velocity`), and ``b -> inf`` gives the undrained
elastic medium.

Viscous attenuation (``viscous_q=(Q0, f0)``, SPECFEM2D's ``ATTENUATION_PORO_FLUID_PART``):
the drag becomes a standard linear solid, ``B (1 + i w tau_e) / (1 + i w tau_s)`` with
``tau_e, tau_s = (sqrt(Q0^2 + 1) +- 1) / (2 pi f0 Q0)`` -- ``B`` at low frequency,
``B tau_e / tau_s`` at high frequency, ``1 / Q0`` at ``f0``. The unrelaxed drag goes into
the exact step and a memory ``r``, ``dr/dt = -(r + (1 - tau_e / tau_s) B q) / tau_s``, acts
as a force on the flux.

Fluid and elastic regions
-------------------------
An element without a solid frame (``rho = 0``) is a fluid, one without a pore fluid
(``M = 0``) a dry elastic solid -- see :func:`sweep.equations.biot_params.biot_fluid` and
:func:`~sweep.equations.biot_params.biot_elastic`; every other coefficient that does not
exist there must be zero, and an element is wholly one medium. In a fluid ``q`` is the
fluid velocity and ``m`` the fluid density (the equations above with ``rho = rho_f = H =
mu = C = b = 0``), so the fluid is the acoustic pressure SEM; a dry solid is
:class:`ElasticVSSEM2DDeformed`. ``v`` lives on the nodes of the solid elements, ``p`` on
those of the fluid-filled ones (masked inverse masses keep the others at zero) and the
flux copies of a dry element stay zero. The interfaces:

* solid (porous or dry) -- fluid: ``-∮ p n φ`` on the solid momentum and ``+∮ ψ v·n`` on
  the pressure (``n`` out of the solid), the two exact transposes of
  :class:`sweep.sem.coupling.FluidSolidInterface`; with the continuous ``p`` that is the
  open-pore condition at a porous one (pressure, traction and the fluid volume flux
  ``(v + q)·n`` continuous), SPECFEM2D's;
* porous -- dry solid: nothing to add. ``v`` and the traction are continuous through the
  shared nodes and the weak form, and the dropped boundary integral of ``D^T W q`` is the
  sealed pores ``q·n = 0`` (SPECFEM2D sets the whole ``w = 0`` on those nodes).

A source loads only the fields of the element it sits in -- a stress source in a fluid or
a pore pressure in a dry solid is refused, a force in a fluid moves the fluid alone -- and
on an interface it is shared among the elements that carry each field.

Boundaries: an open-pore free surface (traction-free, ``p = 0``) on the ``dirichlet``
sides -- ``p`` pinned to zero on their nodes, the traction natural; elsewhere the boundary
integrals are dropped (traction-free, no flow), behind the C-PML. C-PML: the profile of
:class:`ElasticVSSEM2DDeformed`, field memories for the weak forms (stress, flux) and
gradient memories for the strong ones (``grad v``, ``grad p``): 12 element fields. An
interface may cross the side or bottom bands, with the plain coupling there, but no corner
region (as in :class:`~sweep.sem.equations.acoustic_elastic2d.AcousticElasticSEM2D`).

Sources
-------
``src`` is aligned with ``(v_x, v_z, p, q_x, q_z)``. ``src[2]`` injects pore pressure:
``p`` jumps by ``dt src / A(w)`` at the start of the step and the total stress does not move
(FD ``Biot``'s ``p``). The others follow ``source_mode``:

* ``"load"`` (default): weak-form loads, ``src[0:2]`` on the total momentum and ``src[3:5]``
  on the relative flow (shared among the flux copies at a node by ``w``). A moment tensor
  ``-M : grad(phi)`` on the first injects total stress (FD's ``sxx`` / ``szz`` / ``sxz``); the
  same force on both is SPECFEM2D's poroelastic point force.
* ``"velocity"``: FD ``Biot``'s ``vx`` / ``vz`` / ``qx`` / ``qz``, after the momentum update:
  ``v`` jumps by ``dt src / M_rho`` with the flux unchanged, or every flux copy by
  ``dt src / A(w m)`` with ``v`` unchanged.

Eager only, no boundary saving: the viscous drag is dissipative at every node, so the
reverse reconstruction does not exist (as on the FD ``Biot``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, phys_grads
from ..coupling import FluidSolidInterface, assert_no_corner_pml_on_interface
from ..cpml.elastic_vs import SimpleElasticCPMLProfile, build_simple_elastic_cpml
from ..fields import FieldSpec, check_element_arrays, init_state_from_specs
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.domains import DomainMask, masked_inv_mass
from ..mesh.quad9 import Quad9Mesh

_SOURCE_MODES = ("load", "velocity")


def _node_slots(ibool):
    """``(node, slot, kmax)`` of every element-local node in flattened ``ibool`` order: its
    global node and its rank among the element copies of that node."""
    g = ibool.reshape(-1)
    order = torch.argsort(g, stable=True)
    gs = g[order]
    pos = torch.arange(gs.numel(), device=g.device)
    start = torch.ones_like(gs, dtype=torch.bool)
    start[1:] = gs[1:] != gs[:-1]
    run = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    slot = torch.empty_like(pos)
    slot[order] = pos - run
    return g, slot, int(slot.max()) + 1


def _media(H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e, b, tensor):
    """``(solid, fluid)``, ``(E,)`` bool: the elements with a solid frame (``rho > 0``) and
    those with a pore fluid (``M > 0``); both is a porous element, one alone a dry solid or
    a fluid. The coefficients a medium does not have must be zero."""
    drag = ("b_xx", "b_xz", "b_zz") if tensor else ("b", "b_xz", "b")
    names = ("H", "mu", "C", "M", "rho", "rho_f", "m") + drag
    vals = dict(zip(names, (H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e, *b)))
    flags = []
    for key, what in (("rho", "a solid frame (rho > 0)"), ("M", "a pore fluid (M > 0)")):
        x = vals[key].flatten(1)
        if bool((x < 0).any()):
            raise ValueError(f"{key} must be >= 0")
        has, some = (x > 0).all(1), (x > 0).any(1)
        if bool((some & ~has).any()):
            bad = int((some & ~has).nonzero()[0])
            raise ValueError(f"element {bad} has {what} on some of its nodes only; an "
                             f"element is wholly porous, wholly fluid or wholly dry")
        flags.append(has)
    solid, fluid = flags
    if bool((~solid & ~fluid).any()):
        bad = int((~solid & ~fluid).nonzero()[0])
        raise ValueError(f"element {bad} has neither a solid frame (rho > 0) nor a pore "
                         f"fluid (M > 0)")
    for where, zero, what in ((solid & ~fluid, ("C", "rho_f", "m") + drag,
                               "a dry elastic element (M = 0)"),
                              (fluid & ~solid, ("H", "mu", "C", "rho_f") + drag,
                               "a fluid element (rho = 0)")):
        if bool(where.any()):
            bad = [k for k in dict.fromkeys(zero) if bool((vals[k][where] != 0).any())]
            if bad:
                raise ValueError(f"{what} has nonzero {bad}: they do not exist there "
                                 f"and must be 0")
    if bool((m_e[fluid] <= 0).any()):
        raise ValueError("m must be > 0 where there is a fluid (M > 0)")
    return solid, fluid


def viscous_relaxation_times(q0: float, f0: float):
    """``(tau_e, tau_s)`` of SPECFEM2D's viscous standard linear solid, quality ``q0`` at
    ``f0`` Hz."""
    if not (q0 > 0 and f0 > 0):
        raise ValueError(f"viscous_q needs Q0 > 0 and f0 > 0, got ({q0}, {f0})")
    s = math.sqrt(q0 * q0 + 1.0)
    return (s + 1.0) / (2.0 * math.pi * f0 * q0), (s - 1.0) / (2.0 * math.pi * f0 * q0)


@dataclass
class BiotVSSEM2DDeformed:
    """1st-order velocity-stress Biot poroelastic SEM on a deformed Quad9 mesh."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    field_specs: ClassVar[list] = [
        FieldSpec("v_x", supports_source=True, supports_receiver=True),
        FieldSpec("v_z", supports_source=True, supports_receiver=True),
        FieldSpec("p", supports_source=True, supports_receiver=True),
        FieldSpec("q_x", supports_source=True, supports_receiver=True, layout="element"),
        FieldSpec("q_z", supports_source=True, supports_receiver=True, layout="element"),
        FieldSpec("sigma_xx", supports_receiver=True, layout="element"),
        FieldSpec("sigma_zz", supports_receiver=True, layout="element"),
        FieldSpec("sigma_xz", supports_receiver=True, layout="element"),
        FieldSpec("r_x", internal=True, layout="element"),   # viscous memory (viscous_q)
        FieldSpec("r_z", internal=True, layout="element"),
        FieldSpec("m_sxx_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_szz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_qx_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_qz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_px", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_pz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxx", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzx", internal=True, boundary_related=True, layout="element"),
    ]

    # FD Biot's receiver names
    derived_receiver_types: ClassVar[dict] = {
        "vx": ("v_x", None), "vz": ("v_z", None), "qx": ("q_x", None), "qz": ("q_z", None),
        "sxx": ("sigma_xx", None), "szz": ("sigma_zz", None), "sxz": ("sigma_xz", None)}

    # PropSEM's source names: kind and payload over (v_x, v_z, p, q_x, q_z), or the
    # (Mxx, Mzz, Mxz) of a moment. "velocity": FD Biot's injections (source_mode
    # "velocity"); "force": weak-form loads (source_mode "load"), fx / fz being SPECFEM2D's
    # poroelastic point force; "injection": the pore pressure, either mode; "moment": the
    # total-stress moment tensor, driven by the wavelet's integral.
    propsem_source_names: ClassVar[dict] = {
        "vx": ("velocity", (1.0, 0.0, 0.0, 0.0, 0.0)),
        "vz": ("velocity", (0.0, 1.0, 0.0, 0.0, 0.0)),
        "qx": ("velocity", (0.0, 0.0, 0.0, 1.0, 0.0)),
        "qz": ("velocity", (0.0, 0.0, 0.0, 0.0, 1.0)),
        "v_x": ("velocity", (1.0, 0.0, 0.0, 0.0, 0.0)),
        "v_z": ("velocity", (0.0, 1.0, 0.0, 0.0, 0.0)),
        "q_x": ("velocity", (0.0, 0.0, 0.0, 1.0, 0.0)),
        "q_z": ("velocity", (0.0, 0.0, 0.0, 0.0, 1.0)),
        "fx": ("force", (1.0, 0.0, 0.0, 1.0, 0.0)),
        "fz": ("force", (0.0, 1.0, 0.0, 0.0, 1.0)),
        "p": ("injection", (0.0, 0.0, 1.0, 0.0, 0.0)),
        "sxx": ("moment", (1.0, 0.0, 0.0)), "szz": ("moment", (0.0, 1.0, 0.0)),
        "sxz": ("moment", (0.0, 0.0, 1.0)),
    }
    default_source_type: ClassVar[tuple] = ("sxx", "szz")
    default_receiver_type: ClassVar[tuple] = ("v_x", "v_z")
    tensor_drag: ClassVar[bool] = False

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "Biot has no boundary saving: the viscous drag is dissipative at every node, so "
        "stepping back multiplies round-off by exp(gamma dt) per step and the reverse "
        "reconstruction does not exist. Use memory=Full() or Ckpt(chunks=...).")

    mesh: Quad9Mesh
    H_e: torch.Tensor                  # (E, ngll, ngll) undrained P-wave modulus
    mu_e: torch.Tensor
    C_e: torch.Tensor
    M_e: torch.Tensor
    rho_e: torch.Tensor
    rho_f_e: torch.Tensor
    m_e: torch.Tensor
    b_e: tuple                         # (b_xx, b_xz, b_zz) element-local, relaxed
    dirichlet: tuple                   # the open-pore sides: p = 0 on their nodes
    dirichlet_idx: torch.Tensor
    viscous_q: Optional[tuple]         # (Q0, f0) of the viscous attenuation, or None

    mass_rho: torch.Tensor             # (n_global,) A(w rho)
    _inv_M: torch.Tensor               # 1 / mass_rho, 0 off the solid
    _inv_Mm: torch.Tensor              # 1 / A(w m): a flux injected at a point
    _inv_Mp: torch.Tensor              # 1 / A(w / M), the pressure's lumped compliance
    _inv_M1: torch.Tensor              # 1 / A(w) over the fluid: a pressure injected
    _lam_d: torch.Tensor               # drained lambda, H - 2 mu - C^2 / M
    _alpha: torch.Tensor               # C / M (0 in a dry solid)
    # the media (see "Fluid and elastic regions"): (E,) bool, the flux mask (E, 1, 1),
    # the solid-fluid faces
    _solid_e: torch.Tensor
    _fluid_e: torch.Tensor
    _qmask: torch.Tensor
    ifc: FluidSolidInterface
    _tau: tuple                        # (tau_e, tau_s), equal without viscous_q
    # the element copies of the flux at every node: flattened ibool -> node * kmax + slot
    _slot_idx: torch.Tensor            # (E ngll^2,)
    _kmax: int
    _wm: torch.Tensor                  # (n_global, kmax) w m per copy (1 on empty slots)
    _wr: torch.Tensor                  # w rho_f (0 on empty slots)
    _wb: tuple                         # w (b_xx, b_xz, b_zz) tau_e / tau_s: unrelaxed
    _share: torch.Tensor               # w / A(w): a copy's share of a nodal flux load

    cpml: Optional[SimpleElasticCPMLProfile]

    _D: torch.Tensor                   # (ngll, ngll)
    _w_J: torch.Tensor
    _ixx: torch.Tensor
    _ixz: torch.Tensor
    _iex: torch.Tensor
    _iez: torch.Tensor

    source_mode: str = "load"          # see the module docstring
    _drag: tuple = None                # (dt, E, Psi), set by init_state

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        H_e: torch.Tensor,
        mu_e: torch.Tensor,
        C_e: torch.Tensor,
        M_e: torch.Tensor,
        rho_e: torch.Tensor,
        rho_f_e: torch.Tensor,
        m_e: torch.Tensor,
        b_e: torch.Tensor,
        *,
        dirichlet: tuple = (),
        cpml: Optional[SimpleElasticCPMLProfile] = None,
        viscous_q: Optional[tuple] = None,
    ) -> "BiotVSSEM2DDeformed":
        """Build the operators from the eight Biot coefficients at the GLL nodes;
        ``dirichlet`` names the open-pore sides, ``viscous_q = (Q0, f0)`` turns on the
        viscous attenuation."""
        check_element_arrays(mesh, b_e=b_e)
        return cls._build(mesh, H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e,
                          (b_e, torch.zeros_like(b_e), b_e), dirichlet, cpml, viscous_q)

    @classmethod
    def _build(cls, mesh, H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e, b, dirichlet, cpml,
               viscous_q):
        check_element_arrays(mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
                             rho_f_e=rho_f_e, m_e=m_e)
        b_xx, b_xz, b_zz = b
        if bool(((b_xx < 0) | (b_zz < 0) | (b_xx * b_zz < b_xz * b_xz)).any()):
            raise ValueError("the drag eta / kappa must be positive semi-definite")
        solid, fluid = _media(H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e, b, cls.tensor_drag)
        tau = (1.0, 1.0) if viscous_q is None else viscous_relaxation_times(*viscous_q)
        device = H_e.device
        w = mesh.w_jac
        ibool = mesh.ibool
        like = H_e.new_zeros(mesh.n_global)

        def lump(c):
            return assemble(w * c, ibool, like)

        def alive(elements):
            out = torch.zeros(mesh.n_global, dtype=torch.bool, device=device)
            return out.index_fill(0, ibool[elements].reshape(-1), True)

        node, slot, kmax = _node_slots(ibool)
        slot_idx = node * kmax + slot

        def slots(x, fill):
            out = torch.full((mesh.n_global * kmax,), fill, dtype=x.dtype, device=device)
            return out.index_copy(0, slot_idx, x.reshape(-1)).view(mesh.n_global, kmax)

        v_alive, p_alive = alive(solid), alive(fluid)
        fl = fluid.view(-1, 1, 1)
        qmask = fl.to(H_e.dtype)
        # the coefficients a medium does not have are zero (checked above); through the
        # masks their gradient is zero too, so an update keeps them so -- a fluid's rho
        # would otherwise feel the solid's mass on the nodes it shares with a solid
        sm = solid.view(-1, 1, 1).to(H_e.dtype)
        pm = sm * qmask
        H_e, mu_e, rho_e, C_e, rho_f_e = H_e * sm, mu_e * sm, rho_e * sm, C_e * pm, rho_f_e * pm
        m_e = m_e * qmask
        b_xx, b_xz, b_zz = b_xx * pm, b_xz * pm, b_zz * pm
        inv_M_e = torch.where(fl, 1.0 / torch.where(fl, M_e, torch.ones_like(M_e)), 0.0)
        mass_rho = lump(rho_e)
        mass_1 = lump(qmask.expand_as(M_e))
        inv_M1 = masked_inv_mass(mass_1, p_alive)
        alpha = C_e * inv_M_e
        unrelax = tau[0] / tau[1]
        ifc = FluidSolidInterface.detect(mesh, DomainMask.from_element_flags(mesh, ~solid))
        if cpml is not None and ifc.n_face:
            node_profile = {k: like.index_copy(0, ibool.reshape(-1), getattr(cpml, k).reshape(-1))
                            for k in ("b_x", "b_z")}
            assert_no_corner_pml_on_interface(ifc, SimpleNamespace(**node_profile))
        return cls(
            mesh=mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
            rho_f_e=rho_f_e, m_e=m_e, b_e=(b_xx, b_xz, b_zz),
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            viscous_q=None if viscous_q is None else tuple(float(v) for v in viscous_q),
            mass_rho=mass_rho, _inv_M=masked_inv_mass(mass_rho, v_alive),
            _inv_Mm=masked_inv_mass(lump(m_e), p_alive),
            _inv_Mp=masked_inv_mass(lump(inv_M_e), p_alive), _inv_M1=inv_M1,
            _lam_d=H_e - 2.0 * mu_e - alpha * C_e, _alpha=alpha,
            _solid_e=solid, _fluid_e=fluid, _qmask=qmask, ifc=ifc, _tau=tau,
            _slot_idx=slot_idx, _kmax=kmax,
            _wm=slots(torch.where(fl, w * m_e, 1.0), 1.0), _wr=slots(w * rho_f_e, 0.0),
            _wb=tuple(slots(w * bb * unrelax, 0.0) for bb in (b_xx, b_xz, b_zz)),
            _share=slots(w * qmask, 0.0) * inv_M1[:, None],
            cpml=cpml,
            _D=mesh.deriv, _w_J=w,
            _ixx=mesh.inv_J[..., 0, 0], _ixz=mesh.inv_J[..., 0, 1],
            _iex=mesh.inv_J[..., 1, 0], _iez=mesh.inv_J[..., 1, 1],
        )

    # The C-PML builder ``cpml=`` expects, and the speed PropSEM designs it for.
    cpml_builder: ClassVar[object] = staticmethod(build_simple_elastic_cpml)

    @staticmethod
    def pml_speed(models) -> float:
        """The largest P velocity of the coefficients (the drag does not enter): the
        inviscid fast P wave in a porous medium, ``sqrt(H / rho)`` in a dry solid,
        ``sqrt(M / m)`` in a fluid."""
        from ...equations.biot import biot_wave_velocities
        h, mu, c, bm, rho, rho_f, m = torch.broadcast_tensors(
            *(torch.as_tensor(x).detach().double() for x in models[:7]))
        solid, fluid = rho > 0, bm > 0
        speeds = []
        if bool((solid & fluid).any()):
            sel = solid & fluid
            speeds.append(biot_wave_velocities([x[sel] for x in (h, mu, c, bm, rho, rho_f, m)])[0])
        if bool((solid & ~fluid).any()):
            speeds.append(torch.sqrt(h[solid & ~fluid] / rho[solid & ~fluid]))
        if bool((fluid & ~solid).any()):
            speeds.append(torch.sqrt(bm[fluid & ~solid] / m[fluid & ~solid]))
        return float(torch.cat([v.reshape(-1) for v in speeds]).max())

    @property
    def source_field_elements(self):
        """The elements each source field ``(v_x, v_z, p, q_x, q_z)`` lives in: a source
        loads a field only through those (see :func:`sweep.sem.propagator._stamp_loads`)."""
        s, f = self._solid_e, self._fluid_e
        return (s, s, f, f, f)

    # ------------------------------------------------------------------ #
    # The drag: one matrix exponential per node, for the run's dt
    # ------------------------------------------------------------------ #
    def drag_operators(self, dt: float, chunk: int = 1 << 16):
        """``(E, Psi)`` advancing the flux copies at a node by ``q <- E q + Psi r`` over a
        step of frozen forces ``r``: from ``A dq/dt + W_B q = r``,
        ``exp([[-A^-1 W_B, A^-1], [0, 0]] dt) = [[E, Psi], [0, I]]``, in float64. Each is
        ``(n_global, kmax, kmax)`` (both components alike), or ``(n_global, 2 kmax,
        2 kmax)`` over ``(q_x copies, q_z copies)`` with a drag tensor. ``chunk`` nodes at a
        time: the exponential's work arrays are ~10 times its input."""
        wm, wr = self._wm.double(), self._wr.double()
        wb = [t.double() for t in self._wb]
        inv_rho = masked_inv_mass(self.mass_rho.double(), self._inv_M != 0)
        dtype = self.mass_rho.dtype
        Es, Psis = [], []
        for i in range(0, wm.shape[0], chunk):
            n = slice(i, i + chunk)
            A = torch.diag_embed(wm[n]) - wr[n, :, None] * wr[n, None, :] * inv_rho[n, None, None]
            if self.tensor_drag:
                z = torch.zeros_like(A)
                A = torch.cat([torch.cat([A, z], 2), torch.cat([z, A], 2)], 1)
                wxx, wxz, wzz = (torch.diag_embed(t[n]) for t in wb)
                WB = torch.cat([torch.cat([wxx, wxz], 2), torch.cat([wxz, wzz], 2)], 1)
            else:
                WB = torch.diag_embed(wb[0][n])
            k = A.shape[-1]
            A_inv = torch.linalg.inv(A)
            top = torch.cat([-A_inv @ WB, A_inv], 2)
            X = torch.linalg.matrix_exp(torch.cat([top, torch.zeros_like(top)], 1) * dt)
            Es.append(X[:, :k, :k].to(dtype))
            Psis.append(X[:, :k, k:].to(dtype))
        return torch.cat(Es), torch.cat(Psis)

    def init_state(self, dt: float, device, dtype) -> tuple:
        """Zero fields, and the drag operators for this ``dt`` -- computed here, once per
        run, outside any checkpointed chunk of the time loop."""
        if self._drag is None or self._drag[0] != dt:
            self._drag = (dt, *self.drag_operators(dt))
        return init_state_from_specs(self, device, dtype)

    def _to_slots(self, x):
        """Element-local ``(E, ngll, ngll)`` -> ``(n_global, kmax)``, zero on empty slots."""
        k = self._kmax
        out = x.new_zeros(self.mesh.n_global * k).index_add(0, self._slot_idx, x.reshape(-1))
        return out.view(self.mesh.n_global, k)

    def _from_slots(self, s, like):
        return s.reshape(-1)[self._slot_idx].view_as(like)

    def _weak_div(self, *pairs):
        """``sum_g w (d(phi)/dx X + d(phi)/dz Z)`` at the global nodes for every element-local
        pair ``(X, Z)`` -- :class:`ElasticVSSEM2DDeformed`'s stress divergence for the
        momentum, ``D^T W q`` for the pressure."""
        w_J, ixx, ixz, iex, iez = self._w_J, self._ixx, self._ixz, self._iex, self._iez
        F_beta = torch.stack([w_J * (X * ixx + Z * ixz) for X, Z in pairs], dim=-1)
        F_alpha = torch.stack([w_J * (X * iex + Z * iez) for X, Z in pairs], dim=-1)
        R_e = (torch.einsum("db,eadk->eabk", self._D, F_beta)
               + torch.einsum("ga,egbk->eabk", self._D, F_alpha))
        like = pairs[0][0].new_zeros(self.mesh.n_global)
        if len(pairs) == 1:
            return assemble(R_e[..., 0], self.mesh.ibool, like)
        return assemble(R_e, self.mesh.ibool, like)

    def _pin(self, p):
        """``p = 0`` on the open-pore surface."""
        idx = self.dirichlet_idx
        return p if idx.numel() == 0 else p.index_fill(0, idx, 0.0)

    def _advance_flux(self, q_x, q_z, r_x, r_z):
        """The flux copies after the step, ``E q + Psi r``, slot layout ``(n_global, k)``."""
        _, E, Psi = self._drag
        if self.tensor_drag:
            k = self._kmax
            qs = torch.cat([q_x, q_z], 1)
            new = torch.einsum("nij,nj->ni", E, qs) + torch.einsum("nij,nj->ni", Psi,
                                                                    torch.cat([r_x, r_z], 1))
            return new[:, :k], new[:, k:]
        return tuple(torch.einsum("nij,nj->ni", E, q) + torch.einsum("nij,nj->ni", Psi, r)
                     for q, r in ((q_x, r_x), (q_z, r_z)))

    # ================================================================== #
    # SEMPropagator interface: init_state + step (staggered leapfrog).
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v, q at n+1/2; sigma, p, r at n+1)."""
        (v_x, v_z, p, q_x, q_z, s_xx, s_zz, s_xz, r_x, r_z,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_qx_x, m_qz_z, m_px, m_pz,
         m_vxx, m_vzz, m_vxz, m_vzx) = fields
        mode = self.source_mode
        if mode not in _SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {_SOURCE_MODES}, got {mode!r}")
        if self._drag is None or self._drag[0] != dt:
            raise RuntimeError(f"step(dt={dt}) needs init_state(dt) first: the drag "
                               f"operators are computed there, once per run")
        cp = self.cpml
        ibool = self.mesh.ibool
        w = self._w_J

        # ----- pore pressure injected at t^n; the total stress does not move -----
        p = self._pin(p + dt * self._inv_M1 * src[2])

        # ----- forces at t^n: total-stress divergence and pressure gradient -----
        d_p_dx, d_p_dz = phys_grads(self._D, ibool, self._ixx, self._ixz, self._iex,
                                    self._iez, p)
        if cp is not None:
            m_sxx_x = cp.a_x * m_sxx_x + cp.b_x * s_xx
            m_sxz_z = cp.a_z * m_sxz_z + cp.b_z * s_xz
            m_sxz_x = cp.a_x * m_sxz_x + cp.b_x * s_xz
            m_szz_z = cp.a_z * m_szz_z + cp.b_z * s_zz
            m_px = cp.a_x * m_px + cp.b_x * d_p_dx
            m_pz = cp.a_z * m_pz + cp.b_z * d_p_dz
            d_p_dx, d_p_dz = d_p_dx + m_px, d_p_dz + m_pz
            R_x, R_z = self._weak_div((s_xx + m_sxx_x, s_xz + m_sxz_z),
                                      (s_xz + m_sxz_x, s_zz + m_szz_z))
        else:
            R_x, R_z = self._weak_div((s_xx, s_xz), (s_xz, s_zz))
        f_x, f_z = -R_x, -R_z
        ifc = self.ifc
        if ifc.n_face:
            # a fluid's pressure on the solid: -∮ p n φ, n out of the solid
            s_x, s_z = ifc.apply_el(p, p)
            f_x, f_z = f_x + s_x, f_z + s_z

        # ----- v and the flux copies at every node, the drag integrated exactly -----
        wr, inv_m, qmask = self._wr, self._inv_M, self._qmask
        # w (-grad p + r) per copy; nothing moves the copies of a dry element
        g_x, g_z = w * (r_x - d_p_dx) * qmask, w * (r_z - d_p_dz) * qmask
        if mode == "load":
            f_x, f_z = f_x + src[0], f_z + src[1]
        r_xs = self._to_slots(g_x) - wr * (f_x * inv_m)[:, None]
        r_zs = self._to_slots(g_z) - wr * (f_z * inv_m)[:, None]
        if mode == "load":
            r_xs = r_xs + self._share * src[3][:, None]
            r_zs = r_zs + self._share * src[4][:, None]
        qs_x, qs_z = self._to_slots(q_x), self._to_slots(q_z)
        new_x, new_z = self._advance_flux(qs_x, qs_z, r_xs, r_zs)
        v_x = v_x + (dt * f_x - (wr * (new_x - qs_x)).sum(1)) * inv_m
        v_z = v_z + (dt * f_z - (wr * (new_z - qs_z)).sum(1)) * inv_m
        q_x, q_z = self._from_slots(new_x, q_x), self._from_slots(new_z, q_z)
        if mode == "velocity":
            v_x = v_x + dt * inv_m * src[0]
            v_z = v_z + dt * inv_m * src[1]
            q_x = q_x + (dt * self._inv_Mm * src[3])[ibool] * qmask
            q_z = q_z + (dt * self._inv_Mm * src[4])[ibool] * qmask

        # ----- stress, pressure and the viscous memory from the new v, q -----
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = phys_grads(
            self._D, ibool, self._ixx, self._ixz, self._iex, self._iez, v_x, v_z)
        if cp is not None:
            m_vxx = cp.a_x * m_vxx + cp.b_x * d_vx_dx
            m_vzz = cp.a_z * m_vzz + cp.b_z * d_vz_dz
            m_vxz = cp.a_z * m_vxz + cp.b_z * d_vx_dz
            m_vzx = cp.a_x * m_vzx + cp.b_x * d_vz_dx
            d_vx_dx = d_vx_dx + m_vxx
            d_vz_dz = d_vz_dz + m_vzz
            d_vx_dz = d_vx_dz + m_vxz
            d_vz_dx = d_vz_dx + m_vzx
            m_qx_x = cp.a_x * m_qx_x + cp.b_x * q_x
            m_qz_z = cp.a_z * m_qz_z + cp.b_z * q_z
            div_q = self._weak_div((q_x + m_qx_x, q_z + m_qz_z))
        else:
            div_q = self._weak_div((q_x, q_z))
        div_v = d_vx_dx + d_vz_dz
        alpha = self._alpha
        rhs = div_q - assemble(w * alpha * div_v, ibool, p.new_zeros(self.mesh.n_global))
        if ifc.n_face:
            rhs = rhs - ifc.apply_ac(v_x, v_z, p)          # +∮ ψ v·n, n out of the solid
        dp_free = self._pin(dt * self._inv_Mp * rhs)
        p = p + dp_free
        lam_d, mu = self._lam_d, self.mu_e
        dsig_p = alpha * dp_free[ibool]                    # sigma = sigma' - alpha p
        s_xx = s_xx + dt * (lam_d * div_v + 2.0 * mu * d_vx_dx) - dsig_p
        s_zz = s_zz + dt * (lam_d * div_v + 2.0 * mu * d_vz_dz) - dsig_p
        s_xz = s_xz + dt * mu * (d_vx_dz + d_vz_dx)
        if self.viscous_q is not None:
            # dr/dt = -(r - (tau_e / tau_s - 1) B q) / tau_s, exact over the step at q^{n+1/2}
            tau_e, tau_s = self._tau
            decay = math.exp(-dt / tau_s)
            gain = (1.0 - decay) * (tau_e / tau_s - 1.0)
            b_xx, b_xz, b_zz = self.b_e
            r_x = decay * r_x + gain * (b_xx * q_x + b_xz * q_z)
            r_z = decay * r_z + gain * (b_xz * q_x + b_zz * q_z)

        return (v_x, v_z, p, q_x, q_z, s_xx, s_zz, s_xz, r_x, r_z,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_qx_x, m_qz_z, m_px, m_pz,
                m_vxx, m_vzz, m_vxz, m_vzx)


@dataclass
class BiotAnisoPermSEM2DDeformed(BiotVSSEM2DDeformed):
    """:class:`BiotVSSEM2DDeformed` with an anisotropic permeability: the drag
    ``eta kappa^-1`` is the symmetric tensor ``[[b_xx, b_xz], [b_xz, b_zz]]`` (x right, z
    down), SPECFEM2D's ``kxx, kxz, kzz``; build it with
    :func:`sweep.equations.biot_params.biot_drag_tensor`. The flux copies at a node then
    advance with both components coupled through the drag."""

    tensor_drag: ClassVar[bool] = True

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        H_e: torch.Tensor,
        mu_e: torch.Tensor,
        C_e: torch.Tensor,
        M_e: torch.Tensor,
        rho_e: torch.Tensor,
        rho_f_e: torch.Tensor,
        m_e: torch.Tensor,
        b_xx_e: torch.Tensor,
        b_xz_e: torch.Tensor,
        b_zz_e: torch.Tensor,
        *,
        dirichlet: tuple = (),
        cpml: Optional[SimpleElasticCPMLProfile] = None,
        viscous_q: Optional[tuple] = None,
    ) -> "BiotAnisoPermSEM2DDeformed":
        """Build from the seven Biot coefficients and the three drag components at the
        GLL nodes."""
        check_element_arrays(mesh, b_xx_e=b_xx_e, b_xz_e=b_xz_e, b_zz_e=b_zz_e)
        return cls._build(mesh, H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e,
                          (b_xx_e, b_xz_e, b_zz_e), dirichlet, cpml, viscous_q)
