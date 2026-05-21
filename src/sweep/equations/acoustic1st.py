from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec, ensure_field_specs

def step_cpml(p, vx, vz, phix, phiz, psix, psiz, vp, rho, dt, h, b, pd, pml=None):

    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    # az, bz, dbzdz, ax, bx, dbxdx = b

    # Update vx and vz
    p_x = pd.x_backward(p)
    p_z = pd.z_backward(p)
 
    psi_z = azh * psiz + bzh * p_z
    p_z = p_z + psi_z
    vz = vz - dt / rho * p_z

    psi_x = axh * psix + bxh * p_x
    p_x = p_x + psi_x
    vx = vx - dt / rho * p_x
    # Update p
    vx_x = pd.x_forward(vx)
    vz_z = pd.z_forward(vz)
    div_v = 0.

    phiz = az * phiz + bz * vz_z
    vz_z = vz_z + phiz
    div_v = div_v + vz_z

    phix = ax * phix + bx * vx_x
    vx_x = vx_x + phix
    div_v = div_v + vx_x

    p = p - vp**2 * rho * dt * (vz_z + vx_x)

    return p, vx, vz, phix, phiz, psi_x, psi_z

def step_spml(px, pz, vx, vz, vp, rho, dt, h, b, pd, pml=None):

    bx, bz = pml
    p = px + pz

    p_x = pd.x_backward(p)
    p_z = pd.z_backward(p)

    vx = ((1.-0.5*dt*bx)*vx+dt/rho*p_x)/(1.+0.5*dt*bx)
    vz = ((1.-0.5*dt*bz)*vz+dt/rho*p_z)/(1.+0.5*dt*bz)

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_forward(vz)

    px = ((1.-0.5*dt*bx)*px+vp**2 * rho * dt * vx_x)/(1.+0.5*dt*bx)
    pz = ((1.-0.5*dt*bz)*pz+vp**2 * rho * dt * vz_z)/(1.+0.5*dt*bz)

    return px, pz, vx, vz


class Acoustic1st(FirstOrderEquation):
    """First-order 2-D acoustic wave equation in pressure-velocity form.

    Variable-density scalar acoustics on a staggered grid: a pressure field
    ``p`` and particle velocities ``(vx, vz)`` evolve together. The default
    boundary type is ``'cpmls'`` (staggered-grid recursive CPML), which
    couples four memory variables (``phix``, ``phiz``, ``psix``, ``psiz``);
    ``'spml'`` is also supported and switches to a split-pressure
    formulation with two pressure components ``(px, pz)`` instead of
    ``p``.

    Reference: 10.1190/GEO2011-0345.1.

    !!! info "Models (constructor input order)"

        - ``vp`` (m/s): Acoustic P-wave velocity model.
        - ``rho`` (kg/m^3): Density model.

    !!! info "Wavefields (``cpmls`` mode — the default)"

        - ``p`` (aliases: ``pressure``): Acoustic pressure field; default source and receiver.
        - ``vx`` (aliases: ``velocity_x``): Particle velocity in the x direction.
        - ``vz`` (aliases: ``velocity_z``): Particle velocity in the z direction.
        - ``phix``: CPML memory variable for the x-derivative term (internal).
        - ``phiz``: CPML memory variable for the z-derivative term (internal).
        - ``psix``: CPML auxiliary field in the x direction (internal).
        - ``psiz``: CPML auxiliary field in the z direction (internal).

    !!! info "Wavefields (``spml`` mode)"

        - ``px``: Split-field pressure component in the x direction (internal).
        - ``pz``: Split-field pressure component in the z direction (internal).
        - ``vx`` (aliases: ``velocity_x``): Particle velocity in the x direction.
        - ``vz`` (aliases: ``velocity_z``): Particle velocity in the z direction.

    !!! info "Defaults"

        - ``source_type``: ``['p']``
        - ``receiver_type``: ``['p']``
        - ``pml_type``: ``'cpmls'``
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic P-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )

    default_pml_type = "cpmls"  # acoustic1st does not support 'cpmlr'; supported: ['cpmls', 'spml'].

    def __init__(self, spatial_order=4, device='cpu', backend='jax', **kwargs):
        """Build the first-order acoustic equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``). Higher orders
                reduce grid dispersion at the cost of more compute per
                step and a wider PML halo. This equation has **no compiled
                `impl='c'` path; use `impl='eager'`** (the default). The
                PyTorch / JAX eager path accepts any even order. Defaults
                to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                Defaults to ``'jax'`` for this class (the original
                research implementation); pass ``backend='torch'`` to
                stay on PyTorch.
            **kwargs: forwarded to :class:`FirstOrderEquation` for
                forward-compatibility (currently unused).
        """
        super().__init__(spatial_order, device, backend)
        # Pre-set the cpmls wavefield list so introspection (default_source_fields,
        # field_specs, etc.) works before the propagator calls setup_pml.
        self.wavefields = self.wavefields_cpml()

    def setup_pml(self, pml_type):
        self.wavefields = {
            'cpmls': self.wavefields_cpml(),
            'spml': self.wavefields_spml()
            }[pml_type]
        self.step = {
            'cpmls': step_cpml,
            'spml': step_spml
            }[pml_type]
    
    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    def wavefields_cpml(self):
        return ['p', 'vx', 'vz', 'phix', 'phiz', 'psix', 'psiz']

    def wavefields_spml(self):
        return ['px', 'pz', 'vx', 'vz']

    @property
    def field_specs(self):
        specs = []
        if "p" in self.wavefields:
            specs.append(
                FieldSpec("p", aliases=("pressure",), description="Acoustic pressure field.", supports_source=True, supports_receiver=True)
            )
        specs.extend([
            FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
            FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
            FieldSpec("px", description="Split-field pressure component in the x direction.", internal=True, boundary_related=True),
            FieldSpec("pz", description="Split-field pressure component in the z direction.", internal=True, boundary_related=True),
            FieldSpec("phix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
            FieldSpec("phiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
            FieldSpec("psix", description="CPML auxiliary field in the x direction.", internal=True, boundary_related=True),
            FieldSpec("psiz", description="CPML auxiliary field in the z direction.", internal=True, boundary_related=True),
        ])
        return ensure_field_specs(self.wavefields, specs)

    @property
    def default_source_fields(self):
        # 'p' exists only in cpmls mode; in spml mode the pressure is split into px/pz.
        return ["p"] if "p" in self.wavefields else ["px"]

    @property
    def default_receiver_fields(self):
        # Pressure if available; otherwise particle velocity in x (spml mode).
        return ["p"] if "p" in self.wavefields else ["vx"]

    @property
    def supported_pml(self):
        return ['cpmls', 'spml']

    def func(self, wavefields, models, dt, h, b, **kwargs):
        return self.step(*wavefields, *models, dt, h, b, pd=self.pd, pml=self.b, **kwargs)
