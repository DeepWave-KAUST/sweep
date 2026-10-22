import sys
import warnings

from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec, history_fields, record_single
from .fields import FieldSpec, ModelSpec
from ._registry import register_equation


S_GRADIENTS = ("adjoint", "frozen", "exact")
_S_MODE = {"frozen": 0.0, "exact": 1.0, "adjoint": 2.0}   # csrc qp2d: State::exact_s


def stop_gradient(x):
    """``x`` without its autograd history (torch) / ``lax.stop_gradient`` (jax).

    An explicit ``isinstance``: under torch.compile ``hasattr(x, "detach")`` is
    False inside an autograd.Function forward and sent torch tensors to jax."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach()
    import jax
    return jax.lax.stop_gradient(x)


def check_s_gradient(value, backend="torch"):
    if value not in S_GRADIENTS:
        raise ValueError(f"s_gradient must be one of {S_GRADIENTS}, got {value!r}")
    if value == "adjoint" and backend != "torch":
        raise NotImplementedError("s_gradient='adjoint' is implemented for backend='torch' (and impl='c'); "
                                  "use s_gradient='frozen' or 'exact' with jax")
    return value


def s_gradient_aux(equation):
    """``eq_aux`` for impl='c': one float32 CPU flag, the s_gradient mode
    (0 frozen, 1 exact, 2 adjoint)."""
    import torch
    return [torch.tensor([_S_MODE[equation.s_gradient]], dtype=torch.float32)]


def refuse_boundary_saving(equation, boundary_saving, boundary_dtype=None):
    """Boundary-saving configurations impl='c' refuses for the qP equations.

    * ``s_gradient='exact'``: the exact gradient is ill-conditioned and the
      reconstruction's round-off moves it by ~10%.
    * int8 storage: S_n is homogeneous of degree 0 in p, so the quantised
      frame turns into direction errors of the rebuilt field; the gradients
      move by 8-15% away from the source (cos 0.989-0.996), not only near it.
    """
    if boundary_saving and equation.s_gradient == "exact":
        raise NotImplementedError(
            f"{type(equation).__name__}(s_gradient='exact') with boundary saving: the exact gradient "
            "of the S_n scheme is ill-conditioned (its adjoint coefficients spike where grad p "
            "vanishes) and the reverse reconstruction's round-off moves it by ~10%. Use "
            "memory=Full() for the exact gradient, or s_gradient='adjoint' (default) / 'frozen'.")
    if boundary_saving and boundary_dtype == "int8":
        raise NotImplementedError(
            f"{type(equation).__name__} with int8 boundary storage: S_n depends on the direction "
            "of grad p only, so the quantised frame moves the gradients by 8-15% away from the "
            "source. Use storage_dtype='fp32' (or memory=Full()).")


def s_operator(px, pz, epsilon, delta):
    """S_n of a gradient pair (10.1190/geo2022-0292.1 eq. 19)."""
    numerator = -2 * (epsilon - delta) * px**2 * pz**2
    denominator = (1 + 2 * epsilon) * px**4 + pz**4 + 2 * (1 + delta) * px**2 * pz**2
    # NOT ``numerator * ((denominator + 1e-26) ** -1)``. That form is finite in
    # the forward, but autograd's pow backward evaluates ``x ** -2``, and in
    # fp32 ``(1e-26) ** -2`` overflows to +inf. Wherever the numerator is also
    # zero -- the whole grid at t=0, the quiet region ahead of the wavefront,
    # the PML corners -- the chain rule computes ``0 * inf = NaN``, and a single
    # NaN poisons the entire model gradient. Division's backward is
    # ``-grad * result / denominator`` and never forms ``den ** -2``.
    return numerator / (denominator + 1e-26)


def qp_stability_ratio(vp, epsilon, delta, dt, spacing, spatial_order, tilted=False):
    """The largest left side of the stability condition of the explicit
    pure-qP scheme over its right side; above 1 the scheme is unstable.

    VTI: eqs. B-8 to B-10 of Liang et al. (2023),

        dt vp sqrt([(1 + 2 eps) + S] / dx^2 + (1 + S) / dz^2)
            <= sqrt(2) / sqrt(sum_m c_2m [1 - (-1)^m]),

    with S evaluated at the Nyquist corner (B-10) and c_2m the second-
    derivative coefficients of ``spatial_order``.  TTI (``tilted``): the paper
    derives VTI only, and the rotation mixes the two axes, so both axes take
    the operator's largest coefficient, 1 + 2 eps + max(S, 0) -- over the
    directions S peaks at (delta - eps) / (sqrt(1 + 2 eps) + 1 + delta) --
    which bounds the limit from the safe side.  ``spacing`` is (dz, dx).
    """
    import torch
    from sweep.scalars import fd_coefficients

    c = fd_coefficients(2, int(spatial_order))
    nyquist = sum(float(cm) * (1 - (-1) ** m) for m, cm in enumerate(c, 1))
    dz, dx = float(spacing[-2]), float(spacing[-1])
    ix2, iz2 = 1.0 / dx ** 2, 1.0 / dz ** 2
    vp, eps, dl = (torch.as_tensor(m).detach().double() for m in (vp, epsilon, delta))
    if tilted:
        s_max = torch.clamp((dl - eps) / (torch.sqrt(1 + 2 * eps) + 1 + dl), min=0.0)
        k2 = (1 + 2 * eps + s_max) * (ix2 + iz2)
    else:
        skm = -2 * (eps - dl) * ix2 * iz2 / ((1 + 2 * eps) * ix2 ** 2 + iz2 ** 2 + 2 * (1 + dl) * ix2 * iz2)
        k2 = ((1 + 2 * eps) + skm) * ix2 + (1 + skm) * iz2
    lhs = float(dt) * float((vp * torch.sqrt(k2)).max())
    return lhs * nyquist ** 0.5 / 2 ** 0.5


def check_qp_stability(equation, models, dt, spacing):
    """Warn once per equation when ``dt`` breaks :func:`qp_stability_ratio`
    anywhere in the model.  The run is not stopped (``PropTorch`` calls this
    before every forward)."""
    if getattr(equation, "_stability_warned", False):
        return
    m = dict(zip(equation.models, models))
    ratio = qp_stability_ratio(m["vp"], m["epsilon"], m["delta"], dt, spacing, equation.so,
                               tilted="theta" in m)
    if ratio > 1.0:
        equation._stability_warned = True
        warnings.warn(
            f"{type(equation).__name__}: dt = {float(dt):g} s is {ratio:.2f}x the stability limit of "
            f"the explicit pure-qP scheme (Liang et al. 2023, eq. B-9) at spacing (dz, dx) = "
            f"{tuple(float(h) for h in spacing[-2:])}; the run is not stopped, but expect it to "
            f"blow up. Use dt <= {float(dt) / ratio:.3g} s.", stacklevel=3)


def cpml_memories(u_now, psix, psiz, zetax, zetaz, h, pml, lap_x, lap_z, grad_op, grad_kernels=None):
    """The next CPML memories of a step, the expressions of ``rhs_cpml``."""
    az, bz, dbzdz, ax, bx, dbxdx = pml
    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)
    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)
    return (bx * dudx + ax * psix, bz * dudz + az * psiz, bx * tmpx + ax * zetax, bz * tmpz + az * zetaz)


class _AdjointSRhs:
    """``s_gradient='adjoint'``: the step's update with a backward that
    propagates the adjoint wavefield with S_n taken from ITS OWN gradient (the
    adjoint of the wavenumber-domain operator, eq. 14, is the same qP
    operator), and differentiates the models at the forward field's S_n.

    ``rhs_fn(u, psix, psiz, zetax, zetaz, models, s_frozen, s_override)`` is the
    equation's update (``(rhs, memories)``); ``s_fn(field, models)`` is S_n of a
    field's plain gradient. The forward values are those of the other modes.
    """

    _fn = None

    @classmethod
    def apply(cls, rhs_fn, s_fn, *tensors):
        if cls._fn is None:
            import torch

            class Fn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, rhs_fn, s_fn, u, psix, psiz, zetax, zetaz, *models):
                    ctx.rhs_fn, ctx.s_fn = rhs_fn, s_fn
                    ctx.save_for_backward(u, psix, psiz, zetax, zetaz, *models)
                    return rhs_fn(u, psix, psiz, zetax, zetaz, models, True, None)[0]

                @staticmethod
                def backward(ctx, grad):
                    saved = ctx.saved_tensors
                    wf, models = saved[:5], saved[5:]
                    need = ctx.needs_input_grad[2:]
                    grad = grad.detach()
                    det = [m.detach() for m in models]
                    out = [None] * (5 + len(models))
                    with torch.enable_grad():
                        if any(need[:5]):
                            s_adj = ctx.s_fn(grad, det).detach()
                            w = [t.detach().requires_grad_(True) for t in wf]
                            r = ctx.rhs_fn(*w, det, True, s_adj)[0]
                            g = torch.autograd.grad(r, w, grad, allow_unused=True)
                            out[:5] = g
                        if any(need[5:]):
                            ms = [m.detach().requires_grad_(True) for m in models]
                            r = ctx.rhs_fn(*[t.detach() for t in wf], ms, True, None)[0]
                            g = torch.autograd.grad(r, ms, grad, allow_unused=True)
                            out[5:] = g
                    out = [o if n else None for o, n in zip(out, need)]
                    return (None, None, *out)

            cls._fn = Fn
        return cls._fn.apply(rhs_fn, s_fn, *tensors)


def adjoint_mode_step(rhs_fn, s_fn, lap_fn, wavefields, models, h, pml, grad_op, grad_kernels):
    """One step of an equation in ``s_gradient='adjoint'`` mode (torch)."""
    u_now, u_pre, psix, psiz, zetax, zetaz = wavefields
    rhs = _AdjointSRhs.apply(rhs_fn, s_fn, u_now, psix, psiz, zetax, zetaz, *models)
    lap_z, lap_x = lap_fn(u_now)
    aux = cpml_memories(u_now, psix, psiz, zetax, zetaz, h, pml, lap_x, lap_z, grad_op, grad_kernels)
    return rhs, aux


def rhs_cpml(
    u_now,
    psix,
    psiz,
    zetax,
    zetaz,
    vp,
    epsilon,
    delta,
    dt,
    h,
    b,
    pml,
    lap_x,
    lap_z,
    grad_op,
    grad_kernels=None,
    s_frozen=False,
    s_override=None,
):
    """The update ``v^2 dt^2 [((1 + 2 eps) + S) Lpx + (1 + S) Lpz]`` of one step
    and the next CPML memories ``(psixn, psizn, zetaxn, zetazn)``.

    ``step_cpml`` adds it to ``2 u_now - u_pre``; the Born equation
    (``qP_lsrtm``) also scatters it into the perturbation field.
    ``s_frozen`` takes S's wavefield direction out of the gradient (see
    ``AcousticVTI(s_gradient=...)``); the values are the same either way.
    ``s_override`` replaces S by a given field (the adjoint mode's backward).
    """

    az, bz, dbzdz, ax, bx, dbxdx = pml

    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)

    # Use the split-field PML corrected first derivatives in the anisotropy term.
    dpdz = (1 + bz) * dudz + az * psiz
    dpdx = (1 + bx) * dudx + ax * psix

    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)

    lap_z_pml = (1 + bz) * tmpz + az * zetaz
    lap_x_pml = (1 + bx) * tmpx + ax * zetax

    # 10.1190/geo2022-0292.1 EQ(19) from 10.1190/geo2014-0242.1
    if s_override is not None:
        sk = s_override
    else:
        px, pz = (stop_gradient(dpdx), stop_gradient(dpdz)) if s_frozen else (dpdx, dpdz)
        sk = s_operator(px, pz, epsilon, delta)

    vp2dt2 = vp**2 * dt**2
    rhs = vp2dt2 * (((1 + 2 * epsilon) + sk) * lap_x_pml + (1 + sk) * lap_z_pml)

    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz

    return rhs, (psixn, psizn, zetaxn, zetazn)


def step_cpml(
    u_now,
    u_pre,
    psix,
    psiz,
    zetax,
    zetaz,
    vp,
    epsilon,
    delta,
    dt,
    h,
    b,
    pml,
    lap_x,
    lap_z,
    grad_op,
    grad_kernels=None,
    s_frozen=False,
):
    rhs, aux = rhs_cpml(u_now, psix, psiz, zetax, zetaz, vp, epsilon, delta, dt, h, b, pml,
                        lap_x, lap_z, grad_op, grad_kernels, s_frozen)
    # ((2 u - u_pre) + rhs): the expression tree of the inline form, bit for bit
    u_next = 2 * u_now - u_pre + rhs
    return (u_next, u_now, *aux)


@register_equation(aliases=('AcousticVTIDefault', 'AcousticVTILiang'))
class AcousticVTI(SecondOrderEquation):
    """Second-order 2-D pure-qP VTI wave equation (Liang et al. 2023).

    The decoupled qP-wave equation of the "new acoustic approximation" (the
    qSV phase velocity set to zero in every direction), solved in the space
    domain: the pressure-like field ``h1`` is driven by PML-corrected
    Laplacian terms with anisotropy weights S_n that depend on the local
    propagation direction of the wavefront, the spatial gradient of ``h1``
    itself (eqs. 19-23 of the paper). A pure qP equation: no auxiliary ``f``
    field as in the Alkhalifah / eta family, no qSV artefact, and stable for
    epsilon < delta too.

    ``source_type=['h1']`` is the default. The explicit scheme is stable when
    ``dt`` meets eq. B-9 of the paper, which tightens as ``epsilon`` grows;
    ``PropTorch`` warns once when a forward breaks it
    (:func:`qp_stability_ratio`). Also exposed as
    :class:`AcousticAniso(method='liang', symmetry='vti')`.

    Reference: Liang K. et al. 2023, 10.1190/geo2022-0292.1; the S_n operator
    follows Xu and Zhou (2014), 10.1190/geo2014-0242.1.

    Gradient (``s_gradient``). S_n takes its direction from the wavefield, so
    the discrete step is nonlinear in ``h1``; the forward is the same in all
    three modes, the backward differs:

    * ``'adjoint'`` (default): the adjoint of the wavenumber-domain operator
      (eq. 14) is the same qP operator, so the adjoint wavefield is propagated
      with S_n taken from its own gradient; the model gradient is imaged with
      the forward field's S_n and its epsilon / delta derivatives. Well
      conditioned, and the back-propagated residuals travel with the right
      anisotropic kinematics.
    * ``'frozen'``: the transpose of the step with S_n held at the forward
      field's value (the adjoint then travels with the forward wave's
      direction).
    * ``'exact'``: the exact derivative of the discrete forward map, through
      S_n's wavefield direction too. Its adjoint coefficients
      (dS_n/dgrad p ~ 1/|grad p|) spike where the gradient vanishes and do not
      shrink with the amplitude, so it moves by percents with the forward's
      rounding (fp32 against fp64 eager ~2%); full storage only.

    See docs/user-guide/qp_born.md.

    
    """

    # The isotropic image-method free surface is WRONG for an anisotropic
    # medium (see EquationBase.supports_free_surface); fail loud instead.
    supports_free_surface = False

    # Eager boundary saving rebuilds the whole field by reverse stepping with
    # the CPML zeroed, so the absorbing layer holds garbage there.  The qP
    # adjoint is nonlinear in the forward field (S depends on grad p), so that
    # garbage reaches the gradient (4-10% measured).  The compiled path keeps
    # the forward state of the whole CPML frame instead (boundary_frame).
    supports_boundary_saving_eager = False

    C_NAME = "qp_vti2d"
    # no checkpoint backward: the qP adjoint needs the forward state of step
    # it+1 at step it, which the chunk replay of the shared skeleton does not
    # keep; the core refuses checkpointing at forward time.
    C_HAS_RECURSIVE_CKPT = False
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="VTI acoustic reference velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon parameter."),
        ModelSpec("delta", description="Thomsen delta parameter."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic-VTI pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2, s_gradient="adjoint"):
        """Build the 2-D pure-qP VTI equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels used by the anisotropy direction term — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer (``2, 4, 6, 8,
                10, …``). ``impl='c'`` (CUDA) runs the same scheme with full or
                boundary-saving gradients (no checkpointing). Defaults to 4.
            device: Device for the operator's static kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU eager runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class.
                Defaults to 2.
            s_gradient: ``'adjoint'``, ``'frozen'`` or ``'exact'``, how the
                gradient treats S_n (see the class docstring). Defaults to
                ``'adjoint'``.
        """
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
        super().init_grad_kernels()
        self.s_gradient = check_s_gradient(s_gradient, backend)

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hx = self._spacings_2d(h)
        if self.s_gradient == "adjoint":
            rhs, aux = adjoint_mode_step(*self._adjoint_fns(dt, h, b), wavefields, models, h, self.b,
                                         self.gradient, self.grad_kernels)
            return (2 * wavefields[0] - wavefields[1] + rhs, wavefields[0], *aux)
        nabla_z, nabla_x = self.separable_d2_2d(wavefields[0], self.laplace_kernels, hz, hx)
        return step_cpml(*wavefields, *models, dt, h, b, self.b, nabla_x, nabla_z, self.gradient, self.grad_kernels,
                         s_frozen=self.s_gradient == "frozen", **kwargs)

    def _adjoint_fns(self, dt, h, b):
        """(rhs_fn, s_fn, lap_fn) of ``adjoint_mode_step``."""
        hz, hx = self._spacings_2d(h)

        def lap_fn(u):
            return self.separable_d2_2d(u, self.laplace_kernels, hz, hx)

        def rhs_fn(u, psix, psiz, zetax, zetaz, models, s_frozen, s_override):
            lap_z, lap_x = lap_fn(u)
            return rhs_cpml(u, psix, psiz, zetax, zetaz, *models, dt, h, b, self.b, lap_x, lap_z,
                            self.gradient, self.grad_kernels, s_frozen=s_frozen, s_override=s_override)

        def s_fn(field, models):
            _, epsilon, delta = models
            return s_operator(self.gradient(field, h, -1, kernels=self.grad_kernels),
                              self.gradient(field, h, -2, kernels=self.grad_kernels), epsilon, delta)

        return rhs_fn, s_fn, lap_fn

    @property
    def cuda_layout(self):
        return qp_cuda_layout()

    def c_eq_aux(self, prop):
        return s_gradient_aux(self)

    def check_stability(self, models, dt, spacing):
        """Warn once when ``dt`` breaks the scheme's stability condition
        (:func:`qp_stability_ratio`); ``PropTorch`` calls it before every forward."""
        check_qp_stability(self, models, dt, spacing)

    def c_refuse_grads(self, wavelet_grad, model_grad_mask, boundary_saving=False, boundary_dtype=None,
                       checkpoint=False):
        refuse_boundary_saving(self, boundary_saving, boundary_dtype)


def qp_cuda_layout():
    """``cuda_layout`` of the compiled qP pair (csrc/cuda/equations/qp2d).

    The qP adjoint is NONLINEAR in the forward field (S depends on grad p), so
    the backward needs the full forward state of step it+1 -- p and the four
    CPML memories -- at reverse step it:

    * full mode stores that state per step, ``(nt, 5, B, nz, nx)``
      (p, psix, psiz, zetax, zetaz; aux at full grid size);
    * boundary saving stores the same five fields on the whole CPML frame
      (``boundary_frame``: the absorbing pad plus M + 1 cells of the physical
      box) and rebuilds the deep interior by reverse stepping, so the adjoint
      in the absorbing layer sees the true forward state.
    """
    return CUDALayoutSpec(
        record_shape=record_single(),
        save_all_shape=history_fields(5),
        base_nvar=3,
        # psix, psiz, zetax, zetaz (full grids, the frame saver indexes them
        # like p) + psixn, psizn: the forward reads psi at neighbours and
        # writes the next psi to the shadow; zeta is own-cell, in place.
        pml_nvar=6,
        # the adjoint double-buffers zeta too: 3 + 6 + 2 = 11 tensors
        adjoint_extra_nvar=2,
        last_two_nvar=2,
        last_two_storage_nvar=1,
        boundary_save_nvar=5,
        boundary_frame=True,
        # reconstruction: p triple + the restored psix, psiz, zetax, zetaz
        bs_reconstruction_nvar=7,
        # adjoint scratch per step: the two tmp adjoints and the two
        # first-derivative adjoints (qp2d::Scratch)
        backward_workspace_nvar=4,
        grads_out_has_wavelet=True,
    )
