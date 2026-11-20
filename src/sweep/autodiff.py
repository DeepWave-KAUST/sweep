"""Jacobian-, Gauss-Newton- and Hessian-vector products of a propagator's misfit.

For a record ``d = F(m)`` of the models ``m`` and a misfit ``J(m) = loss(F(m))``,

    J h                      the Jacobian-vector product (the Born record of h),
    J^T (d2 loss) J h        the Gauss-Newton product,
    H h = J^T (d2 loss) J h + sum_i (d loss / d d_i) d2F_i h,

each from one call, choosing among the routes a propagator can run:

``'forward'``
    Forward-mode AD carries ``h`` through the uncompiled eager step with
    ``memory=Full()``; the tangent of the record is J h.  For H h, the gradient
    of ``<grad loss(d), s.detach()> + <s, grad loss(d).detach()>`` with
    ``s = J h`` is exact and takes first derivatives only (reverse over
    forward).
``'double_backward'``
    H h only: the gradient taken with ``create_graph=True`` and differentiated
    again.  Exact; the uncompiled eager step, ``Full()`` or ``Ckpt()``.
``'difference'``
    Central differences, ``(g(m + e h) - g(m - e h)) / (2 e)`` for H h and
    ``(F(m + e h) - F(m - e h)) / (2 e)`` for J h: two gradients or two
    forwards, with every impl and memory strategy and with ``ModelParallel``.
    The step ``e`` changes no model by more than ``rel_step`` of its peak; a
    model that is all zero (the zero reflectivity LSRTM starts from) counts as
    1, and under ``ModelParallel`` the peaks are those of the whole model, so
    every rank takes the same step from its own tile.  The error is O(e^2)
    plus float32 round-off of O(1 / e): a strongly curved loss needs a smaller
    ``rel_step``, a model whose peak is far below the size of its changes a
    larger one.
``'auto'`` (default)
    ``'forward'`` where the propagator carries a tangent (the uncompiled eager
    step with ``Full()``), else ``'difference'`` -- also on the uncompiled
    step with ``Ckpt()``, where ``'double_backward'`` is exact too.  Forward
    mode is exact but not the fastest: two gradients take less time than
    reverse over forward.

``directions`` holds one tensor per model, or ``None`` for a model held fixed:
it is not perturbed, needs no gradient (``impl='c'`` ``ElasticBorn`` and the qP
Born pair refuse a background model that requires one, as ``AcousticBorn`` does
under checkpointing) and its entry of a product is ``None``.  A zero direction
keeps a model a variable, whose entry is then a cross term.  Directions are
cast to their model's dtype and device.  ``loss`` maps the record to a scalar
and must be twice differentiable for the exact routes; terms of the misfit
that depend on the models directly (a regulariser) are not part of these
products.  An equation can refuse the curvature products (Gauss-Newton and
H h, its ``curvature_refusal``) or the ``'difference'`` route (its
``difference_refusal``) where they are not reliable: the qP equations
(``AcousticVTI``, ``AcousticTTI`` and their Born pair) refuse both, because
where epsilon differs from delta their record and gradient are not smooth
functions of the model.  J h of ``AcousticVTI`` and ``AcousticTTI`` along a
smooth direction still holds by forward mode (``s_gradient='exact'``).

``'forward'`` and ``'difference'`` differentiate the gradient the propagator
returns, which on the outermost ring of model cells is the cropped one (the
model is edge-replicated into the absorbing layer and its backward crops): for
a direction on that ring they give ``C H E h``, which is not symmetric, while
``'double_backward'`` gives the symmetric ``C H C^T h``.  Keep directions off
the outermost ring when the routes are compared or when a solver needs a
symmetric operator.
"""
import torch
import torch.autograd.forward_ad as fwAD

ROUTES = ("auto", "forward", "double_backward", "difference")


def _check_inputs(models, directions):
    """The models and the directions, each direction cast to its model's dtype
    and device (a float64 direction would otherwise reach the float32 step as a
    float64 tangent or model)."""
    models = list(models)
    directions = list(directions)
    if len(directions) != len(models):
        raise ValueError(f"directions has {len(directions)} entries for {len(models)} models; "
                         "give one direction per model (None for a model held fixed).")
    for i, (m, h) in enumerate(zip(models, directions)):
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"models[{i}] must be a tensor, got {type(m).__name__}.")
        if h is not None and tuple(h.shape) != tuple(m.shape):
            raise ValueError(f"directions[{i}] has shape {tuple(h.shape)}; models[{i}] has {tuple(m.shape)}.")
    if all(h is None for h in directions):
        raise ValueError("every direction is None: there is nothing to differentiate along.")
    directions = [None if h is None else h.detach().to(device=m.device, dtype=m.dtype)
                  for m, h in zip(models, directions)]
    return models, directions


def _carries_tangent(propagator):
    """A ``PropTorch`` running the uncompiled eager step on the full tape, with
    an equation and no imaging condition that would refuse a tangent."""
    from sweep.propagator.torch import PropTorch

    if not isinstance(propagator, PropTorch) or propagator.impl != "eager":
        return False
    if getattr(propagator, "use_compile", False) or propagator.memory_strategy != "full":
        return False
    refusal = getattr(propagator.equation, "forward_mode_refusal", None)
    if callable(refusal) and refusal() is not None:
        return False
    registered = getattr(propagator, "_registered_indices", None)
    return not (callable(registered) and registered("gradient"))


def _route(route, propagator, kinds):
    if route not in ROUTES or route not in kinds:
        raise ValueError(f"route must be one of {kinds}, got {route!r}.")
    if route == "auto":
        return "forward" if _carries_tangent(propagator) else "difference"
    return route


def _refuse(propagator, hook, what):
    """Raise when the propagator's equation says ``what`` is not reliable for it
    (``curvature_refusal``, ``difference_refusal``)."""
    equation = getattr(propagator, "equation", None)
    refusal = getattr(equation, hook, None)
    why = refusal() if callable(refusal) else None
    if why is not None:
        raise RuntimeError(f"{type(equation).__name__} has no reliable {what}: {why}.")


def _tile_group(propagator):
    """The process group whose ranks each hold a tile of the model
    (``ModelParallel`` across ranks), else None."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return None
    from sweep.parallel.dd_propagator import ModelParallel

    if not isinstance(propagator, ModelParallel) or propagator.mesh is None:
        return None
    return propagator.mesh.model_pg


def _step(propagator, models, directions, rel_step):
    """The largest e with |e h| <= rel_step * max|m| for every perturbed model
    (max|m| of an all-zero model counts as 1), the peaks taken over every
    tile of a ``ModelParallel`` model -- on its device, the only one its
    process group may serve -- so that every rank takes the same step, or
    raises the same error."""
    if not rel_step > 0:
        raise ValueError(f"rel_step must be positive, got {rel_step!r}.")
    peaks = torch.tensor([[float(m.detach().abs().max()), float(h.abs().max()),
                           float(not bool(torch.isfinite(h).all()))]
                          for m, h in zip(models, directions) if h is not None], dtype=torch.float64)
    group = _tile_group(propagator)
    if group is not None:
        peaks = peaks.to(propagator.dev)
        torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX, group=group)
    peaks = peaks.tolist()
    if any(not_finite > 0 for _, _, not_finite in peaks):
        raise ValueError("a direction is not finite (NaN or inf).")
    steps = [rel_step * (mmax if mmax > 0 else 1.0) / hmax for mmax, hmax, _ in peaks if hmax > 0]
    if not steps:
        raise ValueError("every direction is zero: there is nothing to differentiate along.")
    return min(steps)


def _perturbed(models, directions, e):
    return [m.detach() if h is None else (m.detach() + e * h) for m, h in zip(models, directions)]


def _variables(models, directions):
    """Leaves to differentiate: a model with a direction requires grad, a model
    held fixed does not."""
    return [m.detach() if h is None else m.detach().clone().requires_grad_(True)
            for m, h in zip(models, directions)]


def _grad(output, models, directions, **kwargs):
    """``d output / d models`` for the models with a direction (zeros where
    ``output`` does not depend on one), None for a model held fixed."""
    wrt = [m for m, h in zip(models, directions) if h is not None]
    grads = iter(torch.autograd.grad(output, wrt, allow_unused=True, **kwargs))
    out = []
    for m, h in zip(models, directions):
        g = None if h is None else next(grads)
        out.append(torch.zeros_like(m) if h is not None and g is None else g)
    return out


def _forward_dual(propagator, wavelet, sources, receivers, models, directions):
    with fwAD.dual_level():
        duals = [m if h is None else fwAD.make_dual(m, h) for m, h in zip(models, directions)]
        d, s = fwAD.unpack_dual(propagator(wavelet, sources, receivers, models=duals))
    if s is None:
        s = torch.zeros_like(d)
    return d, s


def jacobian_vector_product(propagator, wavelet, sources, receivers, models, directions, *,
                            route="auto", rel_step=1e-2):
    """``(record, J h)``: the record at ``models`` and its derivative along
    ``directions`` (one per model, ``None`` for a model held fixed)."""
    models, directions = _check_inputs(models, directions)
    route = _route(route, propagator, ("auto", "forward", "difference"))
    models = [m.detach() for m in models]
    with torch.no_grad():
        if route == "forward":
            return _forward_dual(propagator, wavelet, sources, receivers, models, directions)
        _refuse(propagator, "difference_refusal", "product by central differences")
        e = _step(propagator, models, directions, rel_step)
        record = propagator(wavelet, sources, receivers, models=models)
        plus = propagator(wavelet, sources, receivers, models=_perturbed(models, directions, e))
        minus = propagator(wavelet, sources, receivers, models=_perturbed(models, directions, -e))
        return record, (plus - minus) / (2 * e)


def _loss_gradient(loss, record):
    """``grad loss`` at ``record``, with its graph: it requires grad unless the
    loss is linear in the record (second derivative zero)."""
    value = loss(record)
    if value.dim() != 0:
        raise ValueError(f"loss must return a scalar, got shape {tuple(value.shape)}.")
    (y,) = torch.autograd.grad(value, record, create_graph=True)
    return y


def hessian_vector_product(propagator, wavelet, sources, receivers, models, directions, loss, *,
                           route="auto", rel_step=1e-2):
    """``H h`` of ``loss(propagator(..., models))`` along ``directions`` (one per
    model, ``None`` for a model held fixed): one tensor per model, ``None`` for
    a model held fixed."""
    models, directions = _check_inputs(models, directions)
    _refuse(propagator, "curvature_refusal", "Hessian-vector product")
    route = _route(route, propagator, ROUTES)
    with torch.enable_grad():
        if route == "difference":
            _refuse(propagator, "difference_refusal", "product by central differences")
            e = _step(propagator, models, directions, rel_step)

            def gradient(sign):
                ms = _variables(_perturbed(models, directions, sign * e), directions)
                return _grad(loss(propagator(wavelet, sources, receivers, models=ms)), ms, directions)

            plus, minus = gradient(1.0), gradient(-1.0)
            return [None if p is None else (p - q) / (2 * e) for p, q in zip(plus, minus)]

        ms = _variables(models, directions)
        if route == "double_backward":
            value = loss(propagator(wavelet, sources, receivers, models=ms))
            gs = _grad(value, ms, directions, create_graph=True)
            inner = sum((g * h).sum() for g, h in zip(gs, directions) if h is not None)
            return _grad(inner, ms, directions)

        d, s = _forward_dual(propagator, wavelet, sources, receivers, ms, directions)
        y = _loss_gradient(loss, d)
        psi = (s * y.detach()).sum()
        if y.requires_grad:
            psi = psi + (y * s.detach()).sum()
        return _grad(psi, ms, directions)


def gauss_newton_vector_product(propagator, wavelet, sources, receivers, models, directions, loss, *,
                                route="auto", rel_step=1e-2):
    """``J^T (d2 loss) J h`` along ``directions`` (one per model, ``None`` for a
    model held fixed): one tensor per model, ``None`` for a model held fixed;
    positive semi-definite for a convex loss, the product a Gauss-Newton
    method takes."""
    models, directions = _check_inputs(models, directions)
    _refuse(propagator, "curvature_refusal", "Gauss-Newton product")
    d, s = jacobian_vector_product(propagator, wavelet, sources, receivers, models, directions,
                                   route=route, rel_step=rel_step)
    with torch.enable_grad():
        dd = d.detach().requires_grad_(True)
        y = _loss_gradient(loss, dd)
        if not y.requires_grad:              # a loss linear in the record
            return [None if h is None else torch.zeros_like(m) for m, h in zip(models, directions)]
        (w,) = torch.autograd.grad((y * s.detach()).sum(), dd)
        ms = _variables(models, directions)
        record = propagator(wavelet, sources, receivers, models=ms)
        return _grad(record, ms, directions, grad_outputs=w)
