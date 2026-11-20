"""Refusing ``create_graph=True`` and forward-mode tangents where the derivative is off the tape.

A Hessian-vector product by double backward differentiates the first backward
a second time, so it needs that backward to be on the autograd tape.  The
compiled adjoint (impl='c', domain decomposition), the eager boundary-saving
step and the custom-gradient bridge compute their gradient off the tape: under
``create_graph=True`` the gradient used to come back without a graph, which
surfaced later as "element 0 of tensors does not require grad" -- or, when the
loss had another term on the tape (a regulariser), as a Hessian of that term
alone, silently.  Each of them now raises here instead, naming what works.

Forward-mode AD (``torch.autograd.forward_ad``) carries a tangent through every
op, so the tangent of the record is the Jacobian-vector product J h.  The same
paths cannot carry one, and the compiled eager step dropped it without a word
(its tangent came back as zero); they refuse a dual tensor here instead.
"""
import torch
import torch.autograd.forward_ad as fwAD

HESSIAN_ALTERNATIVES = (
    "For an exact Hessian-vector product by double backward use impl='eager' with "
    "memory=Full() or Ckpt() and eager_options=EagerOptions(use_compile=False). At any "
    "size, take H h from first derivatives: exactly as the gradient of the Born system "
    "(AcousticBorn, impl='c' with BoundarySaving()), or as a central difference of two "
    "gradients, (g(m + eps*h) - g(m - eps*h)) / (2*eps), with every impl and memory strategy "
    "(examples/notebooks/46_hessian_vector_product.ipynb).")

DIFFERENCE_ALTERNATIVES = (
    "Take H h as a central difference of two gradients, (g(m + eps*h) - g(m - eps*h)) / (2*eps), "
    "which runs with every impl and memory strategy "
    "(examples/notebooks/46_hessian_vector_product.ipynb).")

JVP_ALTERNATIVES = (
    "Forward-mode AD runs through impl='eager' with memory=Full() and "
    "eager_options=EagerOptions(use_compile=False); a Born equation (AcousticBorn, "
    "ElasticBorn, ...) gives J h with every impl and memory strategy.")


def refuse_create_graph(what, why, instead=HESSIAN_ALTERNATIVES):
    """Raise if this backward runs under ``create_graph=True`` (grad mode is on
    inside a backward only then).  ``what`` names the path, ``why`` says why its
    gradient is off the tape."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            f"{what} gives first derivatives only and is not twice-differentiable: {why}. "
            f"create_graph=True (Hessian-vector products, higher-order gradients) needs "
            f"the backward on the autograd tape. {instead}")


def refuse_second_derivative(what, why, instead=DIFFERENCE_ALTERNATIVES):
    """Raise for a second derivative through a path whose create_graph=True
    gradient is a valid first-order graph (it still serves Jacobian-vector
    products by differentiating it with respect to the residual) but whose
    derivative with respect to its inputs is not a Hessian."""
    raise RuntimeError(
        f"{what} is not twice-differentiable: {why}. Its create_graph=True gradient "
        f"serves first-order uses (differentiating it with respect to the residual, "
        f"as a Gauss-Newton solver does), but a second derivative with respect to "
        f"the models or the wavelet is refused. {instead}")


def substitute_for(equation, default):
    """What a refusal recommends instead: ``default``, unless the equation says
    a difference of its gradients does not converge either (its
    ``difference_refusal``)."""
    refusal = getattr(equation, "difference_refusal", None)
    why = refusal() if callable(refusal) else None
    return default if why is None else f"A difference of gradients is no substitute here: {why}."


def refuse_forward_ad(what, why, instead=JVP_ALTERNATIVES):
    """Raise for a forward-mode tangent on a path that cannot carry one (the
    ``jvp`` of its autograd.Function, or a dual tensor seen on entry)."""
    raise RuntimeError(
        f"{what} does not carry forward-mode tangents (torch.autograd.forward_ad, "
        f"Jacobian-vector products): {why}. {instead}")


def has_tangent(*tensors):
    """True if any of ``tensors`` is a dual tensor of the current forward-AD level."""
    return any(isinstance(t, torch.Tensor) and fwAD.unpack_dual(t).tangent is not None
               for t in tensors)


class CreateGraphGuard(torch.autograd.Function):
    """Identity on ``tensors`` whose backward first calls ``check(ctx)``.

    It sits on the input side of the rollout (the models after their padding,
    the wavelet), so a backward reaches it last.  ``check`` can refuse
    ``create_graph=True`` there for a path whose own backward nodes are off the
    tape or not ours to edit (a torch.compile'd step, eager boundary saving, an
    equation whose derivative is not twice-differentiable) -- raising in a node
    at the start of the graph instead leaves the rest of the engine's work
    queued, and the interpreter then aborted at exit.  A second derivative
    reaches the guard a second time (through the forward graph), a derivative
    of the gradient with respect to the residual does not; ``ctx`` lets
    ``check`` tell the two apart.  Its tangent is the input's."""

    @staticmethod
    def forward(ctx, check, *tensors):
        ctx.check = check
        # views, not the inputs as-is: forward-mode AD then requires (and gets)
        # views as tangents too, for leaf dual inputs as well
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        ctx.check(ctx)
        return (None, *grads)

    @staticmethod
    def jvp(ctx, check_tangent, *tangents):
        return tuple(None if t is None else t.view_as(t) for t in tangents)
