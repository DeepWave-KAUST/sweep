"""Uniform-tile padding for domain decomposition.

v1 DD requires uniform tiles: ``Nx`` must be a multiple of ``px`` and
``Ny`` (3-D) a multiple of ``py``. Models must never be cropped to fit;
instead, :func:`pad_to_mesh` pads the split axes UP to the next multiple,
replicating the edge so the invented cells introduce no impedance contrast.

Two invariants make the pad transparent to everything around it:

* **High side only, and z never padded.** Every existing grid index still
  addresses the same physical cell, so source / receiver coordinates,
  water masks and any other index-based bookkeeping remain valid as-is.

* **Differentiable, with the replicate adjoint.** For torch tensors the
  pad is built from ``cat`` + ``expand``, so gradients that land on the
  invented cells accumulate back onto the physical edge cells they were
  copied from. Keep the UNPADDED tensor as the optimisation variable and
  apply :func:`pad_to_mesh` inside the forward/loss closure::

      vp = torch.tensor(vp_np, requires_grad=True)      # physical grid
      loss = misfit(ddp(wav, src, rec,
                        models=[pad_to_mesh(vp, mesh)]))
      loss.backward()                                    # vp.grad: physical grid

  The optimiser state, ``vp.grad`` and every checkpoint then live on the
  physical grid; resuming under a different mesh just pads differently on
  the next run, with no strip / re-pad bookkeeping anywhere.

Note the padded run is a (slightly) different physical problem from the
unpadded one — the domain is a few cells longer before the high-side
absorbing boundary. DD-vs-single-GPU parity therefore requires padding
BOTH sides of the comparison; that is why :class:`~sweep.parallel.ModelParallel`
never pads implicitly and the divisibility check stays loud.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["pad_to_mesh", "unpad_from_mesh"]


def _factors(mesh, py: int, px: int) -> Tuple[int, int]:
    if mesh is not None:
        if py != 1 or px != 1:
            raise ValueError("pass either mesh or py/px, not both")
        py, px = int(mesh.py), int(mesh.px)
    if py < 1 or px < 1:
        raise ValueError(f"py={py} and px={px} must be >= 1")
    return py, px


def _pad_amounts(shape: Sequence[int], py: int, px: int) -> Tuple[int, int]:
    """(dy, dx) high-side pad for the trailing two axes of ``shape``."""
    if len(shape) < 2:
        raise ValueError(f"model must be at least 2-D, got shape {tuple(shape)}")
    if len(shape) == 2 and py != 1:
        # trailing two of a 2-D model are (nz, nx); padding nz would move
        # the free surface / sea floor. 2-D DD is x-cut only.
        raise ValueError(f"2-D DD requires py=1, got py={py}")
    dy = (-int(shape[-2])) % py
    dx = (-int(shape[-1])) % px
    return dy, dx


def pad_to_mesh(model, mesh=None, *, py: int = 1, px: int = 1):
    """Pad the trailing split axes up to the mesh's tile multiple.

    Parameters
    ----------
    model : torch.Tensor or numpy.ndarray
        ``(nz, nx)``, ``(nz, ny, nx)``, or the same with leading batch /
        parameter axes. Trailing axes are padded on the HIGH side only:
        the last axis to a multiple of ``px``, the second-to-last to a
        multiple of ``py``. The depth axis is never a split axis and is
        never padded.
    mesh : MeshTopology or ModelParallelMesh, optional
        Source of ``py`` / ``px``. Mutually exclusive with the keywords.

    Returns
    -------
    Same type as ``model``; the input object itself when no pad is needed.
    Padding replicates the edge (``np.pad(mode="edge")`` semantics), so
    the invented cells add no impedance contrast, and for torch tensors
    the operation is differentiable with the replicate adjoint (pad-cell
    gradients sum back onto the edge cells).
    """
    py, px = _factors(mesh, py, px)
    dy, dx = _pad_amounts(model.shape, py, px)
    if dy == 0 and dx == 0:
        return model

    if isinstance(model, torch.Tensor):
        out = model
        if dy:
            row = out[..., -1:, :].expand(*out.shape[:-2], dy, out.shape[-1])
            out = torch.cat([out, row], dim=-2)
        if dx:
            col = out[..., -1:].expand(*out.shape[:-1], dx)
            out = torch.cat([out, col], dim=-1)
        return out

    arr = np.asarray(model)
    widths = [(0, 0)] * (arr.ndim - 2) + [(0, dy), (0, dx)]
    return np.pad(arr, widths, mode="edge")


def unpad_from_mesh(arr, orig_shape: Sequence[int], mesh=None, *,
                    py: int = 1, px: int = 1):
    """Slice a padded array back to the physical grid — loudly.

    ``orig_shape`` is the trailing physical grid shape (2 or 3 ints).
    The trailing shape of ``arr`` must equal exactly what
    :func:`pad_to_mesh` would produce for this mesh; anything else (a
    stale shape after resampling, the wrong mesh) raises instead of
    silently mis-cropping. Leading batch axes pass through unchanged.
    """
    py, px = _factors(mesh, py, px)
    orig = tuple(int(s) for s in orig_shape)
    dy, dx = _pad_amounts(orig, py, px)
    expect = orig[:-2] + (orig[-2] + dy, orig[-1] + dx)
    got = tuple(int(s) for s in arr.shape[-len(orig):])
    if got != expect:
        raise ValueError(
            f"trailing shape {got} does not match the padded_shape {expect} "
            f"expected for orig_shape={orig} under py={py}, px={px}"
        )
    if got == orig:
        return arr
    index = (Ellipsis,) + tuple(slice(0, s) for s in orig)
    return arr[index]
