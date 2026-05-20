"""Visualise eager-vs-CUDA backward gradients for AcousticVTI1st3D.

Uses the canonical 3-D sizing from `sweep/test/solver_gradient_mode_suite.py`
so the result lines up with the gradient-consistency suite used for the
2-D companion and the other equations.

  * nz × ny × nx = 24 × 20 × 24       (interior; abcn=30 added around)
  * dh           = 10.0 m
  * dt           = 0.0015 s
  * nt           = 120
  * Ricker       = f=10 Hz, delay=0.06 s
  * source       = (nx//2, ny//2, nz//4)   — interior scenario
  * receivers    = 2-D grid at z=radius, x-stride 6, y-stride 4
  * vp model     = depth ramp 1800→2400 + box +180 (true vs init)
  * rho model    = depth ramp 1000→1200 + box +60
  * eps, delta   = 0.05, 0.03 (near-isotropic so the test stays well-posed)

Produces three PNGs in outputs/:

  * vti3d_gradients_xz_at_ysrc.png — 4×3 grid of XZ-slices through the source y.
  * vti3d_gradients_yz_at_xsrc.png — 4×3 grid of YZ-slices through the source x.
  * vti3d_gradients_xsection.png   — 1-D cross-section through the receiver
                                      depth (x cut at y=ny//2, z=z_rec).

Colour-map limits use the shared `percentile_clip` helper on the
**signed** gradient array (no abs).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st3D
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ----------------------------------------------------------------------
# Canonical sizing (mirrors solver_gradient_mode_suite.py defaults)
# ----------------------------------------------------------------------
NZ, NY, NX = 24, 20, 24
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
FREQ = 10.0
DELAY = 0.06
SPATIAL_ORDER = 4
RADIUS = SPATIAL_ORDER // 2
REC_STRIDE_X = 6
REC_STRIDE_Y = 4

device = "cuda"
if not torch.cuda.is_available():
    raise SystemExit("This script needs CUDA — eager-vs-CUDA gradient comparison "
                     "requires the compiled binding.  Skipping.")

shape = (NZ, NY, NX)

src_x = NX // 2
src_y = NY // 2
src_z = max(1, min(NZ - 1, NZ // 4))
rec_z = max(1, RADIUS)
rec_x = np.arange(max(2, RADIUS),
                  max(max(2, RADIUS) + 1, NX - max(2, RADIUS)),
                  REC_STRIDE_X, dtype=np.int64)
rec_y = np.arange(max(2, RADIUS),
                  max(max(2, RADIUS) + 1, NY - max(2, RADIUS)),
                  REC_STRIDE_Y, dtype=np.int64)
rx, ry = np.meshgrid(rec_x, rec_y, indexing="xy")
rec = np.stack([rx.ravel(), ry.ravel(),
                np.full(rx.size, rec_z, dtype=np.int64)], axis=-1)[None]
src = np.array([[src_x, src_y, src_z]], dtype=np.int64)[None]

t = np.arange(NT, dtype=np.float32) * DT - DELAY
wavelet = torch.tensor(
    (1e6 * ricker(t, f=FREQ)).astype(np.float32)
)[None, None, :].cuda()


# ----------------------------------------------------------------------
# Models (depth-ramped, with a box anomaly inside the volume).
# ----------------------------------------------------------------------
def _ramp(top, bottom):
    depth = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    col = (top + (bottom - top) * depth).astype(np.float32)
    return np.broadcast_to(col[:, None, None], shape).copy()


def _box(arr, val):
    out = arr.copy()
    out[NZ // 3: max(NZ // 3 + 2, (2 * NZ) // 3),
        NY // 4: max(NY // 4 + 2, (3 * NY) // 4),
        NX // 4: max(NX // 4 + 2, (3 * NX) // 4)] += val
    return out


vp_init  = _ramp(1800.0, 2400.0)
vp_true  = _box(vp_init, 180.0)
rho_init = _ramp(1000.0, 1200.0)
rho_true = _box(rho_init, 60.0)
eps_a    = np.full(shape, 0.05, dtype=np.float32)
delta_a  = np.full(shape, 0.03, dtype=np.float32)


def _to(*arrs, req_grad=False):
    return [
        torch.tensor(a, dtype=torch.float32, device=device, requires_grad=req_grad)
        for a in arrs
    ]


# ----------------------------------------------------------------------
# Synthetic target via eager.
# ----------------------------------------------------------------------
def _prop(impl):
    eq = AcousticVTI1st3D(spatial_order=SPATIAL_ORDER, device=device, backend="torch")
    return PropTorch(
        eq, shape,
        source_type=["sH", "sV"], receiver_type=["vz"],
        abcn=ABCN, dh=DH, dt=DT, nt=NT, device=device,
        impl=impl, use_ckpt=False,
    )


print("Running eager forward (target) ...", flush=True)
prop_t = _prop("eager")
true_models = _to(vp_true, eps_a, delta_a, rho_true)
with torch.no_grad():
    target = prop_t(wavelet, src, rec, models=true_models).detach()


def _grads(impl):
    prop = _prop(impl)
    init_models = _to(vp_init, eps_a, delta_a, rho_init, req_grad=True)
    rec_syn = prop(wavelet, src, rec, models=init_models)
    loss = ((rec_syn - target) ** 2).mean()
    gs = torch.autograd.grad(loss, init_models)
    return [g.detach().cpu().numpy() for g in gs]


print("Running eager backward (autograd) ...", flush=True)
g_eager = _grads("eager")

print("Running CUDA backward (Phase 1) ...", flush=True)
g_cuda = _grads("c")


# ----------------------------------------------------------------------
# Per-model metrics (printed to stdout AND embedded in the figure title).
# ----------------------------------------------------------------------
names = ["grad_vp", "grad_eps", "grad_delta", "grad_rho"]
metrics_lines = []
for name, ge, gc in zip(names, g_eager, g_cuda):
    ref  = ge.astype(np.float64).ravel()
    cand = gc.astype(np.float64).ravel()
    ref_l2  = float(np.linalg.norm(ref))
    cand_l2 = float(np.linalg.norm(cand))
    diff_l2 = float(np.linalg.norm(ref - cand))
    rel_l2  = diff_l2 / max(ref_l2, 1e-30)
    cos_sim = float(np.dot(ref, cand) / max(ref_l2 * cand_l2, 1e-30))
    line = (f"  {name:11s}  rel_l2={rel_l2:.3e}  cos_sim={cos_sim:.4f}  "
            f"|ref|_max={float(np.abs(ge).max()):.3e}")
    metrics_lines.append(line)
print("Gradient comparison (eager autograd vs CUDA Phase-1 backward):")
for line in metrics_lines:
    print(line)


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def _imshow(ax, data2d, title, *, cmap="seismic"):
    vmin, vmax = percentile_clip(data2d, percentiles=(2.0, 98.0))
    im = ax.imshow(data2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)


def _save_slice_figure(slice_axis, slice_idx, out_name, axis_label):
    """One PNG: 4 rows (models) × 3 cols (eager, CUDA, diff).

    `slice_axis` is one of 'y' (XZ at y=slice_idx) or 'x' (YZ at x=slice_idx).
    """
    fig, axes = plt.subplots(4, 3, figsize=(11, 12), constrained_layout=True)
    fig.suptitle(
        f"AcousticVTI1st3D backward gradients — eager autograd vs CUDA Phase 1\n"
        f"slice: {axis_label} ; grid nz×ny×nx={NZ}×{NY}×{NX} abcn={ABCN} "
        f"nt={NT} dh={DH}m dt={DT*1e3:.1f}ms",
        fontsize=10,
    )
    for r, (name, ge, gc) in enumerate(zip(names, g_eager, g_cuda)):
        # ge/gc shape: (1, 1, NZ, NY, NX) — squeeze leading batch dims.
        ge_v = ge.reshape(NZ, NY, NX)
        gc_v = gc.reshape(NZ, NY, NX)
        if slice_axis == "y":
            s_ge = ge_v[:, slice_idx, :]      # (NZ, NX)
            s_gc = gc_v[:, slice_idx, :]
        elif slice_axis == "x":
            s_ge = ge_v[:, :, slice_idx]      # (NZ, NY)
            s_gc = gc_v[:, :, slice_idx]
        else:
            raise ValueError(f"slice_axis must be 'y' or 'x', got {slice_axis!r}")
        diff = s_gc - s_ge

        _imshow(axes[r, 0], s_ge, f"{name} — eager")
        _imshow(axes[r, 1], s_gc, f"{name} — CUDA")
        _imshow(axes[r, 2], diff, f"{name} — diff (CUDA − eager)")

    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


print("Plotting ...", flush=True)
_save_slice_figure("y", src_y, "vti3d_gradients_xz_at_ysrc.png",
                   axis_label=f"XZ at y={src_y} (source y)")
_save_slice_figure("x", src_x, "vti3d_gradients_yz_at_xsrc.png",
                   axis_label=f"YZ at x={src_x} (source x)")


# ----------------------------------------------------------------------
# 1-D cross section at the receiver depth, central y.  Quick "where does
# the gradient cluster" sanity plot.
# ----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), constrained_layout=True, sharex=True)
fig.suptitle(
    f"AcousticVTI1st3D backward gradients — 1-D cut at y={NY//2}, z={rec_z}\n"
    f"eager autograd vs CUDA Phase 1",
    fontsize=10,
)
xs = np.arange(NX)
for ax, name, ge, gc in zip(axes, names, g_eager, g_cuda):
    ge_v = ge.reshape(NZ, NY, NX)
    gc_v = gc.reshape(NZ, NY, NX)
    ax.plot(xs, ge_v[rec_z, NY // 2, :], label="eager",  lw=1.4, color="C0")
    ax.plot(xs, gc_v[rec_z, NY // 2, :], label="CUDA",   lw=1.4, color="C3", linestyle="--")
    ax.set_ylabel(name, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("x (grid index)", fontsize=9)
out_path = OUTPUT_DIR / "vti3d_gradients_xsection.png"
plt.savefig(out_path, dpi=140)
plt.close(fig)
print(f"  wrote {out_path}")

print("\nDone.  Metrics:")
for line in metrics_lines:
    print(line)
