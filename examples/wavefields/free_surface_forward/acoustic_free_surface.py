import numpy as np
import torch

from common import OUTPUT_DIR, plot_model, plot_panel_grid, plot_record_comparison
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import EagerOptions
from sweep.signal import ricker


PHYS_SHAPE = (120, 180)
DH = 5.0
DT = 0.001
NT = 500
ABCN = 20
DOM_FREQ = 18.0
SNAPSHOT_TIMES = [150, 250, 380]
SPATIAL_ORDER = 8


def build_solver(free_surface, dev):
    return PropTorch(
        Acoustic(spatial_order=SPATIAL_ORDER, device=dev, backend="torch"),
        shape=PHYS_SHAPE,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=NT,
        abcn=ABCN,
        free_surface=free_surface,
        source_type=["h1"],
        receiver_type=["h1"],
        pml_type="cpmlr",
        backend="eager",
        eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
    )


def build_inputs():
    nz, nx = PHYS_SHAPE
    vp = torch.full((nz, nx), 2200.0, dtype=torch.float32)
    t = np.arange(NT, dtype=np.float32) * DT - 0.12
    wavelet = 1e6 * ricker(t, f=DOM_FREQ).astype(np.float32)
    sources = np.array([[nx // 2, 4]], dtype=np.int64)
    rec_x = np.arange(8, nx - 8, dtype=np.int64)
    receivers = np.stack([rec_x, np.full_like(rec_x, 8)], axis=1)[None, ...]
    return wavelet, sources, receivers, [vp]


def run_case(free_surface):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(free_surface, dev)
    wavelet, sources, receivers, models = build_inputs()
    record, snapshots = solver(
        wavelet,
        sources,
        receivers,
        models=[m.to(dev) for m in models],
        return_wavefield=True,
        snapshot_times=SNAPSHOT_TIMES,
    )
    # snapshots: (nsnap, nfields, B, 1, nz, nx)
    panels = [
        crop_panel(snapshots[i, 0, 0, 0].detach().cpu().numpy(), free_surface)
        for i in range(len(SNAPSHOT_TIMES))
    ]
    record_panel = to_record_panel(record)
    return panels, record_panel, sources, receivers, models[0].numpy()


def crop_panel(panel, free_surface):
    if free_surface:
        return panel[: PHYS_SHAPE[0], ABCN : ABCN + PHYS_SHAPE[1]]
    return panel[ABCN : ABCN + PHYS_SHAPE[0], ABCN : ABCN + PHYS_SHAPE[1]]


def to_record_panel(record):
    data = record.detach().cpu().numpy()
    if data.ndim == 4:
        return data[0, :, :, 0]
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Unexpected record shape {data.shape}")


def main():
    fs_panels, fs_record, sources, receivers, model = run_case(True)
    abs_panels, abs_record, _, _, _ = run_case(False)
    plot_model(
        model,
        sources,
        receivers,
        OUTPUT_DIR / "acoustic_model.png",
        "Acoustic Free-Surface Test Model",
        DH,
        "Vp (m/s)",
    )
    plot_panel_grid(
        [fs_panels, abs_panels],
        ["Free Surface", "Absorbing Top"],
        SNAPSHOT_TIMES,
        OUTPUT_DIR / "acoustic_free_surface_snapshots.png",
        "Acoustic Forward Snapshots",
        DH,
    )
    plot_record_comparison(
        [fs_record, abs_record, fs_record - abs_record],
        ["Free Surface", "Absorbing Top", "Residual"],
        OUTPUT_DIR / "acoustic_free_surface_records.png",
        "Acoustic Seismogram Comparison",
    )
    print(f"Saved acoustic free-surface figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
