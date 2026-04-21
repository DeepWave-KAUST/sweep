import argparse

import numpy as np
import torch

from common import (
    ABCN,
    ORDERS,
    OUTPUT_DIR,
    PropCUDA,
    crop_panel,
    extract_saved_wavefield,
    make_geometry,
    make_wavelet,
    plot_order_grid,
    require_cuda,
)
from sweep.equations import Acoustic

PHYS_SHAPE = (301, 301)
DH = 10.0
DT = 0.001
NT = 900
DOM_FREQ = 15.0
SNAPSHOT_TIME = 520
VP = 2000.0


def build_solver(order, dev):
    return PropCUDA(
        Acoustic(spatial_order=order, device=dev, backend="torch"),
        shape=PHYS_SHAPE,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=NT,
        abcn=ABCN,
        free_surface=False,
        pml_type="cpmlr",
        use_ckpt=False,
    )


def run_order(order, dev):
    solver = build_solver(order, dev)
    wavelet = make_wavelet(NT, DT, DOM_FREQ, delay=0.15)
    sources, receivers = make_geometry(PHYS_SHAPE)

    # Requiring gradients forces the CUDA wrapper to retain the full time history,
    # which lets this diagnostic script inspect the actual PropCUDA wavefield output.
    vp = torch.full(PHYS_SHAPE, VP, dtype=torch.float32, device=dev, requires_grad=True)

    record = solver(wavelet, sources, receivers, models=[vp])
    wavefield = extract_saved_wavefield(record)
    panel = wavefield[SNAPSHOT_TIME, 0].detach().cpu().numpy()
    return crop_panel(panel, PHYS_SHAPE, order)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PropCUDA acoustic wavefields across FD orders.")
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=list(ORDERS),
        help="Spatial orders to run. Default: 2 6 10 14",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = require_cuda()
    panels = []
    for order in args.orders:
        print(f"Running acoustic CUDA order {order}...")
        panels.append(run_order(order, dev))
    titles = [f"Order {order}" for order in args.orders]
    out_path = OUTPUT_DIR / "acoustic_fd_orders.png"
    plot_order_grid(
        panels,
        titles,
        out_path,
        f"Acoustic CUDA Wavefields at t={SNAPSHOT_TIME * DT:.3f} s",
        PHYS_SHAPE,
        DH,
    )
    print(f"Saved acoustic comparison to {out_path}")


if __name__ == "__main__":
    main()
