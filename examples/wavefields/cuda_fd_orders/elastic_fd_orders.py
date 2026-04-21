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
from sweep.equations import Elastic

PHYS_SHAPE = (401, 401)
DH = 10.0
DT = 0.0008
NT = 1200
DOM_FREQ = 18.0
SNAPSHOT_TIME = 700
VP = 2200.0
VS = 1000.0
RHO = 2000.0


def build_solver(order, dev):
    return PropCUDA(
        Elastic(spatial_order=order, device=dev, backend="torch"),
        shape=PHYS_SHAPE,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=NT,
        abcn=ABCN,
        source_type=["sxx", "szz"],
        receiver_type=["vx"],
        free_surface=False,
        pml_type="cpmls",
        use_ckpt=False,
    )


def run_order(order, dev):
    solver = build_solver(order, dev)
    wavelet = make_wavelet(NT, DT, DOM_FREQ, delay=0.16)
    sources, receivers = make_geometry(PHYS_SHAPE)

    vp = torch.full(PHYS_SHAPE, VP, dtype=torch.float32, device=dev, requires_grad=True)
    vs = torch.full(PHYS_SHAPE, VS, dtype=torch.float32, device=dev)
    rho = torch.full(PHYS_SHAPE, RHO, dtype=torch.float32, device=dev)

    record = solver(wavelet, sources, receivers, models=[vp, vs, rho])
    wavefield = extract_saved_wavefield(record)
    vx_panel = crop_panel(wavefield[SNAPSHOT_TIME, 0, 0].detach().cpu().numpy(), PHYS_SHAPE, order)
    vz_panel = crop_panel(wavefield[SNAPSHOT_TIME, 1, 0].detach().cpu().numpy(), PHYS_SHAPE, order)
    return vx_panel, vz_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PropCUDA elastic wavefields across FD orders.")
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
    vx_panels = []
    vz_panels = []
    for order in args.orders:
        print(f"Running elastic CUDA order {order}...")
        vx_panel, vz_panel = run_order(order, dev)
        vx_panels.append(vx_panel)
        vz_panels.append(vz_panel)

    titles = [f"Order {order}" for order in args.orders]
    vx_path = OUTPUT_DIR / "elastic_fd_orders_vx.png"
    vz_path = OUTPUT_DIR / "elastic_fd_orders_vz.png"
    plot_order_grid(
        vx_panels,
        titles,
        vx_path,
        f"Elastic CUDA Vx at t={SNAPSHOT_TIME * DT:.3f} s",
        PHYS_SHAPE,
        DH,
    )
    plot_order_grid(
        vz_panels,
        titles,
        vz_path,
        f"Elastic CUDA Vz at t={SNAPSHOT_TIME * DT:.3f} s",
        PHYS_SHAPE,
        DH,
    )
    print(f"Saved elastic comparisons to {vx_path} and {vz_path}")


if __name__ == "__main__":
    main()
