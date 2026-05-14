#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import DASZhao, DASZhao3D
from sweep.propagator.torch import PropTorch


def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * np.float32(dt) - np.float32(delay)
    arg = np.pi * np.float32(fm) * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def clip_limits(record, percentile=(2.0, 98.0)):
    finite = np.asarray(record)[np.isfinite(record)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        scale = max(float(np.max(np.abs(finite))), 1.0)
        return -scale, scale
    if vmin < 0.0 < vmax:
        scale = max(abs(float(vmin)), abs(float(vmax)))
        return -scale, scale
    return float(vmin), float(vmax)


def normalize_record(record):
    return record[0].detach().cpu().numpy().transpose(1, 0, 2)


def layered_2d(nz, nx, dh):
    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dh)
    vp = np.empty((nz, nx), dtype=np.float32)
    vs = np.empty((nz, nx), dtype=np.float32)
    for mask, vp_value, vs_value in [
        (z_m < 750.0, 1500.0, 1000.0),
        ((z_m >= 750.0) & (z_m < 1500.0), 2500.0, 1400.0),
        (z_m >= 1500.0, 3000.0, 1600.0),
    ]:
        vp[mask[:, 0], :] = vp_value
        vs[mask[:, 0], :] = vs_value
    rho = np.full_like(vp, 2100.0, dtype=np.float32)
    return vp, vs, rho


def layered_3d(nz, ny, nx, dh):
    z_m = np.arange(nz, dtype=np.float32)[:, None, None] * np.float32(dh)
    vp = np.empty((nz, ny, nx), dtype=np.float32)
    vs = np.empty((nz, ny, nx), dtype=np.float32)
    for mask, vp_value, vs_value in [
        (z_m < 200.0, 1500.0, 1000.0),
        ((z_m >= 200.0) & (z_m < 400.0), 2500.0, 1400.0),
        (z_m >= 400.0, 3000.0, 1600.0),
    ]:
        vp[mask[:, 0, 0], :, :] = vp_value
        vs[mask[:, 0, 0], :, :] = vs_value
    rho = np.full_like(vp, 2100.0, dtype=np.float32)
    return vp, vs, rho


def geometry_2d(nz, nx, dh):
    source = np.array([[[round(2.0 * 1000.0 / dh), 0]]], dtype=np.int32)
    surface = np.stack([np.arange(nx, dtype=np.int32), np.zeros(nx, dtype=np.int32)], axis=-1)
    horizontal_x = np.linspace(round(0.5 * 1000.0 / dh), round(2.5 * 1000.0 / dh), 201, dtype=np.int32)
    horizontal = np.stack([horizontal_x, np.full(horizontal_x.size, round(1.2 * 1000.0 / dh), dtype=np.int32)], axis=-1)
    vertical_z = np.linspace(0, nz - 1, 201, dtype=np.int32)
    vertical = np.stack([np.full(vertical_z.size, round(3.0 * 1000.0 / dh), dtype=np.int32), vertical_z], axis=-1)
    receivers = np.concatenate([surface, horizontal, vertical], axis=0)[None, ...]
    slices = {
        "surface": slice(0, surface.shape[0]),
        "horizontal": slice(surface.shape[0], surface.shape[0] + horizontal.shape[0]),
        "vertical": slice(surface.shape[0] + horizontal.shape[0], receivers.shape[1]),
    }
    return source, receivers, slices


def geometry_3d(nz, ny, nx):
    sx, sy, sz = nx // 2, ny // 2, 4
    source = np.array([[[sx, sy, sz]]], dtype=np.int32)
    inline_x = np.arange(nx, dtype=np.int32)
    inline = np.stack([inline_x, np.full(nx, sy, dtype=np.int32), np.full(nx, sz, dtype=np.int32)], axis=-1)
    crossline_y = np.arange(ny, dtype=np.int32)
    crossline = np.stack([np.full(ny, sx, dtype=np.int32), crossline_y, np.full(ny, sz, dtype=np.int32)], axis=-1)
    vertical_z = np.arange(nz, dtype=np.int32)
    vertical = np.stack([np.full(nz, int(round(0.75 * (nx - 1))), dtype=np.int32), np.full(nz, sy, dtype=np.int32), vertical_z], axis=-1)
    receivers = np.concatenate([inline, crossline, vertical], axis=0)[None, ...]
    slices = {
        "inline": slice(0, inline.shape[0]),
        "crossline": slice(inline.shape[0], inline.shape[0] + crossline.shape[0]),
        "vertical": slice(inline.shape[0] + crossline.shape[0], receivers.shape[1]),
    }
    return source, receivers, slices


def plot_records(records, slices, channels, rows, cols, duration, out_path):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.9 * len(cols), 3.2 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]
    for row_index, (row_name, row_title) in enumerate(rows):
        sl = slices[row_name]
        for col_index, (field, title) in enumerate(cols):
            record = records[sl, :, channels[field]]
            vmin, vmax = clip_limits(record)
            ax = axes[row_index, col_index]
            ax.imshow(
                record.T,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, record.shape[0] - 1, duration, 0],
            )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time, s")
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.text(
                    0.02,
                    0.94,
                    row_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
                )
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_2d_model(vp, source, receivers, slices, dh, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0), constrained_layout=True)
    extent = [0, vp.shape[1] * dh / 1000.0, vp.shape[0] * dh / 1000.0, 0]
    im = ax.imshow(vp / 1000.0, cmap="viridis", aspect="auto", extent=extent)
    colors = {"surface": "lime", "horizontal": "black", "vertical": "dodgerblue"}
    for name, sl in slices.items():
        pts = receivers[0, sl]
        ax.scatter(pts[:, 0] * dh / 1000.0, pts[:, 1] * dh / 1000.0, s=8, c=colors[name], linewidths=0, label=name)
    ax.scatter(source[0, 0, 0] * dh / 1000.0, source[0, 0, 1] * dh / 1000.0, marker="*", s=100, c="red", edgecolors="black", linewidths=0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Position, km")
    ax.set_ylabel("Depth, km")
    fig.colorbar(im, ax=ax, label="Vp, km/s")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_3d_model_slice(vp, source, receivers, slices, dh, out_path):
    mid_y = vp.shape[1] // 2
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), constrained_layout=True)
    extent = [0, vp.shape[2] * dh / 1000.0, vp.shape[0] * dh / 1000.0, 0]
    im = ax.imshow(vp[:, mid_y, :] / 1000.0, cmap="viridis", aspect="auto", extent=extent)
    for name in ["inline", "vertical"]:
        pts = receivers[0, slices[name]]
        ax.scatter(pts[:, 0] * dh / 1000.0, pts[:, 2] * dh / 1000.0, s=12, linewidths=0, label=name)
    ax.scatter(source[0, 0, 0] * dh / 1000.0, source[0, 0, 2] * dh / 1000.0, marker="*", s=100, c="red", edgecolors="black", linewidths=0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("x, km")
    ax.set_ylabel("z, km")
    ax.set_title("3D model y-mid slice")
    fig.colorbar(im, ax=ax, label="Vp, km/s")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def run_2d(args, output_dir, device):
    nz, nx = 201, 401
    vp_np, vs_np, rho_np = layered_2d(nz, nx, args.dh)
    source, receivers, slices = geometry_2d(nz, nx, args.dh)
    nt = int(round(args.duration2d / args.dt2d))
    wavelet = ricker(nt, args.dt2d, args.freq2d, args.delay2d).reshape(1, 1, nt)
    solver = PropTorch(
        DASZhao(spatial_order=args.spatial_order2d, device=device, backend="torch"),
        shape=(nz, nx),
        source_type=["sxx", "szz"],
        receiver_type=["exx_t", "ezz_t", "das35_t", "das54x_t", "das54z_t"],
        abcn=args.abcn2d,
        dh=args.dh,
        dt=args.dt2d,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )
    models = [torch.as_tensor(arr, dtype=torch.float32, device=device) for arr in [vp_np, vs_np, rho_np]]
    start = time.perf_counter()
    with torch.no_grad():
        record = solver(wavelet, sources=source, receivers=receivers, models=models)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    records = normalize_record(record)
    channels = {name: i for i, name in enumerate(solver.receiver_type)}
    np.savez_compressed(output_dir / "das_2d_records.npz", records=records, receiver_type=np.asarray(solver.receiver_type), source=source, receivers=receivers)
    plot_records(
        records,
        slices,
        channels,
        [("surface", "surface"), ("horizontal", "horizontal well"), ("vertical", "vertical well")],
        [("exx_t", "exx_t"), ("ezz_t", "ezz_t"), ("das35_t", "das35_t"), ("das54z_t", "das54z_t")],
        args.duration2d,
        output_dir / "das_2d_shot_records.png",
    )
    plot_2d_model(vp_np, source, receivers, slices, args.dh, output_dir / "das_2d_geometry.png")
    return {"shape": [nz, nx], "nt": nt, "elapsed_s": elapsed, "receiver_type": solver.receiver_type}


def run_3d(args, output_dir, device):
    nz, ny, nx = args.nz3d, args.ny3d, args.nx3d
    vp_np, vs_np, rho_np = layered_3d(nz, ny, nx, args.dh)
    source, receivers, slices = geometry_3d(nz, ny, nx)
    nt = int(round(args.duration3d / args.dt3d))
    wavelet = ricker(nt, args.dt3d, args.freq3d, args.delay3d).reshape(1, 1, nt)
    solver = PropTorch(
        DASZhao3D(spatial_order=args.spatial_order3d, device=device, backend="torch"),
        shape=(nz, ny, nx),
        source_type=["sxx", "syy", "szz"],
        receiver_type=["exx_t", "eyy_t", "ezz_t", "das35_t", "das54x_t", "das54y_t", "das54z_t"],
        abcn=args.abcn3d,
        dh=args.dh,
        dt=args.dt3d,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )
    models = [torch.as_tensor(arr, dtype=torch.float32, device=device) for arr in [vp_np, vs_np, rho_np]]
    start = time.perf_counter()
    with torch.no_grad():
        record = solver(wavelet, sources=source, receivers=receivers, models=models)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    records = normalize_record(record)
    channels = {name: i for i, name in enumerate(solver.receiver_type)}
    np.savez_compressed(output_dir / "das_3d_records.npz", records=records, receiver_type=np.asarray(solver.receiver_type), source=source, receivers=receivers)
    plot_records(
        records,
        slices,
        channels,
        [("inline", "inline x"), ("crossline", "crossline y"), ("vertical", "vertical z")],
        [("exx_t", "exx_t"), ("eyy_t", "eyy_t"), ("ezz_t", "ezz_t"), ("das35_t", "das35_t")],
        args.duration3d,
        output_dir / "das_3d_shot_records.png",
    )
    plot_records(
        records,
        slices,
        channels,
        [("inline", "inline x"), ("crossline", "crossline y"), ("vertical", "vertical z")],
        [("das54x_t", "das54x_t"), ("das54y_t", "das54y_t"), ("das54z_t", "das54z_t")],
        args.duration3d,
        output_dir / "das_3d_helical54_shot_records.png",
    )
    plot_3d_model_slice(vp_np, source, receivers, slices, args.dh, output_dir / "das_3d_geometry_slice.png")
    return {"shape": [nz, ny, nx], "nt": nt, "elapsed_s": elapsed, "receiver_type": solver.receiver_type}


def main():
    parser = argparse.ArgumentParser(description="Generate 2D and 3D DAS shot records with CPML.")
    parser.add_argument("--output-dir", type=Path, default=Path("test/test_outputs/das_shot_records"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--dt2d", type=float, default=0.001)
    parser.add_argument("--duration2d", type=float, default=4.0)
    parser.add_argument("--freq2d", type=float, default=10.0)
    parser.add_argument("--delay2d", type=float, default=0.08)
    parser.add_argument("--spatial-order2d", type=int, default=8)
    parser.add_argument("--abcn2d", type=int, default=50)
    parser.add_argument("--nz3d", type=int, default=61)
    parser.add_argument("--ny3d", type=int, default=61)
    parser.add_argument("--nx3d", type=int, default=61)
    parser.add_argument("--dt3d", type=float, default=0.001)
    parser.add_argument("--duration3d", type=float, default=0.6)
    parser.add_argument("--freq3d", type=float, default=12.0)
    parser.add_argument("--delay3d", type=float, default=0.04)
    parser.add_argument("--spatial-order3d", type=int, default=4)
    parser.add_argument("--abcn3d", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    metadata = {
        "device": str(device),
        "dh_m": args.dh,
        "pml_type": "cpmls",
        "two_d": run_2d(args, output_dir, device),
        "three_d": run_3d(args, output_dir, device),
        "outputs": [
            "das_2d_geometry.png",
            "das_2d_shot_records.png",
            "das_2d_records.npz",
            "das_3d_geometry_slice.png",
            "das_3d_shot_records.png",
            "das_3d_helical54_shot_records.png",
            "das_3d_records.npz",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
