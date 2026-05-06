#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import DASElastic, Elastic, gauge_average
from sweep.propagator.torch import PropTorch


class ElasticStressReceiver(Elastic):
    """Elastic variant used only to expose stress fields as diagnostic receivers."""

    FIELD_SPECS = tuple(
        replace(spec, supports_receiver=True)
        if spec.name in {"sxx", "szz"}
        else spec
        for spec in Elastic.FIELD_SPECS
    )


def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt - np.float32(delay)
    arg = np.pi * np.float32(fm) * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def layered_model(nz=201, nx=401, dh=10.0):
    """Three-layer model matching the Fig. 3 interfaces in Zhao et al. (2026)."""

    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dh)
    vp = np.empty((nz, nx), dtype=np.float32)
    vs = np.empty((nz, nx), dtype=np.float32)

    layers = [
        (z_m < 750.0, 1500.0, 1000.0),
        ((z_m >= 750.0) & (z_m < 1500.0), 2500.0, 1400.0),
        (z_m >= 1500.0, 3000.0, 1600.0),
    ]
    for mask, vp_value, vs_value in layers:
        vp[mask[:, 0], :] = vp_value
        vs[mask[:, 0], :] = vs_value

    rho = np.full_like(vp, 2100.0, dtype=np.float32)
    return vp, vs, rho


FIG3_GEOMETRY_KM = {
    "source": (2.0, 0.0),
    "surface": ((0.0, 0.0), (4.0, 0.0), 401),
    "horizontal_well": ((0.5, 1.2), (2.5, 1.2), 201),
    "vertical_well": ((3.0, 0.0), (3.0, 2.0), 201),
}


def km_to_index(value_km, dh, upper):
    return int(np.clip(round(float(value_km) * 1000.0 / float(dh)), 0, upper))


def build_layered_geometry(
    nz,
    nx,
    dh,
    *,
    source_x_km=FIG3_GEOMETRY_KM["source"][0],
    source_depth_km=FIG3_GEOMETRY_KM["source"][1],
    surface_depth_km=FIG3_GEOMETRY_KM["surface"][0][1],
    horizontal_depth_km=FIG3_GEOMETRY_KM["horizontal_well"][0][1],
    horizontal_x_min_km=FIG3_GEOMETRY_KM["horizontal_well"][0][0],
    horizontal_x_max_km=FIG3_GEOMETRY_KM["horizontal_well"][1][0],
    vertical_x_km=FIG3_GEOMETRY_KM["vertical_well"][0][0],
):
    """Receiver layout digitized from Fig. 3 of Zhao et al. (2026)."""

    source_x = km_to_index(source_x_km, dh, nx - 1)
    source_depth = km_to_index(source_depth_km, dh, nz - 1)
    source = np.array([[[source_x, source_depth]]], dtype=np.int32)

    surface_z = km_to_index(surface_depth_km, dh, nz - 1)
    surface = np.stack(
        [np.arange(nx, dtype=np.int32), np.full(nx, surface_z, dtype=np.int32)],
        axis=-1,
    )

    horizontal_z = km_to_index(horizontal_depth_km, dh, nz - 1)
    horizontal_x0 = km_to_index(horizontal_x_min_km, dh, nx - 1)
    horizontal_x1 = km_to_index(horizontal_x_max_km, dh, nx - 1)
    if horizontal_x0 > horizontal_x1:
        horizontal_x0, horizontal_x1 = horizontal_x1, horizontal_x0
    horizontal_x = np.linspace(horizontal_x0, horizontal_x1, 201, dtype=np.int32)
    horizontal = np.stack(
        [horizontal_x, np.full(horizontal_x.size, horizontal_z, dtype=np.int32)],
        axis=-1,
    )

    vertical_x = km_to_index(vertical_x_km, dh, nx - 1)
    vertical_z = np.linspace(0, nz - 1, 201, dtype=np.int32)
    vertical = np.stack(
        [np.full(vertical_z.size, vertical_x, dtype=np.int32), vertical_z],
        axis=-1,
    )

    receivers = np.concatenate([surface, horizontal, vertical], axis=0)[None, ...]
    slices = {
        "surface": slice(0, surface.shape[0]),
        "horizontal_well": slice(surface.shape[0], surface.shape[0] + horizontal.shape[0]),
        "vertical_well": slice(surface.shape[0] + horizontal.shape[0], receivers.shape[1]),
    }
    geometry = {
        "source": source[0, 0].copy(),
        "surface": surface,
        "horizontal_well": horizontal,
        "vertical_well": vertical,
        "receivers": receivers,
        "slices": slices,
        "meta": {
            "source_km": [source_x * dh / 1000.0, source_depth * dh / 1000.0],
            "surface_km": {
                "start": [surface[0, 0] * dh / 1000.0, surface[0, 1] * dh / 1000.0],
                "end": [surface[-1, 0] * dh / 1000.0, surface[-1, 1] * dh / 1000.0],
                "ntraces": int(surface.shape[0]),
            },
            "horizontal_well_km": {
                "start": [horizontal[0, 0] * dh / 1000.0, horizontal[0, 1] * dh / 1000.0],
                "end": [horizontal[-1, 0] * dh / 1000.0, horizontal[-1, 1] * dh / 1000.0],
                "ntraces": int(horizontal.shape[0]),
            },
            "vertical_well_km": {
                "start": [vertical[0, 0] * dh / 1000.0, vertical[0, 1] * dh / 1000.0],
                "end": [vertical[-1, 0] * dh / 1000.0, vertical[-1, 1] * dh / 1000.0],
                "ntraces": int(vertical.shape[0]),
            },
        },
    }
    return geometry


def clip_limits(record, percentile=(2.0, 98.0)):
    finite = np.asarray(record)[np.isfinite(record)]
    if finite.size == 0:
        return -1.0, 1.0
    if np.isscalar(percentile):
        percentiles = (100.0 - float(percentile), float(percentile))
    else:
        percentiles = tuple(float(value) for value in percentile)
    vmin, vmax = np.percentile(finite, percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        scale = max(float(np.max(np.abs(finite))), 1.0)
        return -scale, scale
    if vmin < 0.0 < vmax:
        scale = max(abs(float(vmin)), abs(float(vmax)))
        return -scale, scale
    return float(vmin), float(vmax)


def trace_normalize(record, eps=1e-12):
    scale = np.max(np.abs(record), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return record / scale


def plot_model(vp, vs, geometry, dh, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    extent = [0, vp.shape[1] * dh / 1000.0, vp.shape[0] * dh / 1000.0, 0]
    for ax, model, title, label in [
        (axes[0], vp / 1000.0, "Layered Vp", "km/s"),
        (axes[1], vs / 1000.0, "Layered Vs", "km/s"),
    ]:
        im = ax.imshow(model, cmap="viridis", aspect="auto", extent=extent)
        ax.scatter(
            geometry["source"][0] * dh / 1000.0,
            geometry["source"][1] * dh / 1000.0,
            marker="*",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.5,
            label="source",
        )
        for name, color, size in [
            ("surface", "lime", 3),
            ("horizontal_well", "black", 4),
            ("vertical_well", "dodgerblue", 4),
        ]:
            pts = geometry[name]
            ax.scatter(pts[:, 0] * dh / 1000.0, pts[:, 1] * dh / 1000.0, s=size, c=color)
        ax.set_title(title)
        ax.set_xlabel("Position (km)")
        ax.set_ylabel("Depth (km)")
        fig.colorbar(im, ax=ax, label=label)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_receiver_geometry_check(vp, geometry, dh, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.0), constrained_layout=True)
    extent = [0, vp.shape[1] * dh / 1000.0, vp.shape[0] * dh / 1000.0, 0]
    im = ax.imshow(vp / 1000.0, cmap="viridis", aspect="auto", extent=extent)
    ax.scatter(
        geometry["surface"][:, 0] * dh / 1000.0,
        geometry["surface"][:, 1] * dh / 1000.0,
        s=8,
        c="lime",
        linewidths=0,
        label="surface receivers",
        zorder=3,
    )
    ax.scatter(
        geometry["horizontal_well"][:, 0] * dh / 1000.0,
        geometry["horizontal_well"][:, 1] * dh / 1000.0,
        s=9,
        c="black",
        linewidths=0,
        label="horizontal well",
        zorder=3,
    )
    ax.scatter(
        geometry["vertical_well"][:, 0] * dh / 1000.0,
        geometry["vertical_well"][:, 1] * dh / 1000.0,
        s=9,
        c="dodgerblue",
        linewidths=0,
        label="vertical well",
        zorder=3,
    )
    ax.scatter(
        geometry["source"][0] * dh / 1000.0,
        geometry["source"][1] * dh / 1000.0,
        marker="*",
        s=110,
        c="red",
        edgecolors="black",
        linewidths=0.5,
        label="source",
        zorder=4,
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Position, km")
    ax.set_ylabel("Depth, km")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(2.0, 0.0)
    fig.colorbar(im, ax=ax, label="Vp, km/s")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_common_shots(records, geometry, channels, dh, dt, duration, out_path, *, trace_norm=False):
    rows = [
        ("surface", "surface receivers", "das54x"),
        ("horizontal_well", "horizontal well", "das54x"),
        ("vertical_well", "vertical well", "das54z"),
    ]
    cols = [
        ("pressure", "pressure-like"),
        ("exx", "x strain-rate"),
        ("ezz", "z strain-rate"),
        ("das35", "helical 35.3 deg"),
        ("helical54", "helical 54.7 deg"),
    ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(15.0, 8.6), constrained_layout=True)
    for row, (geom_name, geom_title, helical54_channel) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for col, (key, title) in enumerate(cols):
            if key == "pressure":
                record = -0.5 * (records[sl, :, channels["sxx"]] + records[sl, :, channels["szz"]])
            elif key == "helical54":
                record = records[sl, :, channels[helical54_channel]]
            else:
                record = records[sl, :, channels[key]]
            display = trace_normalize(record) if trace_norm else record
            ax = axes[row, col]
            vmin, vmax = clip_limits(display)
            ax.imshow(
                display,
                cmap="seismic",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, record.shape[0] * dh / 1000.0, 0],
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(geom_title)
            if row == len(rows) - 1:
                ax.set_xlabel("Time (s)")
    suffix = "trace-normalized" if trace_norm else "true amplitude"
    fig.suptitle(f"Layered model common-shot records with paper-scale geometry ({suffix})", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_gauge(records, geometry, channels, dh, dt, duration, out_path):
    sl = geometry["slices"]["vertical_well"]
    original = records[sl, :, channels["ezz"]]
    gauge_specs = [
        ("origin", original),
        ("10 m", gauge_average(original, gauge_length=10.0, spacing=dh, axis=0)),
        ("20 m", gauge_average(original, gauge_length=20.0, spacing=dh, axis=0)),
        ("40 m", gauge_average(original, gauge_length=40.0, spacing=dh, axis=0)),
    ]

    fig, axes = plt.subplots(1, len(gauge_specs), figsize=(12.5, 3.4), constrained_layout=True)
    for ax, (title, record) in zip(axes, gauge_specs):
        vmin, vmax = clip_limits(record)
        ax.imshow(
            record,
            cmap="seismic",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[0, duration, record.shape[0] * dh / 1000.0, 0],
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Vertical well depth (km)")
    fig.suptitle("Approximate gauge-length smoothing on vertical-well z strain-rate", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def elastic_derivative_stencils(spatial_order):
    """Return the effective torch receiver stencils used by Elastic stress updates."""

    equation = Elastic(spatial_order=spatial_order, device="cpu", backend="torch")

    def offsets_and_weights(kernel):
        kernel = kernel.astype(np.float32)
        center = kernel.size // 2
        keep = np.flatnonzero(np.abs(kernel) > 0)
        offsets = keep.astype(np.int32) - center
        weights = kernel[keep].astype(np.float32)
        return offsets, weights

    kxb = equation.pd.kxb.detach().cpu().numpy()[0, 0, 0, :]
    kzb = equation.pd.kzb.detach().cpu().numpy()[0, 0, :, 0]
    return offsets_and_weights(kxb), offsets_and_weights(kzb)


def augmented_receivers_for_strain(points, x_stencil, z_stencil):
    coords = []
    index = {}

    def add(x, z):
        x = int(x)
        z = int(z)
        key = (x, z)
        if key not in index:
            index[key] = len(coords)
            coords.append(key)
        return index[key]

    x_offsets, x_weights = x_stencil
    z_offsets, z_weights = z_stencil
    maps = {"center": [], "x_indices": [], "z_indices": []}
    for x, z in points:
        center = add(x, z)
        maps["center"].append(center)
        maps["x_indices"].append([add(x + offset, z) for offset in x_offsets])
        maps["z_indices"].append([add(x, z + offset) for offset in z_offsets])

    maps["x_weights"] = np.broadcast_to(x_weights, (len(points), x_weights.size)).copy()
    maps["z_weights"] = np.broadcast_to(z_weights, (len(points), z_weights.size)).copy()
    maps["x_offsets"] = x_offsets
    maps["z_offsets"] = z_offsets

    return np.asarray(coords, dtype=np.int32), {key: np.asarray(value) for key, value in maps.items()}


def elastic_reference_records(vp_np, vs_np, rho_np, geometry, args):
    nz, nx = vp_np.shape
    target_points = geometry["receivers"][0]
    x_stencil, z_stencil = elastic_derivative_stencils(args.spatial_order)
    augmented, maps = augmented_receivers_for_strain(target_points, x_stencil, z_stencil)
    nt = int(round(args.duration / args.dt))
    wavelet = ricker(nt, args.dt, args.peak_frequency, args.delay).reshape(1, 1, nt)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    solver = PropTorch(
        ElasticStressReceiver(spatial_order=args.spatial_order, device=device, backend="torch"),
        shape=(nz, nx),
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz", "sxx", "szz"],
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )
    vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)
    vs = torch.as_tensor(vs_np, dtype=torch.float32, device=device)
    rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        record = solver(
            wavelet,
            sources=geometry["source"][None, None, :],
            receivers=augmented[None, :, :],
            models=[vp, vs, rho],
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    arr = record[0].detach().cpu().numpy()
    vx = arr[:, :, 0]
    vz = arr[:, :, 1]
    sxx = arr[:, :, 2]
    szz = arr[:, :, 3]
    center = maps["center"]
    exx = np.einsum("tnk,nk->tn", vx[:, maps["x_indices"]], maps["x_weights"]) / args.dh
    ezz = np.einsum("tnk,nk->tn", vz[:, maps["z_indices"]], maps["z_weights"]) / args.dh
    vx_center = vx[:, center]
    vz_center = vz[:, center]
    sxx_center = sxx[:, center]
    szz_center = szz[:, center]
    das35 = exx + ezz
    das54x = 4 * exx + ezz
    das54z = exx + 4 * ezz
    records = np.stack(
        [vx_center.T, vz_center.T, sxx_center.T, szz_center.T, exx.T, ezz.T, das35.T, das54x.T, das54z.T],
        axis=-1,
    )
    conversion_info = {
        "x_offsets": maps["x_offsets"].astype(int).tolist(),
        "x_weights": x_stencil[1].astype(float).tolist(),
        "z_offsets": maps["z_offsets"].astype(int).tolist(),
        "z_weights": z_stencil[1].astype(float).tolist(),
    }
    return records, ["vx", "vz", "sxx", "szz", "exx", "ezz", "das35", "das54x", "das54z"], elapsed, conversion_info


def plot_elastic_reference(records, geometry, channels, dh, duration, out_path, *, trace_norm=False):
    rows = [
        ("surface", "surface receivers", "das54x"),
        ("horizontal_well", "horizontal well", "das54x"),
        ("vertical_well", "vertical well", "das54z"),
    ]
    cols = [
        ("vx", "x particle velocity"),
        ("vz", "z particle velocity"),
        ("exx", "x strain-rate"),
        ("ezz", "z strain-rate"),
        ("das35", "helical 35.3 deg"),
        ("helical54", "helical 54.7 deg"),
    ]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(17.5, 8.6), constrained_layout=True)
    for row, (geom_name, geom_title, helical54_channel) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for col, (key, title) in enumerate(cols):
            channel = helical54_channel if key == "helical54" else key
            record = records[sl, :, channels[channel]]
            display = trace_normalize(record) if trace_norm else record
            vmin, vmax = clip_limits(display)
            ax = axes[row, col]
            ax.imshow(
                display,
                cmap="seismic",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, record.shape[0] * dh / 1000.0, 0],
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(geom_title)
            if row == len(rows) - 1:
                ax.set_xlabel("Time (s)")
    suffix = "trace-normalized" if trace_norm else "true amplitude"
    fig.suptitle(f"Elastic reference converted to DAS strain-rate ({suffix})", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_elastic_reference_focus(records, geometry, channels, dh, duration, out_path):
    rows = [
        ("surface", "surface receivers", "das54x"),
        ("vertical_well", "vertical well", "das54z"),
    ]
    cols = [
        ("vz", "z particle velocity"),
        ("exx", "x strain-rate"),
        ("ezz", "z strain-rate"),
        ("helical54", "helical 54.7 deg"),
    ]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(13.8, 6.4), constrained_layout=True)
    for row, (geom_name, geom_title, helical54_channel) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for col, (key, title) in enumerate(cols):
            channel = helical54_channel if key == "helical54" else key
            record = records[sl, :, channels[channel]]
            vmin, vmax = clip_limits(record)
            ax = axes[row, col]
            ax.imshow(
                record,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, record.shape[0] * dh / 1000.0, 0],
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(geom_title)
            if row == len(rows) - 1:
                ax.set_xlabel("Time (s)")
    fig.suptitle("Focused paper-parameter shot records from elastic reference conversion", fontsize=14)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_figure4_reproduction(direct_records, direct_channels, reference_records, reference_channels, geometry, dh, duration, out_path):
    rows = [
        ("surface", "surface receivers"),
        ("horizontal_well", "horizontal well"),
        ("vertical_well", "vertical well"),
    ]
    cols = [
        ("vx", "x-component particle velocity", reference_records, reference_channels),
        ("vz", "z-component particle velocity", reference_records, reference_channels),
        ("exx", "x-component strain-rate", direct_records, direct_channels),
        ("ezz", "z-component strain-rate", direct_records, direct_channels),
    ]
    panel_labels = list("abcdefghijkl")
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(12.8, 8.6), constrained_layout=True)
    label_index = 0
    for row, (geom_name, geom_title) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for col, (field, title, records, channels) in enumerate(cols):
            record = records[sl, :, channels[field]]
            vmin, vmax = clip_limits(record)
            ax = axes[row, col]
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
            ax.text(
                0.02,
                0.94,
                f"({panel_labels[label_index]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
            label_index += 1
            if row == 0:
                ax.set_title(title)
            ax.set_xlabel("Trace")
            if col == 0:
                ax.set_ylabel(f"{geom_title}\nTime, s")
            else:
                ax.set_ylabel("Time, s")
    fig.suptitle("Figure 4 reproduction: common-shot gathers for the Fig. 3 layered model", fontsize=14)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def conversion_metrics(direct_records, direct_channels, reference_records, reference_channels):
    metrics = {}
    for name in ["sxx", "szz", "exx", "ezz", "das35", "das54x", "das54z"]:
        direct = direct_records[..., direct_channels[name]].ravel()
        reference = reference_records[..., reference_channels[name]].ravel()
        mask = np.isfinite(direct) & np.isfinite(reference)
        if not np.any(mask):
            continue
        direct = direct[mask]
        reference = reference[mask]
        if np.max(np.abs(direct)) == 0 or np.max(np.abs(reference)) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(direct, reference)[0, 1]
        metrics[name] = {
            "corr": float(corr),
            "direct_max_abs": float(np.max(np.abs(direct))),
            "reference_max_abs": float(np.max(np.abs(reference))),
            "max_abs_diff": float(np.max(np.abs(direct - reference))),
        }
    return metrics


def conversion_metrics_by_geometry(direct_records, direct_channels, reference_records, reference_channels, geometry):
    by_geometry = {}
    for geom_name in ["surface", "horizontal_well", "vertical_well"]:
        sl = geometry["slices"][geom_name]
        by_geometry[geom_name] = conversion_metrics(
            direct_records[sl],
            direct_channels,
            reference_records[sl],
            reference_channels,
        )
    return by_geometry


def plot_stress_consistency(
    direct_records,
    direct_channels,
    reference_records,
    reference_channels,
    geometry,
    dh,
    duration,
    out_path,
):
    rows = [
        ("surface", "surface receivers"),
        ("horizontal_well", "horizontal well"),
        ("vertical_well", "vertical well"),
    ]
    fields = [("sxx", "sxx"), ("szz", "szz")]
    fig, axes = plt.subplots(len(rows), 6, figsize=(17.0, 8.7), constrained_layout=True)
    for row, (geom_name, geom_title) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for field_index, (field, field_title) in enumerate(fields):
            direct = direct_records[sl, :, direct_channels[field]]
            reference = reference_records[sl, :, reference_channels[field]]
            norm = max(np.max(np.abs(reference)), 1e-20)
            panels = [
                (direct / norm, f"{field_title} DASElastic"),
                (reference / norm, f"{field_title} Elastic"),
                ((direct - reference) / norm, f"{field_title} residual"),
            ]
            for panel_index, (panel, title) in enumerate(panels):
                col = field_index * 3 + panel_index
                ax = axes[row, col]
                if panel_index < 2:
                    vmin, vmax = -1.0, 1.0
                else:
                    vmin, vmax = -0.05, 0.05
                ax.imshow(
                    panel,
                    cmap="seismic",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[0, duration, direct.shape[0] * dh / 1000.0, 0],
                )
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(geom_title)
                if row == len(rows) - 1:
                    ax.set_xlabel("Time (s)")
    fig.suptitle("Stress-field consistency for intermediate variables", fontsize=14)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_conversion_consistency(
    direct_records,
    direct_channels,
    reference_records,
    reference_channels,
    geometry,
    dh,
    duration,
    out_path,
):
    rows = [
        ("surface", "surface helical 54.7 deg", "das54x"),
        ("vertical_well", "vertical-well helical 54.7 deg", "das54z"),
    ]
    cols = ["DASElastic direct", "Elastic converted", "residual / max(reference)"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(10.8, 5.9), constrained_layout=True)
    for row, (geom_name, geom_title, channel) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        direct = direct_records[sl, :, direct_channels[channel]]
        reference = reference_records[sl, :, reference_channels[channel]]
        norm = max(np.max(np.abs(reference)), 1e-20)
        panels = [
            direct / norm,
            reference / norm,
            (direct - reference) / norm,
        ]
        for col, (label, panel) in enumerate(zip(cols, panels)):
            ax = axes[row, col]
            if col < 2:
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = -0.01, 0.01
            ax.imshow(
                panel,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, direct.shape[0] * dh / 1000.0, 0],
            )
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(geom_title)
            if row == len(rows) - 1:
                ax.set_xlabel("Time (s)")
    fig.suptitle("DAS equation and Elastic-to-DAS conversion consistency", fontsize=13)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def normalize_record(record):
    # PropTorch returns (batch, nt, nreceiver, nchannel). Plot as
    # (nreceiver, nt, nchannel), matching the paper's shot-gather layout.
    return record[0].detach().cpu().numpy().transpose(1, 0, 2)


def run_layered(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "layered_common_shot_records_trace_normalized.png",
        "layered_elastic_reference_converted_trace_normalized.png",
    ]:
        (output_dir / stale_name).unlink(missing_ok=True)

    nz, nx = args.nz, args.nx
    vp_np, vs_np, rho_np = layered_model(nz=nz, nx=nx, dh=args.dh)
    geometry = build_layered_geometry(
        nz,
        nx,
        args.dh,
        source_x_km=args.source_x_km,
        source_depth_km=args.source_depth_km,
        surface_depth_km=args.surface_depth_km,
        horizontal_depth_km=args.horizontal_depth_km,
        horizontal_x_min_km=args.horizontal_x_min_km,
        horizontal_x_max_km=args.horizontal_x_max_km,
        vertical_x_km=args.vertical_x_km,
    )
    nt = int(round(args.duration / args.dt))
    wavelet = ricker(nt, args.dt, args.peak_frequency, args.delay).reshape(1, 1, nt)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    equation = DASElastic(spatial_order=args.spatial_order, device=device, backend="torch")
    solver = PropTorch(
        equation,
        shape=(nz, nx),
        source_type=["sxx", "szz"],
        receiver_type=["sxx", "szz", "exx", "ezz", "das35", "das54x", "das54z"],
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)
    vs = torch.as_tensor(vs_np, dtype=torch.float32, device=device)
    rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        record = solver(
            wavelet,
            sources=geometry["source"][None, None, :],
            receivers=geometry["receivers"],
            models=[vp, vs, rho],
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    records = normalize_record(record)
    channels = {name: i for i, name in enumerate(solver.receiver_type)}

    np.savez_compressed(
        output_dir / "layered_records.npz",
        records=records,
        vp=vp_np,
        vs=vs_np,
        rho=rho_np,
        wavelet=wavelet.squeeze(),
        receivers=geometry["receivers"],
        source=np.asarray(geometry["source"], dtype=np.int32),
        receiver_type=np.asarray(solver.receiver_type),
    )

    plot_model(vp_np, vs_np, geometry, args.dh, output_dir / "layered_model_geometry.png")
    plot_receiver_geometry_check(vp_np, geometry, args.dh, output_dir / "layered_receiver_geometry_check.png")
    plot_common_shots(records, geometry, channels, args.dh, args.dt, args.duration, output_dir / "layered_common_shot_records.png")
    plot_gauge(records, geometry, channels, args.dh, args.dt, args.duration, output_dir / "layered_gauge_smoothing_vertical.png")

    reference_records, reference_channel_names, reference_elapsed, reference_conversion = elastic_reference_records(vp_np, vs_np, rho_np, geometry, args)
    reference_channels = {name: i for i, name in enumerate(reference_channel_names)}
    np.savez_compressed(
        output_dir / "layered_elastic_reference_records.npz",
        records=reference_records,
        receiver_type=np.asarray(reference_channel_names),
        conversion_x_offsets=np.asarray(reference_conversion["x_offsets"], dtype=np.int32),
        conversion_x_weights=np.asarray(reference_conversion["x_weights"], dtype=np.float32),
        conversion_z_offsets=np.asarray(reference_conversion["z_offsets"], dtype=np.int32),
        conversion_z_weights=np.asarray(reference_conversion["z_weights"], dtype=np.float32),
    )
    plot_elastic_reference(
        reference_records,
        geometry,
        reference_channels,
        args.dh,
        args.duration,
        output_dir / "layered_elastic_reference_converted.png",
    )
    plot_elastic_reference_focus(
        reference_records,
        geometry,
        reference_channels,
        args.dh,
        args.duration,
        output_dir / "layered_elastic_reference_focus.png",
    )
    plot_figure4_reproduction(
        records,
        channels,
        reference_records,
        reference_channels,
        geometry,
        args.dh,
        args.duration,
        output_dir / "layered_figure4_reproduction.png",
    )
    metrics = conversion_metrics(records, channels, reference_records, reference_channels)
    plot_conversion_consistency(
        records,
        channels,
        reference_records,
        reference_channels,
        geometry,
        args.dh,
        args.duration,
        output_dir / "layered_conversion_consistency.png",
    )
    plot_stress_consistency(
        records,
        channels,
        reference_records,
        reference_channels,
        geometry,
        args.dh,
        args.duration,
        output_dir / "layered_stress_consistency.png",
    )

    metadata = {
        "paper": "Zhao et al., Petroleum Science 23 (2026) 626-642",
        "case": "layered",
        "paper_parameters_used": {
            "grid": [201, 401],
            "dh_m": 10.0,
            "ricker_peak_frequency_hz": 10.0,
            "surface_traces": 401,
            "horizontal_well_traces": 201,
            "vertical_well_traces": 201,
            "spatial_order": 8,
            "time_order": 2,
            "layer_interfaces_km": [0.75, 1.5],
            "vp_km_s": [1.5, 2.5, 3.0],
            "vs_km_s": [1.0, 1.4, 1.6],
            "fig3_geometry_km": FIG3_GEOMETRY_KM,
            "source_km": geometry["meta"]["source_km"],
            "surface_km": geometry["meta"]["surface_km"],
            "horizontal_well_km": geometry["meta"]["horizontal_well_km"],
            "vertical_well_km": geometry["meta"]["vertical_well_km"],
        },
        "assumptions": [
            "The paper gives source/receiver geometry graphically but not as a coordinate table; this script digitizes the Fig. 3 source and receiver lines to the nearest 10 m grid cell.",
            "The Fig. 3 layered model is represented as three constant layers with interfaces at 0.75 km and 1.5 km; velocities are read from the Fig. 3 colorbar levels.",
            "The main layered shot-gather time sampling is not stated explicitly; the default 4.0 s window is chosen to match the Fig. 4 vertical time axis.",
            "Gauge smoothing is applied on the 10 m receiver grid, so it approximates the paper's high-resolution 1 m gauge experiment rather than reproducing it exactly.",
        ],
        "run_parameters": {
            "nz": nz,
            "nx": nx,
            "dh_m": args.dh,
            "dt_s": args.dt,
            "duration_s": args.duration,
            "nt": nt,
            "peak_frequency_hz": args.peak_frequency,
            "delay_s": args.delay,
            "spatial_order": args.spatial_order,
            "abcn": args.abcn,
            "device": str(device),
            "elapsed_s": elapsed,
            "elastic_reference_elapsed_s": reference_elapsed,
        },
        "elastic_reference_conversion": {
            "description": "vx/vz are converted to exx/ezz with the same SWEEP staggered-grid x_backward/z_backward derivative kernels used inside Elastic.step stress updates.",
            **reference_conversion,
        },
        "direct_vs_elastic_reference": metrics,
        "direct_vs_elastic_reference_by_geometry": conversion_metrics_by_geometry(
            records,
            channels,
            reference_records,
            reference_channels,
            geometry,
        ),
        "outputs": [
            "layered_model_geometry.png",
            "layered_receiver_geometry_check.png",
            "layered_common_shot_records.png",
            "layered_gauge_smoothing_vertical.png",
            "layered_figure4_reproduction.png",
            "layered_conversion_consistency.png",
            "layered_stress_consistency.png",
            "layered_records.npz",
            "layered_elastic_reference_converted.png",
            "layered_elastic_reference_focus.png",
            "layered_elastic_reference_records.npz",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["run_parameters"], indent=2))
    print(f"wrote {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Reproduce DAS paper-style common-shot records with SWEEP Python equations.")
    parser.add_argument("--output-dir", type=Path, default=Path("test/test_outputs/das_paper_reproduction/layered"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nz", type=int, default=201)
    parser.add_argument("--nx", type=int, default=401)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--peak-frequency", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--spatial-order", type=int, default=8)
    parser.add_argument("--abcn", type=int, default=50)
    parser.add_argument("--source-x-km", type=float, default=2.0)
    parser.add_argument("--source-depth-km", type=float, default=0.0)
    parser.add_argument("--surface-depth-km", type=float, default=0.0)
    parser.add_argument("--horizontal-depth-km", type=float, default=1.2)
    parser.add_argument("--horizontal-x-min-km", type=float, default=0.5)
    parser.add_argument("--horizontal-x-max-km", type=float, default=2.5)
    parser.add_argument("--vertical-x-km", type=float, default=3.0)
    return parser


def main():
    args = build_parser().parse_args()
    run_layered(args)


if __name__ == "__main__":
    main()
