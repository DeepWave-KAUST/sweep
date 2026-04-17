import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


def parse_int_list(text):
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_compile_options(text):
    options = []
    for part in text.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value in {"off", "false", "0", "no"}:
            options.append(False)
        elif value in {"on", "true", "1", "yes"}:
            options.append(True)
        else:
            raise ValueError(
                f"Unsupported compile option '{part}'. Use a comma-separated list of on/off values."
            )
    deduped = []
    for value in options:
        if value not in deduped:
            deduped.append(value)
    return deduped


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PropTorch no-ckpt/ckpt and compile modes for memory, speed, "
            "loss consistency, and gradient correctness."
        )
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--fm", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--spatial-order", type=int, default=4, dest="spatial_order")
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--nshots", type=int, default=6)
    parser.add_argument("--nreceivers", type=int, default=96)
    parser.add_argument("--src-z", type=int, default=4, dest="src_z")
    parser.add_argument("--rec-z", type=int, default=4, dest="rec_z")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-sizes", default="1,64,128,256")
    parser.add_argument("--include-no-ckpt", action="store_true")
    parser.add_argument("--compile-options", default="off,on")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save gradient and simulated-data comparison figures.",
    )
    return parser


def build_case(args):
    vp = np.full((args.nz, args.nx), 1800.0, dtype=np.float32)
    vp[args.nz // 2 :, :] = 2400.0
    vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0

    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, f=args.fm).astype(np.float32)

    src_x = np.linspace(8, args.nx - 9, args.nshots, dtype=np.int64)
    src_z = np.full(args.nshots, args.src_z, dtype=np.int64)
    sources = np.stack([src_x, src_z], axis=1)

    rec_x = np.linspace(0, args.nx - 1, args.nreceivers, dtype=np.int64)
    rec_z = np.full(args.nreceivers, args.rec_z, dtype=np.int64)
    receiver_template = np.stack([rec_x, rec_z], axis=1)
    receivers = np.repeat(receiver_template[None, :, :], args.nshots, axis=0)

    return vp, wave, sources, receivers


def build_solver(args, device, use_ckpt, ckpt_chunks):
    return PropTorch(
        Acoustic(spatial_order=args.spatial_order, device=device, backend="torch"),
        shape=(args.nz, args.nx),
        dev=device,
        dh=args.dh,
        dt=args.dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        free_surface=False,
        use_ckpt=use_ckpt,
        ckpt_mode="chunk",
        ckpt_chunks=ckpt_chunks,
        pml_type="cpmlr",
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
    )


def reset_memory_stats(device):
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(args, case_name, use_ckpt, ckpt_chunks, vp_np, wave, sources, receivers, device):
    solver = build_solver(args, device, use_ckpt=use_ckpt, ckpt_chunks=ckpt_chunks)
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32, requires_grad=True)

    def step():
        if vp.grad is not None:
            vp.grad = None
        out = solver(wave, sources, receivers, models=[vp])
        loss = out.square().mean()
        loss.backward()
        return out, loss

    for _ in range(args.warmup):
        reset_memory_stats(device)
        step()
        sync_device(device)

    allocated_peaks = []
    reserved_peaks = []
    timings = []
    out = None
    loss = None

    for _ in range(args.repeats):
        reset_memory_stats(device)
        sync_device(device)
        start = time.perf_counter()
        out, loss = step()
        sync_device(device)
        timings.append(time.perf_counter() - start)
        if device.type == "cuda":
            allocated_peaks.append(torch.cuda.max_memory_allocated(device))
            reserved_peaks.append(torch.cuda.max_memory_reserved(device))

    grad = vp.grad.detach().cpu().to(torch.float64)
    record = out.detach().cpu().to(torch.float64)
    finite = torch.isfinite(grad)

    if device.type == "cuda":
        peak_allocated_mean_mib = float(np.mean(allocated_peaks) / (1024.0 ** 2))
        peak_allocated_max_mib = float(np.max(allocated_peaks) / (1024.0 ** 2))
        peak_reserved_mean_mib = float(np.mean(reserved_peaks) / (1024.0 ** 2))
        peak_reserved_max_mib = float(np.max(reserved_peaks) / (1024.0 ** 2))
    else:
        peak_allocated_mean_mib = float("nan")
        peak_allocated_max_mib = float("nan")
        peak_reserved_mean_mib = float("nan")
        peak_reserved_max_mib = float("nan")

    return {
        "name": case_name,
        "compile": bool(args.use_compile),
        "use_ckpt": use_ckpt,
        "ckpt_chunks": ckpt_chunks,
        "time_mean_ms": float(np.mean(timings) * 1e3),
        "time_std_ms": float(np.std(timings, ddof=0) * 1e3),
        "peak_allocated_mean_mib": peak_allocated_mean_mib,
        "peak_allocated_max_mib": peak_allocated_max_mib,
        "peak_reserved_mean_mib": peak_reserved_mean_mib,
        "peak_reserved_max_mib": peak_reserved_max_mib,
        "loss": float(loss.detach().cpu().item()),
        "grad": grad.numpy().copy(),
        "record": record.numpy().copy(),
        "grad_finite_count": int(finite.sum().item()),
        "grad_total_count": int(grad.numel()),
        "grad_max_abs": float(grad.abs().max().item()),
    }


def compare_arrays(reference, candidate):
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cur = np.asarray(candidate, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(ref) & np.isfinite(cur)
    if not np.any(finite_mask):
        return {
            "finite_overlap": 0,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "l2_rel_error": float("nan"),
            "cosine_similarity": float("nan"),
        }

    ref_f = ref[finite_mask]
    cur_f = cur[finite_mask]
    diff = cur_f - ref_f
    ref_norm = float(np.linalg.norm(ref_f))
    cur_norm = float(np.linalg.norm(cur_f))
    diff_norm = float(np.linalg.norm(diff))
    dot = float(np.dot(ref_f, cur_f))
    denom = ref_norm * cur_norm

    return {
        "finite_overlap": int(finite_mask.sum()),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "l2_rel_error": float(diff_norm / ref_norm) if ref_norm != 0.0 else float("nan"),
        "cosine_similarity": float(dot / denom) if denom != 0.0 else float("nan"),
    }


def sanitize_case_name(row):
    mode = "compile_on" if row["compile"] else "compile_off"
    return f"{row['name']}_{mode}"


def finite_percentile(image, q):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def symmetric_limit(image, percentile=99.0):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit == 0.0:
        limit = float(np.max(np.abs(finite)))
    return limit if limit > 0.0 else 1.0


def plot_gradient_case(row, baseline_grad, output_dir):
    case_key = sanitize_case_name(row)
    grad = np.asarray(row["grad"], dtype=np.float64)
    diff = grad - baseline_grad

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    base_ax, grad_ax, diff_ax = axes[0]

    base_vmin = finite_percentile(baseline_grad, 1.0)
    base_vmax = finite_percentile(baseline_grad, 99.0)
    if base_vmin is None or base_vmax is None or base_vmin == base_vmax:
        base_limit = symmetric_limit(baseline_grad)
        base_vmin, base_vmax = -base_limit, base_limit

    grad_vmin = finite_percentile(grad, 1.0)
    grad_vmax = finite_percentile(grad, 99.0)
    if grad_vmin is None or grad_vmax is None or grad_vmin == grad_vmax:
        grad_limit = symmetric_limit(grad)
        grad_vmin, grad_vmax = -grad_limit, grad_limit

    diff_limit = symmetric_limit(diff)

    base_im = base_ax.imshow(
        baseline_grad,
        cmap="seismic",
        vmin=base_vmin,
        vmax=base_vmax,
        aspect="auto",
    )
    base_ax.set_title("baseline: no_ckpt + compile_off")
    fig.colorbar(base_im, ax=base_ax)

    grad_im = grad_ax.imshow(grad, cmap="seismic", vmin=grad_vmin, vmax=grad_vmax, aspect="auto")
    grad_ax.set_title(
        f"{case_key}\nloss={row['loss']:.6e}, relL2={row['grad_l2_rel_error']:.3e}"
    )
    fig.colorbar(grad_im, ax=grad_ax)

    diff_im = diff_ax.imshow(diff, cmap="seismic", vmin=-diff_limit, vmax=diff_limit, aspect="auto")
    diff_ax.set_title(
        f"diff vs baseline\nmaxabs={row['grad_max_abs_diff']:.3e}, cos={row['grad_cosine_similarity']:.8f}"
    )
    fig.colorbar(diff_im, ax=diff_ax)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_key}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def extract_record_image(record):
    data = np.asarray(record, dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"Expected record with 4 dims (batch, nt, nrec, channels), got {data.shape}")
    return data[0, :, :, 0]


def plot_record_case(row, baseline_record, output_dir):
    case_key = sanitize_case_name(row)
    record = np.asarray(row["record"], dtype=np.float64)
    diff = record - baseline_record

    base_img = extract_record_image(baseline_record)
    rec_img = extract_record_image(record)
    diff_img = extract_record_image(diff)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    base_ax, rec_ax, diff_ax = axes[0]

    base_vmin = finite_percentile(base_img, 1.0)
    base_vmax = finite_percentile(base_img, 99.0)
    if base_vmin is None or base_vmax is None or base_vmin == base_vmax:
        base_limit = symmetric_limit(base_img)
        base_vmin, base_vmax = -base_limit, base_limit

    rec_vmin = finite_percentile(rec_img, 1.0)
    rec_vmax = finite_percentile(rec_img, 99.0)
    if rec_vmin is None or rec_vmax is None or rec_vmin == rec_vmax:
        rec_limit = symmetric_limit(rec_img)
        rec_vmin, rec_vmax = -rec_limit, rec_limit

    diff_limit = symmetric_limit(diff_img)

    base_im = base_ax.imshow(base_img, cmap="seismic", vmin=base_vmin, vmax=base_vmax, aspect="auto")
    base_ax.set_title("baseline data: no_ckpt + compile_off")
    base_ax.set_xlabel("receiver")
    base_ax.set_ylabel("time")
    fig.colorbar(base_im, ax=base_ax)

    rec_im = rec_ax.imshow(rec_img, cmap="seismic", vmin=rec_vmin, vmax=rec_vmax, aspect="auto")
    rec_ax.set_title(f"{case_key} simulated data")
    rec_ax.set_xlabel("receiver")
    rec_ax.set_ylabel("time")
    fig.colorbar(rec_im, ax=rec_ax)

    diff_im = diff_ax.imshow(diff_img, cmap="seismic", vmin=-diff_limit, vmax=diff_limit, aspect="auto")
    diff_ax.set_title("simulated - baseline")
    diff_ax.set_xlabel("receiver")
    diff_ax.set_ylabel("time")
    fig.colorbar(diff_im, ax=diff_ax)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_key}_record.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_artifacts(results, baseline, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    baseline_grad = np.asarray(baseline["grad"], dtype=np.float64)
    baseline_record = np.asarray(baseline["record"], dtype=np.float64)
    for row in results:
        plot_gradient_case(row, baseline_grad, output_dir)
        plot_record_case(row, baseline_record, output_dir)


def print_results_table(results, device):
    baseline = next(
        row
        for row in results
        if row["name"] == "no_ckpt" and row["compile"] is False
    )

    headers = [
        "Case",
        "Compile",
        "Time ms",
        "Speedup",
        "Alloc MiB",
        "Alloc x",
        "Loss RelErr",
        "Grad RelL2",
        "Grad Cos",
    ]

    rows = []
    for row in results:
        speedup = baseline["time_mean_ms"] / row["time_mean_ms"]
        if device.type == "cuda":
            alloc_ratio = row["peak_allocated_max_mib"] / baseline["peak_allocated_max_mib"]
            alloc_mib = f"{row['peak_allocated_max_mib']:.2f}"
            alloc_ratio_text = f"{alloc_ratio:.3f}x"
        else:
            alloc_mib = "-"
            alloc_ratio_text = "-"

        rows.append(
            [
                row["name"],
                "on" if row["compile"] else "off",
                f"{row['time_mean_ms']:.2f}",
                f"{speedup:.3f}x",
                alloc_mib,
                alloc_ratio_text,
                f"{row['loss_rel_error']:.3e}",
                f"{row['grad_l2_rel_error']:.3e}",
                f"{row['grad_cosine_similarity']:.8f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt_row(values):
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print("Baseline: no_ckpt + compile=off")
    print(fmt_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt_row(row))


def collect_cases(args, compile_options):
    cases = [("no_ckpt", False, 1)]
    for use_compile in compile_options:
        if args.include_no_ckpt and use_compile:
            pass
    # Baseline is always included, then optional no_ckpt for compile=on and ckpt cases.
    scheduled = [(False, "no_ckpt", False, 1)]
    for use_compile in compile_options:
        if use_compile:
            if args.include_no_ckpt:
                scheduled.append((True, "no_ckpt", False, 1))
        elif args.include_no_ckpt:
            pass
        for chunk in parse_int_list(args.chunk_sizes):
            scheduled.append((use_compile, f"ckpt_chunk_{chunk}", True, chunk))
    return scheduled


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    vp_np, wave, sources, receivers = build_case(args)
    compile_options = parse_compile_options(args.compile_options)
    original_config = vars(args).copy()

    results = []
    scheduled = collect_cases(args, compile_options)

    for use_compile, case_name, use_ckpt, ckpt_chunks in scheduled:
        args.use_compile = use_compile
        results.append(
            run_case(
                args,
                case_name=case_name,
                use_ckpt=use_ckpt,
                ckpt_chunks=ckpt_chunks,
                vp_np=vp_np,
                wave=wave,
                sources=sources,
                receivers=receivers,
                device=device,
            )
        )

    baseline = next(
        row
        for row in results
        if row["name"] == "no_ckpt" and row["compile"] is False
    )
    baseline_loss = baseline["loss"]
    baseline_grad = baseline["grad"]
    baseline_record = baseline["record"]

    for row in results:
        grad_metrics = compare_arrays(baseline_grad, row["grad"])
        record_metrics = compare_arrays(baseline_record, row["record"])
        row["loss_abs_diff"] = float(abs(row["loss"] - baseline_loss))
        row["loss_rel_error"] = (
            float(abs(row["loss"] - baseline_loss) / abs(baseline_loss))
            if baseline_loss != 0.0
            else float("nan")
        )
        row["grad_max_abs_diff"] = grad_metrics["max_abs_diff"]
        row["grad_mean_abs_diff"] = grad_metrics["mean_abs_diff"]
        row["grad_l2_rel_error"] = grad_metrics["l2_rel_error"]
        row["grad_cosine_similarity"] = grad_metrics["cosine_similarity"]
        row["grad_finite_overlap"] = grad_metrics["finite_overlap"]
        row["record_max_abs_diff"] = record_metrics["max_abs_diff"]
        row["record_mean_abs_diff"] = record_metrics["mean_abs_diff"]
        row["record_l2_rel_error"] = record_metrics["l2_rel_error"]
        row["record_cosine_similarity"] = record_metrics["cosine_similarity"]
        row["record_finite_overlap"] = record_metrics["finite_overlap"]

    if args.output_dir:
        save_comparison_artifacts(results, baseline, args.output_dir)
        print(f"Saved comparison figures to: {args.output_dir}")

    print_results_table(results, device)

    for row in results:
        del row["grad"]
        del row["record"]

    payload = {
        "config": original_config,
        "device": str(device),
        "torch": torch.__version__,
        "baseline": {
            "name": baseline["name"],
            "compile": baseline["compile"],
            "loss": baseline["loss"],
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
