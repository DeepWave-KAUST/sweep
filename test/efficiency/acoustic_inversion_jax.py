import argparse
import json
import sys
from pathlib import Path

from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from _common import (
    add_benchmark_args,
    append_summary_csv,
    benchmark_params,
    make_acoustic_2d_case,
    print_params,
    run_benchmark,
    summary_stats,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 2D acoustic inversion/gradient computation for the JAX backend."
    )
    return add_benchmark_args(parser)


def main():
    import jax
    import jax.numpy as jnp

    args = build_parser().parse_args()
    case = make_acoustic_2d_case(args)

    print(f"Backend: JAX")
    print(f"Model shape: {case['shape']}, nt={args.nt}, receivers={case['receivers'].shape[1]}")
    print(f"JAX devices: {jax.devices()}")
    print_params(args, backend="jax", device="jax", shape=list(case["shape"]), nreceivers=int(case["receivers"].shape[1]))

    def build_step():
        solver = PropJax(
            Acoustic(spatial_order=args.spatial_order, backend="jax"),
            shape=case["shape"],
            dev=None,
            dh=args.dh,
            dt=args.dt,
            source_type=["h1"],
            receiver_type=["h1"],
            abcn=args.abcn,
            free_surface=False,
            pml_type="cpmlr",
            use_ckpt=False,
        )
        wave = jnp.array(case["wave"])
        sources = jnp.array(case["sources"])
        receivers = jnp.array(case["receivers"])
        vp = jnp.array(case["vp"])

        def loss_fn(model):
            record = solver(wave, sources, receivers, models=[model])
            return jnp.mean(record**2)

        grad_fn = jax.jit(jax.grad(loss_fn))

        def run():
            grad = grad_fn(vp)
            jax.block_until_ready(grad)
            return grad

        return run

    print("\nInversion benchmark")
    timings = run_benchmark("JAX", build_step, lambda: None, args.warmup, args.repeats)
    params = benchmark_params(
        args,
        backend="jax",
        device="jax",
        shape=list(case["shape"]),
        nreceivers=int(case["receivers"].shape[1]),
    )
    stats = summary_stats(timings)
    append_summary_csv(
        Path(__file__).resolve().parent / "benchmark_inversion_summary.csv",
        {
            "backend": "jax",
            "variant": "default",
            "label": "JAX",
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "mode": "inversion",
            "script": Path(__file__).name,
            "python": sys.executable,
            "script_args": " ".join(sys.argv[1:]),
            "nz": params.get("nz"),
            "nx": params.get("nx"),
            "nt": params.get("nt"),
            "dh": params.get("dh"),
            "dt": params.get("dt"),
            "delay": params.get("delay"),
            "fm": params.get("fm"),
            "spatial_order": params.get("spatial_order"),
            "abcn": params.get("abcn"),
            "warmup": params.get("warmup"),
            "repeats": params.get("repeats"),
            "receiver_stride": params.get("receiver_stride"),
            "shape": json.dumps(params.get("shape"), ensure_ascii=True),
            "nreceivers": params.get("nreceivers"),
            "device": params.get("device"),
            "parameters_json": json.dumps({"script": params}, ensure_ascii=True, sort_keys=True),
        },
    )


if __name__ == "__main__":
    main()
