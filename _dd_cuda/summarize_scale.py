"""Summarize a dd_v100_round10-style slurm log into scale tables.

Usage: python _dd_cuda/summarize_scale.py <slurm.log>

Parses DD_BENCH lines in document order, tracking the @@@ section headers
the sbatch script prints, so runs that don't encode every parameter in
their tag (e.g. abcn before the -abNN tag existed) are still attributed
correctly by position.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"DD_BENCH ndim=(\d) tile=\(([\d, ]+)\) px=(\d+) nt=(\d+) "
    r"ex=(\S+) best=([\d.]+)s per_step=([\d.]+)ms cv=([\d.]+) "
    r"peak_mem=([\d.]+)GB"
)


def parse(path):
    section = "?"
    rows = []
    for raw in open(path, errors="replace"):
        if raw.startswith("@@@"):
            section = raw.strip().lstrip("@ ").split(":")[0]
            continue
        m = LINE.search(raw)
        if m:
            nd, tile, px, nt, tag, best, ps, cv, mem = m.groups()
            rows.append(dict(
                section=section, ndim=int(nd),
                tile=tuple(int(x) for x in tile.replace(" ", "").split(",")),
                px=int(px), nt=int(nt), tag=tag, best=float(best),
                per_step=float(ps), cv=float(cv), mem=float(mem),
            ))
    return rows


def mode_of(tag):
    if tag.startswith("none"):
        return "none"
    if "overlap" in tag:
        return "overlap"
    if tag.startswith("full"):
        return "serial"
    return "base"  # std-so* single-GPU baselines


def main(path):
    rows = parse(path)
    by_sec = defaultdict(list)
    for r in rows:
        by_sec[r["section"]].append(r)

    for sec, rs in by_sec.items():
        print(f"\n===== {sec} =====")
        # group weak-scaling style: key = (ndim, tile, nt, so/ab part of tag)
        def cfg(r):
            extra = "-".join(p for p in r["tag"].split("-")[1:])
            return (r["ndim"], r["tile"], r["nt"], extra)

        groups = defaultdict(dict)
        baselines = {}
        order = []
        for r in rs:
            m = mode_of(r["tag"])
            if m == "base" and r["px"] == 1:
                key = (r["ndim"], r["tile"], r["nt"])
                baselines.setdefault(key, []).append(r["per_step"])
                continue
            k = cfg(r) + (r["px"],)
            if k not in groups:
                order.append(k)
            groups[k][m] = r["per_step"]

        for k in order:
            ndim, tile, nt, extra, px = k
            g = groups[k]
            none = g.get("none")
            parts = [f"{ndim}D tile={tile} nt={nt} {extra} px={px}:"]
            for m in ("none", "serial", "overlap"):
                if m in g:
                    s = f"{m}={g[m]:.3f}ms"
                    if none and m != "none":
                        s += f" (weak-eff {none / g[m] * 100:.1f}%)"
                    parts.append(s)
            base_key = (ndim, tile, nt)
            if base_key in baselines and not none:
                b = min(baselines[base_key])
                for m in ("overlap", "serial"):
                    if m in g:
                        sp = b / g[m]
                        parts.append(
                            f"[vs 1-GPU {b:.3f}ms: speedup {sp:.2f}x "
                            f"eff {sp / px * 100:.1f}%]")
                        break
            print("  " + " ".join(parts))

        for key, vals in baselines.items():
            print(f"  baseline px=1 {key[0]}D tile={key[1]} nt={key[2]}: "
                  f"{', '.join(f'{v:.3f}' for v in sorted(vals))} ms/step")


if __name__ == "__main__":
    main(sys.argv[1])
