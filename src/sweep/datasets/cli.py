"""``sweep datasets`` CLI: ``list`` / ``info`` / ``download`` / ``where``.

Also installed as the standalone ``sweep-datasets`` command.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_list(args) -> int:
    from sweep.datasets import catalog

    catalog(args.name)
    return 0


def _cmd_info(args) -> int:
    from sweep.datasets import info

    e = info(args.name, args.variant)
    print(f"{e.name}:{e.variant}")
    print(f"  kind        {e.kind}")
    print(f"  description {e.description}")
    print(f"  citation    {e.citation}")
    print(f"  license     {e.license}")
    print(f"  re-host     {'yes' if e.redistributable else 'no'}")
    print(f"  default     {e.default}")
    print(f"  tags        {', '.join(e.tags)}")
    return 0


def _cmd_download(args) -> int:
    from sweep.datasets import load

    kw = {}
    if args.downsample:
        kw["downsample"] = args.downsample[0] if len(args.downsample) == 1 else tuple(args.downsample)
    out = load(args.name, args.variant, **kw)
    print(f"downloaded {out['name']}:{out['variant']}")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k:8s} {tuple(v.shape)} {v.dtype}")
    print(f"  dh={out.get('dh')} dt={out.get('dt')} nt={out.get('nt')}")
    return 0


def _cmd_where(_args) -> int:
    from sweep.datasets._cache import cache_root

    print(cache_root())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="sweep datasets",
        description="Benchmark velocity models (Marmousi, Overthrust, BP, Hess, ...).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="list registered datasets (optionally one family)")
    lp.add_argument("name", nargs="?", default=None)
    lp.set_defaults(func=_cmd_list)

    ip = sub.add_parser("info", help="show one dataset entry's metadata")
    ip.add_argument("name")
    ip.add_argument("variant", nargs="?", default=None)
    ip.set_defaults(func=_cmd_info)

    dp = sub.add_parser("download", help="fetch a dataset to the cache and report its shape")
    dp.add_argument("name")
    dp.add_argument("variant", nargs="?", default=None)
    dp.add_argument(
        "--downsample", type=int, nargs="+", default=None,
        help="uniform int, or one factor per axis (e.g. --downsample 2 4)",
    )
    dp.set_defaults(func=_cmd_download)

    wp = sub.add_parser("where", help="print the cache directory")
    wp.set_defaults(func=_cmd_where)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
