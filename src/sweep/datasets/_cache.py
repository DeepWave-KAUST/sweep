"""Cache directory resolution and download helpers.

``requests`` and ``tqdm`` are imported lazily inside :func:`download` so that
importing ``sweep.datasets`` (and loading the embedded demo models) never
requires them. Install them with ``pip install sweep-solver[datasets]``.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Sequence


def cache_root() -> Path:
    """Resolve the cache directory.

    Honors ``SWEEP_DATASETS_CACHE`` (preferred) or the legacy
    ``SWEEP_ZOO_CACHE``, then falls back to ``$XDG_CACHE_HOME/sweep-datasets``
    (or ``~/.cache/sweep-datasets``).
    """
    env = os.environ.get("SWEEP_DATASETS_CACHE") or os.environ.get("SWEEP_ZOO_CACHE")
    if env:
        root = Path(env)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg or Path.home() / ".cache") / "sweep-datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_path(name: str, filename: str) -> Path:
    """Return the cached path for ``<cache>/<name>/<filename>``."""
    p = cache_root() / name / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _digests(path: Path, algorithms) -> dict[str, str]:
    """Hex digest of ``path`` for each hashlib algorithm named, in one read."""
    hs = {a: hashlib.new(a) for a in algorithms}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            for h in hs.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hs.items()}


def checksum_error(path: Path, *, sha256: str | None = None, md5: str | None = None) -> str | None:
    """``None`` if ``path`` matches every checksum given, else what did not match."""
    want = {a: v.lower() for a, v in (("sha256", sha256), ("md5", md5)) if v}
    if not want:
        return None
    got = _digests(path, want)
    bad = [f"{a} {got[a]} (expected {v})" for a, v in want.items() if got[a] != v]
    return "; ".join(bad) or None


def download(
    urls: str | Sequence[str],
    dest: Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream-download ``dest`` with a progress bar from the first of ``urls`` that serves it.

    ``urls`` is one URL or several mirrors of the same file, tried in order.
    With ``sha256``/``md5`` given, a download is kept only if it matches them
    -- an error page served with status 200, a truncated transfer or a
    different file falls through to the next URL -- and an existing ``dest``
    is re-verified on every call (and fetched again if it does not match).
    Without a checksum an existing ``dest`` is returned as is.
    ``requests``/``tqdm`` are imported lazily here.
    """
    dest = Path(dest)
    urls = [urls] if isinstance(urls, str) else list(urls)
    if dest.exists():
        bad = checksum_error(dest, sha256=sha256, md5=md5)
        if bad is None:
            return dest
        warnings.warn(f"{dest}: cached file does not match its checksum ({bad}); fetching it again")

    try:
        import requests
        from tqdm.auto import tqdm
    except ImportError as e:  # pragma: no cover - trivial guard
        raise ImportError(
            "Downloading benchmark models needs `requests` and `tqdm`. "
            "Install with `pip install sweep-solver[datasets]`."
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    # One temp file per process: every rank of a multi-GPU run may fetch the
    # same file at once, and a shared one ends up a mix of their writes.
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    failures = []
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                with open(tmp, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
            bad = checksum_error(tmp, sha256=sha256, md5=md5)
            if bad is not None:
                raise OSError(f"checksum mismatch: {bad}")
        except OSError as e:  # requests' exceptions are OSErrors too
            tmp.unlink(missing_ok=True)
            failures.append(f"  {url}\n    {type(e).__name__}: {e}")
            continue
        os.replace(tmp, dest)
        return dest
    want = ", ".join(f"{a} {v}" for a, v in (("sha256", sha256), ("md5", md5)) if v)
    raise OSError(
        f"could not download {dest.name} from any source:\n" + "\n".join(failures)
        + f"\nIf you have the original file, place it at {dest}"
        + (f" (it must match {want})" if want else "") + " and load again."
    )
