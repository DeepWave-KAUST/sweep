"""Cache directory resolution and download helpers.

``requests`` and ``tqdm`` are imported lazily inside :func:`download` so that
importing ``sweep.datasets`` (and loading the embedded demo models) never
requires them. Install them with ``pip install sweep-solver[datasets]``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream-download ``url`` to ``dest`` with a progress bar.

    If ``dest`` exists and its sha256 matches (or no sha256 is given), the
    download is skipped. ``requests``/``tqdm`` are imported lazily here.
    """
    dest = Path(dest)
    if dest.exists() and sha256 is not None:
        if _sha256(dest) == sha256:
            return dest
    elif dest.exists() and sha256 is None:
        return dest

    try:
        import requests
        from tqdm.auto import tqdm
    except ImportError as e:  # pragma: no cover - trivial guard
        raise ImportError(
            "Downloading benchmark models needs `requests` and `tqdm`. "
            "Install with `pip install sweep-solver[datasets]`."
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
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
    if sha256 is not None:
        got = _sha256(tmp)
        if got != sha256:
            tmp.unlink(missing_ok=True)
            raise IOError(f"{dest.name}: sha256 mismatch (got {got}, want {sha256})")
    os.replace(tmp, dest)
    return dest
