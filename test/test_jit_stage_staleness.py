"""The JIT staging dir must follow csrc, not the package version.

`_stage` mirrors `src/sweep/csrc` into the torch-extension build dir with
unique compiled-source basenames. The staleness check used to be "does
`.staged` exist", and the directory was named after the installed
`sweep-solver` version -- so editing a kernel without bumping the version left
the previous copy in place, ninja compiled the OLD source, and the resulting
`.so` silently did not contain the edit. These tests pin the manifest
behaviour that replaced it; no compiler and no GPU are involved.
"""
import json
import os
from pathlib import Path

import pytest

from sweep import _jit


def _make_csrc(root: Path) -> None:
    (root / "cuda" / "equations" / "acoustic2d").mkdir(parents=True)
    (root / "cuda" / "common").mkdir(parents=True, exist_ok=True)
    (root / "bindings").mkdir(parents=True, exist_ok=True)
    (root / "cuda" / "equations" / "acoustic2d" / "forward.cu").write_text("v1\n")
    (root / "cuda" / "common" / "shared.cuh").write_text("header v1\n")
    (root / "bindings" / "module.cpp").write_text("binding v1\n")


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """(csrc root, build dir, stage() -> staged source list)."""
    csrc = tmp_path / "csrc"
    _make_csrc(csrc)
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setattr(_jit, "_CSRC", csrc)

    def _sources():
        return [str(csrc / "cuda/equations/acoustic2d/forward.cu"),
                str(csrc / "bindings/module.cpp")]

    monkeypatch.setattr(_jit, "_sources", _sources)
    return csrc, build, lambda: _jit._stage(build)


def _stage_dir(build: Path) -> Path:
    (d,) = list(build.glob("csrc_stage_*"))
    return d


def test_edited_source_is_restaged(staged):
    csrc, build, stage = staged
    sources, _ = stage()
    forward = Path(sources[0])
    assert forward.read_text() == "v1\n"
    assert forward.name == "cuda_equations_acoustic2d_forward.cu", \
        "compiled sources are renamed so object basenames stay unique"

    (csrc / "cuda/equations/acoustic2d/forward.cu").write_text("v2\n")
    sources, _ = stage()
    assert Path(sources[0]).read_text() == "v2\n", \
        "the staged copy still holds the pre-edit source; ninja would compile it"


def test_edited_header_is_restaged(staged):
    csrc, build, stage = staged
    stage()
    header = _stage_dir(build) / "cuda/common/shared.cuh"
    assert header.read_text() == "header v1\n"
    (csrc / "cuda/common/shared.cuh").write_text("header v2\n")
    stage()
    assert header.read_text() == "header v2\n"


def test_unchanged_files_are_not_recopied(staged):
    """Incremental: only what changed is restaged, so ninja rebuilds only the
    affected translation units instead of the whole tree."""
    csrc, build, stage = staged
    stage()
    header = _stage_dir(build) / "cuda/common/shared.cuh"
    before = header.stat().st_mtime_ns
    (csrc / "cuda/equations/acoustic2d/forward.cu").write_text("v2\n")
    stage()
    assert header.stat().st_mtime_ns == before


def test_restaged_file_is_newer_even_when_the_source_goes_back_in_time(staged):
    """`git checkout` of an older revision moves mtimes backwards; the staged
    copy must still look newer than the object built from the previous one."""
    csrc, build, stage = staged
    stage()
    src = csrc / "cuda/equations/acoustic2d/forward.cu"
    src.write_text("older content\n")
    os.utime(src, (1_000_000, 1_000_000))       # 1970-ish
    staged_copy = _stage_dir(build) / "cuda/equations/acoustic2d/cuda_equations_acoustic2d_forward.cu"
    stage()
    assert staged_copy.read_text() == "older content\n"
    assert staged_copy.stat().st_mtime_ns > src.stat().st_mtime_ns


def test_deleted_source_leaves_the_stage(staged):
    csrc, build, stage = staged
    stage()
    header = _stage_dir(build) / "cuda/common/shared.cuh"
    assert header.exists()
    (csrc / "cuda/common/shared.cuh").unlink()
    stage()
    assert not header.exists(), "a deleted source would still compile from the stage"


def test_legacy_ok_sentinel_forces_a_full_restage(staged):
    """Trees staged by the previous implementation carry `.staged` == "ok"."""
    csrc, build, stage = staged
    stage()
    manifest = _stage_dir(build) / ".staged"
    manifest.write_text("ok")
    (csrc / "cuda/equations/acoustic2d/forward.cu").write_text("v3\n")
    sources, _ = stage()
    assert Path(sources[0]).read_text() == "v3\n"
    assert isinstance(json.loads(manifest.read_text()), dict)
