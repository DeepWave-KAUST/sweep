#!/usr/bin/env python3
"""Where the host (CPU) core is built on this runner, and the key to cache it under.

Prints ``dir=`` and ``key=`` lines for $GITHUB_OUTPUT.  The directory is the
``sweep_C`` drawer: the staged sources and the core's objects, so that a
restored drawer leaves ninja nothing to do.  The key covers what the build
reads: the files under csrc with the host flags and the unported list
(``hostcore.source_stamp``), the build code (hostcore.py, jit.py), the
compiler and the ninja the build runs, the package version (it names the
staging directory), and the runner image.  The depfiles record every header
the compiler opens, system headers included, so a drawer restored on an image
with newer headers would rebuild every object, and a cache hit is not saved
again.  Any change to these builds the core from scratch, so a compiler
update on the runner image is still caught by a full build.

    python .github/scripts/host_core_cache_key.py >> "$GITHUB_OUTPUT"
"""
import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from sweep.backend.c import hostcore, jit


def first_line(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip()


def main():
    cxx = hostcore.find_host_cxx()
    if cxx is None:
        sys.exit(f"no compiler for the host core: {hostcore._no_cxx_reason()}")
    jit._ensure_ninja_on_path()
    tools = f"{first_line([cxx, '--version'])} ninja {first_line(['ninja', '--version'])}"
    # ImageOS / ImageVersion are set on GitHub's hosted runners
    image = " ".join(filter(None, (os.environ.get("ImageOS"), os.environ.get("ImageVersion"))))
    h = hashlib.sha256()
    for part in (hostcore.source_stamp(), importlib.metadata.version("sweep-solver"), tools, image):
        h.update(part.encode() + b"\n")
    for module in (hostcore, jit):
        h.update(Path(module.__file__).read_bytes())
    label = re.sub(r"[^A-Za-z0-9.+-]+", "_", f"{tools} {image}").strip("_")
    print(f"dir={jit._build_dir()}")
    print(f"key=host-core-{platform.machine()}-{label}-{h.hexdigest()[:20]}")


if __name__ == "__main__":
    main()
