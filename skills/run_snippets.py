"""Run the code blocks of an agent skill so its examples cannot rot.

Each Markdown file of the skill (SKILL.md first, then the rest sorted) gets
its own temporary working directory; its blocks run there top to bottom:

* ```python  -- written to a file and run in a fresh interpreter
* ```yaml    -- if its first line is ``# file: NAME``, written to NAME
* ```bash / ```sh -- run with ``bash -e``

Other fences are prose. A block right after the line
``<!-- skill-test: skip -->`` is not run.

    python skills/run_snippets.py skills/<name> [-k substring] [--timeout S]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SKIP_MARK = "<!-- skill-test: skip -->"
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)")
FILE_LINE = re.compile(r"^#\s*file:\s*(\S+)\s*$")


@dataclass
class Block:
    path: Path
    line: int  # 1-based line of the opening fence
    lang: str
    code: str
    skip: bool


def blocks_in(path: Path) -> list[Block]:
    out: list[Block] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    prev = ""
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            if lines[i].strip():
                prev = lines[i].strip()
            i += 1
            continue
        indent, fence, lang = m.group(1), m.group(2), m.group(3).lower()
        start = i
        body = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            line = lines[i]
            body.append(line[len(indent):] if line.startswith(indent) else line)
            i += 1
        out.append(Block(path, start + 1, lang, "\n".join(body) + "\n",
                         prev == SKIP_MARK))
        prev = ""
        i += 1
    return out


def skill_files(skill_dir: Path) -> list[Path]:
    files = sorted(p for p in skill_dir.rglob("*.md") if p.name != "SKILL.md")
    head = skill_dir / "SKILL.md"
    return ([head] if head.exists() else []) + files


def run_file(path: Path, timeout: float, env: dict | None = None) -> list[str]:
    """Run every runnable block of one Markdown file; return failure texts."""
    failures: list[str] = []
    env = dict(os.environ if env is None else env)
    # console scripts of the interpreter running this must win on PATH
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="skill-snip-") as cwd:
        for n, b in enumerate(blocks_in(path)):
            where = f"{path}:{b.line} ({b.lang})"
            if b.skip:
                continue
            if b.lang == "python":
                script = Path(cwd) / f"_block{n}.py"
                script.write_text(b.code)
                cmd = [sys.executable, str(script)]
            elif b.lang in ("bash", "sh", "shell"):
                cmd = ["bash", "-e", "-c", b.code]
            elif b.lang in ("yaml", "yml"):
                first = b.code.splitlines()[0] if b.code.strip() else ""
                m = FILE_LINE.match(first)
                if m:
                    (Path(cwd) / m.group(1)).write_text(b.code)
                continue
            else:
                continue
            try:
                r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                                   text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                failures.append(f"{where}: timed out after {timeout:.0f} s")
                continue
            if r.returncode != 0:
                tail = (r.stdout[-1500:] + "\n" + r.stderr[-3000:]).strip()
                failures.append(f"{where}: exit {r.returncode}\n{tail}")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skill_dir", type=Path)
    ap.add_argument("-k", default="", help="only files whose path contains this")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)
    env = dict(os.environ)
    bad = 0
    for f in skill_files(args.skill_dir):
        if args.k not in str(f):
            continue
        fails = run_file(f, args.timeout, env)
        n = sum(1 for b in blocks_in(f) if not b.skip
                and b.lang in ("python", "bash", "sh", "shell"))
        print(f"{'FAIL' if fails else 'ok  '} {f} ({n} blocks)")
        for msg in fails:
            print("  " + msg.replace("\n", "\n  "))
        bad += len(fails)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
