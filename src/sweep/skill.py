"""Install the ``sweep-solver`` Agent Skill for an AI coding assistant.

The skill (``skills/sweep-solver/`` in the repository) is a folder of
Markdown and helper scripts that Claude Code, Codex and other assistants
read when a task uses ``sweep``.  The wheel carries a copy of it under
``sweep/_skill/``, so the installed skill matches the installed solver.

    sweep skill install                 # the assistants found on this machine
    sweep skill install --for codex     # one of them, found or not
    sweep skill install --project .     # into a project instead of $HOME
    sweep skill zip -o sweep-solver.zip # for Claude.ai: upload in settings
    sweep skill path                    # where the bundled copy is

An assistant reads skills from a folder, nothing has to be registered:

* Claude Code: ``~/.claude/skills/<name>/`` and ``<project>/.claude/skills/``;
* Codex: ``~/.agents/skills/<name>/`` and ``<project>/.agents/skills/``.

So a skill copied there before the assistant is installed is picked up
when it is.  Nothing is installed when no assistant is found and none is
named: the command lists the choices instead of guessing.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

SKILL_NAME = "sweep-solver"

#: assistant -> skills folder, relative to $HOME or to a project
TARGETS = {
    "claude": Path(".claude") / "skills",
    "codex": Path(".agents") / "skills",
}

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
_NAME_LINE = re.compile(r"^name:\s*(\S+)\s*$", re.MULTILINE)


class SkillError(RuntimeError):
    """An install that would do the wrong thing; the message says why."""


def skill_dir() -> Path:
    """The skill folder: the wheel's copy, else the source checkout's."""
    pkg = Path(__file__).resolve().parent
    for cand in (pkg / "_skill" / SKILL_NAME,              # wheel
                 pkg.parents[1] / "skills" / SKILL_NAME):  # src/sweep -> repo
        if (cand / "SKILL.md").is_file():
            return cand
    raise SkillError(
        f"the {SKILL_NAME} skill is not in this installation ({pkg}); "
        "take skills/sweep-solver from https://github.com/DeepWave-KAUST/sweep")


def skill_name_of(folder: Path) -> Optional[str]:
    """The ``name:`` in a folder's SKILL.md front matter, or None."""
    try:
        text = (Path(folder) / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    m = _NAME_LINE.search(text.split("---", 2)[1])
    return m.group(1) if m else None


def detect(home: Optional[Path] = None) -> List[str]:
    """Assistants that look installed for this user: their config folder
    exists in ``home`` or their command is on PATH."""
    home = Path.home() if home is None else Path(home)
    found = []
    if (home / ".claude").is_dir() or shutil.which("claude"):
        found.append("claude")
    if ((home / ".agents").is_dir() or (home / ".codex").is_dir()
            or shutil.which("codex")):
        found.append("codex")
    return found


def destination(target: str, base: Path, name: str = SKILL_NAME) -> Path:
    if target not in TARGETS:
        raise SkillError(f"unknown assistant {target!r}; one of {sorted(TARGETS)}")
    return Path(base) / TARGETS[target] / name


def copy_skill(src: Path, dest: Path) -> str:
    """Copy ``src`` to ``dest``; return ``"installed"`` or ``"updated"``.

    An existing ``dest`` is replaced only when it is this same skill (its
    SKILL.md has the same ``name:``); anything else there, or a symlink
    (a checkout someone linked on purpose), is left alone with an error.
    The new copy is written next to ``dest`` and renamed into place, so a
    failed copy never leaves a half-written skill.
    """
    src, dest = Path(src), Path(dest)
    name = skill_name_of(src)
    if dest.is_symlink():
        raise SkillError(f"{dest} is a symlink to {os.readlink(dest)}; "
                         "remove it to install a copy")
    if dest.exists() and skill_name_of(dest) != name:
        raise SkillError(f"{dest} exists and is not the {name} skill; "
                         "move it away to install here")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        shutil.copytree(src, tmp / "skill", ignore=_IGNORE)
        old = None
        if dest.exists():
            old = tmp / "old"
            dest.rename(old)
        try:
            (tmp / "skill").rename(dest)
        except OSError:
            if old is not None:
                old.rename(dest)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "updated" if old is not None else "installed"


def install(targets: Optional[Iterable[str]] = None, *,
            project: Optional[Path] = None, home: Optional[Path] = None,
            src: Optional[Path] = None) -> List[tuple]:
    """Install the skill for ``targets`` (default: :func:`detect`).

    Into ``project``'s ``.claude/skills`` / ``.agents/skills`` when given,
    else into ``home``'s.  Returns ``[(target, path, "installed"|"updated")]``.
    """
    home = Path.home() if home is None else Path(home)
    if targets is None:
        targets = detect(home)
        if not targets:
            raise SkillError(no_assistant_message())
    src = skill_dir() if src is None else Path(src)
    base = home if project is None else Path(project)
    done = []
    for t in dict.fromkeys(targets):
        dest = destination(t, base, skill_name_of(src) or src.name)
        done.append((t, dest, copy_skill(src, dest)))
    return done


def make_zip(out: Path, src: Optional[Path] = None) -> Path:
    """Zip the skill with its folder at the top, as Claude.ai expects."""
    src = skill_dir() if src is None else Path(src)
    name = skill_name_of(src) or src.name
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            rel = f.relative_to(src)
            if f.is_file() and "__pycache__" not in rel.parts and f.suffix != ".pyc":
                zf.write(f, Path(name) / rel)
    return out


def no_assistant_message(command: str = "sweep skill") -> str:
    return (
        "no Claude Code or Codex found for this user (no ~/.claude, ~/.agents "
        "or ~/.codex, no `claude` or `codex` on PATH), so nothing was installed.\n"
        f"  {command} install --for claude   # Claude Code: ~/.claude/skills/\n"
        f"  {command} install --for codex    # Codex: ~/.agents/skills/\n"
        f"  {command} install --for all      # both\n"
        f"  {command} zip                    # Claude.ai: upload the zip in settings\n"
        "Run it on the machine where the assistant runs; add --project DIR to "
        "install into a project (what a cloud session sees).")


def add_parser(sub, command: str = "sweep skill") -> None:
    """Add ``skill`` and its subcommands to an argparse subparsers object."""
    p = sub.add_parser("skill", help="install the Agent Skill for Claude Code / Codex",
                       description=__doc__.split("\n\n")[0])
    ss = p.add_subparsers(dest="skill_command", metavar="<install|zip|path>")
    pi = ss.add_parser("install", help="copy the skill to where an assistant reads it")
    pi.add_argument("--for", dest="targets", choices=[*TARGETS, "all"],
                    action="append",
                    help="assistant to install for (repeatable); default: the "
                         "ones found on this machine")
    pi.add_argument("--project", type=Path, metavar="DIR", nargs="?", const=Path("."),
                    help="install into DIR/.claude/skills or DIR/.agents/skills "
                         "(default DIR: the current directory) instead of $HOME")
    pz = ss.add_parser("zip", help="zip the skill for upload to Claude.ai")
    pz.add_argument("-o", "--output", type=Path, default=Path(f"{SKILL_NAME}.zip"))
    ss.add_parser("path", help="print where the bundled skill is")
    p.set_defaults(skill_parser=p, skill_cmd_name=command)


def run(args) -> int:
    """Carry out a parsed ``skill`` command; return the exit code."""
    import sys

    try:
        if args.skill_command == "install":
            targets = args.targets or detect()
            if not targets:
                print(no_assistant_message(args.skill_cmd_name), file=sys.stderr)
                return 2
            if "all" in targets:
                targets = list(TARGETS)
            done = install(targets, project=args.project)
            for t, path, how in done:
                print(f"{how} for {t}: {path}")
            print("A running Claude Code or Codex session may need a restart "
                  "to see a skill folder it did not have before.")
            return 0
        if args.skill_command == "zip":
            out = make_zip(args.output)
            print(f"wrote {out}; upload it in the skills section of Claude.ai's settings")
            return 0
        if args.skill_command == "path":
            print(skill_dir())
            return 0
    except SkillError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    args.skill_parser.print_help()
    return 0
