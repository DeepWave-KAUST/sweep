"""`sweep skill install` puts the skill where Claude Code / Codex read it.

Pins: the folder lands at ``<base>/.claude/skills/sweep-solver`` and
``<base>/.agents/skills/sweep-solver`` with the same files as the source;
with no assistant found and none named nothing is written; a re-install
replaces our own copy but never a foreign folder or a symlink.
"""
import zipfile
from pathlib import Path

import pytest

from sweep import skill
from sweep.cli import main


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A user with no assistant: empty HOME, nothing on PATH."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    return h


def test_the_bundled_skill_is_found_and_named_like_its_folder():
    src = skill.skill_dir()
    assert (src / "SKILL.md").is_file()
    assert skill.skill_name_of(src) == src.name == skill.SKILL_NAME


def test_nothing_found_nothing_named_installs_nothing(home, capsys):
    assert skill.detect() == []
    assert main(["skill", "install"]) == 2
    assert "--for claude" in capsys.readouterr().err
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("marker, target", [(".claude", "claude"),
                                            (".agents", "codex"),
                                            (".codex", "codex")])
def test_an_assistant_folder_is_detected(home, marker, target):
    (home / marker).mkdir()
    assert skill.detect() == [target]


def test_a_command_on_path_is_detected(home, tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    exe = bin_ / "codex"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_))
    assert skill.detect() == ["codex"]


def test_install_for_all_copies_the_whole_skill(home, capsys):
    assert main(["skill", "install", "--for", "all"]) == 0
    src = skill.skill_dir()
    for rel in (".claude/skills/sweep-solver", ".agents/skills/sweep-solver"):
        assert _files(home / rel) == _files(src)
    assert "installed for claude" in capsys.readouterr().out
    # no temporary folders left next to the skill
    assert sorted(p.name for p in (home / ".claude/skills").iterdir()) == ["sweep-solver"]


def test_default_installs_only_for_the_detected_assistant(home):
    (home / ".claude").mkdir()
    assert main(["skill", "install"]) == 0
    assert (home / ".claude/skills/sweep-solver/SKILL.md").is_file()
    assert not (home / ".agents").exists()


def test_project_install(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    assert main(["skill", "install", "--for", "codex", "--project", str(proj)]) == 0
    assert (proj / ".agents/skills/sweep-solver/SKILL.md").is_file()
    assert list(home.iterdir()) == []


def test_reinstall_replaces_our_own_copy(home):
    skill.install(["claude"])
    dest = home / ".claude/skills/sweep-solver"
    (dest / "stale.md").write_text("from an older release")
    [(_, path, how)] = skill.install(["claude"])
    assert (path, how) == (dest, "updated")
    assert _files(dest) == _files(skill.skill_dir())


def test_a_foreign_folder_is_left_alone(home):
    dest = home / ".claude/skills/sweep-solver"
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("mine")
    with pytest.raises(skill.SkillError, match="not the sweep-solver skill"):
        skill.install(["claude"])
    assert _files(dest) == ["notes.txt"]


def test_a_symlink_is_left_alone(home, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    dest = home / ".agents/skills/sweep-solver"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(checkout)
    assert main(["skill", "install", "--for", "codex"]) == 1
    assert dest.is_symlink() and list(checkout.iterdir()) == []


def test_zip_has_the_folder_on_top(home, tmp_path):
    out = tmp_path / "s.zip"
    assert main(["skill", "zip", "-o", str(out)]) == 0
    names = zipfile.ZipFile(out).namelist()
    assert "sweep-solver/SKILL.md" in names
    assert sorted(n[len("sweep-solver/"):] for n in names) == _files(skill.skill_dir())
