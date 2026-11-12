"""The build lock (``jit._FileBaton``) of the staging and of both cores.

A process killed while holding it -- SIGKILLed by a timeout mid-build, a
restarted notebook kernel, a cancelled job -- never removes its lock file, and
every later build used to wait on that file forever.  The lock now names its
holder, and a waiter on the holder's machine removes it once the holder is
gone.
"""
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from sweep.backend.c import hostcore, jit

linux_proc = pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="needs /proc")


def _bounded(fn, seconds, *args):
    """Run ``fn`` in a thread; fail instead of hanging the suite when it blocks."""
    out = {}

    def run():
        out["value"] = fn(*args)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"{getattr(fn, '__name__', fn)} still blocked after {seconds} s"
    return out.get("value")


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _stale_lock(path, pid=None):
    path.write_text(f"{pid or _dead_pid()} {jit._machine()} 1\n")
    return path


def test_the_lock_names_its_holder(tmp_path):
    baton = jit._FileBaton(str(tmp_path / "lock"))
    assert baton.try_acquire()
    pid, machine, *_ = (tmp_path / "lock").read_text().split()
    assert (int(pid), machine) == (os.getpid(), jit._machine())
    assert baton.holder().startswith(f"pid {os.getpid()} on ")
    assert not jit._FileBaton(str(tmp_path / "lock")).try_acquire()
    baton.release()
    assert not (tmp_path / "lock").exists()


def test_a_holder_killed_mid_build_does_not_block_the_next(tmp_path):
    """The CI failure: a process SIGKILLed while it holds the lock."""
    lock = tmp_path / "sweep_core_cpu_lock"
    holder = ("import importlib.util, os, signal, sys\n"
              "spec = importlib.util.spec_from_file_location('jit', sys.argv[1])\n"
              "jit = importlib.util.module_from_spec(spec)\n"
              "spec.loader.exec_module(jit)\n"
              "assert jit._FileBaton(sys.argv[2]).try_acquire()\n"
              "os.kill(os.getpid(), signal.SIGKILL)\n")
    r = subprocess.run([sys.executable, "-c", holder, jit.__file__, str(lock)], timeout=60)
    assert r.returncode == -signal.SIGKILL and lock.exists()

    baton = jit._FileBaton(str(lock))
    _bounded(baton.wait, 10)
    assert not lock.exists(), "the dead holder's lock is still there"
    assert baton.try_acquire()
    baton.release()


def test_a_live_holder_is_waited_on(tmp_path):
    lock = tmp_path / "lock"
    holder = jit._FileBaton(str(lock))
    assert holder.try_acquire()
    waiter = jit._FileBaton(str(lock), wait_seconds=0.01)
    assert not waiter._remove_if_holder_gone()
    threading.Timer(0.3, holder.release).start()
    t0 = time.monotonic()
    _bounded(waiter.wait, 10)
    assert time.monotonic() - t0 >= 0.25, "returned before the holder released"


@pytest.mark.parametrize("record", [
    "",                                   # an older sweep writes no record
    "garbage\n",
    "12345 another-host|boot|pid:[1] 1\n",
])
def test_a_holder_that_cannot_be_checked_is_not_gone(record):
    assert not jit._holder_gone(record)


def test_a_holder_whose_pid_is_free_is_gone():
    assert jit._holder_gone(f"{_dead_pid()} {jit._machine()} 1\n")


@linux_proc
def test_a_holder_whose_pid_was_reused_is_gone():
    """The PID is alive but names another process: its start time differs."""
    start = jit._proc_stat(os.getpid())[1]
    assert not jit._holder_gone(f"{os.getpid()} {jit._machine()} {start}\n")
    assert jit._holder_gone(f"{os.getpid()} {jit._machine()} {int(start) + 1}\n")


@linux_proc
def test_a_zombie_holder_is_gone():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 30
        while jit._proc_stat(p.pid)[0] != "Z":
            assert time.monotonic() < deadline, "the child did not exit"
            time.sleep(0.01)
        assert jit._holder_gone(f"{p.pid} {jit._machine()} {jit._proc_stat(p.pid)[1]}\n")
    finally:
        p.wait()


def _take_turns(lock, inside, most, done):
    baton = jit._FileBaton(lock, wait_seconds=0.01)
    baton.acquire()
    try:
        with inside.get_lock():
            inside.value += 1
            most.value = max(most.value, inside.value)
        time.sleep(0.05)
        with inside.get_lock():
            inside.value -= 1
    finally:
        baton.release()
    with done.get_lock():
        done.value += 1


def test_waiters_on_a_dead_holder_still_take_turns(tmp_path):
    """Removing the dead holder's lock lets one waiter in at a time: a waiter
    never removes the fresh lock another took meanwhile."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    lock = _stale_lock(tmp_path / "lock")
    inside, most, done = ctx.Value("i", 0), ctx.Value("i", 0), ctx.Value("i", 0)
    procs = [ctx.Process(target=_take_turns, args=(str(lock), inside, most, done), daemon=True)
             for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert all(p.exitcode == 0 for p in procs)
    assert done.value == 6 and most.value == 1, f"{most.value} processes held the lock at once"
    assert not lock.exists() and not (tmp_path / "lock.break").exists()


@linux_proc
def test_the_build_stops_when_its_holder_dies(tmp_path):
    """ninja runs under jit._NINJA_LAUNCHER: killing the process that started
    it SIGTERMs it, so no orphaned build runs on beside the next holder's."""
    bindir, pidfile = tmp_path / "bin", tmp_path / "ninja.pid"
    bindir.mkdir()
    (bindir / "ninja").write_text(f'#!/bin/sh\necho $$ > "{pidfile}.tmp"\n'
                                  f'mv "{pidfile}.tmp" "{pidfile}"\nexec sleep 60\n')
    (bindir / "ninja").chmod(0o755)
    holder = ("import importlib.util, subprocess, sys\n"
              "spec = importlib.util.spec_from_file_location('jit', sys.argv[1])\n"
              "jit = importlib.util.module_from_spec(spec)\n"
              "spec.loader.exec_module(jit)\n"
              "subprocess.run(jit._ninja_command(sys.argv[2]))\n")
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
    p = subprocess.Popen([sys.executable, "-c", holder, jit.__file__, str(tmp_path)], env=env)
    try:
        deadline = time.monotonic() + 30
        while not pidfile.exists():
            assert time.monotonic() < deadline, "the build never started"
            time.sleep(0.02)
        pid = int(pidfile.read_text())
        start = jit._proc_stat(pid)[1]
        p.kill()
        p.wait()

        def running():
            st = jit._proc_stat(pid)
            return st is not None and st[0] not in ("Z", "X") and st[1] == start
        deadline = time.monotonic() + 10
        while running():
            assert time.monotonic() < deadline, "the build outlived the process that started it"
            time.sleep(0.05)
    finally:
        p.kill()
        p.wait()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores the mode")
def test_a_dead_holders_lock_that_cannot_be_removed_is_waited_on(tmp_path):
    """A shared drawer this user may not write: the waiter waits, as before,
    instead of raising or spinning."""
    d = tmp_path / "shared"
    d.mkdir()
    lock = _stale_lock(d / "lock")
    d.chmod(0o555)
    try:
        baton = jit._FileBaton(str(lock))
        baton._gone = (lock.read_text(), time.monotonic() - 5)   # seen gone long enough
        assert not baton._remove_if_holder_gone()
        assert lock.exists()
    finally:
        d.chmod(0o755)


def test_a_breaker_left_by_a_dead_waiter_is_cleared(tmp_path):
    lock = _stale_lock(tmp_path / "lock")
    breaker = tmp_path / "lock.break"
    breaker.write_text("")
    old = time.time() - 120
    os.utime(breaker, (old, old))
    baton = jit._FileBaton(str(lock), wait_seconds=0.01)
    _bounded(baton.acquire, 10)
    baton.release()
    assert not lock.exists() and not breaker.exists()


def test_staging_completes_after_a_dead_holder(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    _stale_lock(build / "sweep_stage_lock")
    staged, _ = _bounded(jit._stage, 120, build)
    assert staged and all(os.path.exists(s) for s in staged)


def test_the_cuda_core_builds_after_a_dead_holder(tmp_path, monkeypatch):
    """The waiter takes the core only when its stamp is current; after a
    holder that died midway it is not, so the waiter builds it."""
    build = tmp_path / "build"
    (build / "core").mkdir(parents=True)
    _stale_lock(build / "sweep_core_lock")
    monkeypatch.setattr(jit, "_core_ninja", lambda *a: "")
    monkeypatch.setattr(jit, "_ensure_ninja_on_path", lambda: None)
    monkeypatch.setattr(jit, "_source_stamp", lambda: "this-tree")
    built = []

    def ninja(cmd, **kw):
        built.append(cmd)
        (build / "core" / "libsweep_core.so").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", ninja)
    so = _bounded(jit._build_core, 30, build, [], [], "/no/cuda", False)
    assert so == build / "core" / "libsweep_core.so" and len(built) == 1
    assert jit._local_core_current(build)


def test_the_host_core_builds_after_a_dead_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setattr(hostcore, "find_host_cxx", lambda: "/usr/bin/g++")
    monkeypatch.setattr(hostcore, "_cxx_describe", lambda cxx: "g++ 11")
    monkeypatch.setattr(jit, "_ensure_ninja_on_path", lambda: None)
    d = hostcore.build_dir()
    d.mkdir(parents=True)
    _stale_lock(d.parent / "sweep_core_cpu_lock")
    built = []

    def ninja(cmd, **kw):
        built.append(cmd)
        (d / "libsweep_core.so").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(hostcore.subprocess, "run", ninja)
    so = _bounded(hostcore.build_core, 120, False)
    assert so == d / "libsweep_core.so" and len(built) == 1
    assert hostcore.cached_local_core() == so
