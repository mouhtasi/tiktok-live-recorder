"""
Tests for the forkserver preload (tiktak §56): the recorder imported once, in
the forkserver, so ~190 monitors share one copy instead of building their own.

## What could go wrong, and why each test exists

The preload changes *what exists before the fork*. Everything the forkserver
creates at import, every child inherits. Each test below names one thing that
inheritance could break:

- **threads** — a fork copies only the calling thread. A thread started at
  import (by curl_cffi, say) holding a lock at fork time leaves that lock held
  forever in every child: a monitor that deadlocks on its first request.
- **TLS** — libcurl and OpenSSL initialise in the forkserver, before the fork.
  The monitors' entire job is HTTPS through curl_cffi, so a child must complete
  a real TLS request, through the recorder's own ``HttpClient``.
- **logs** — the logger singleton opens ``tiktok-recorder.log`` at import; under
  the preload that happens in the forkserver and every child inherits it. The
  2026-08-29 storm destroyed its own evidence through exactly one shared
  rotating handler, so the per-account split must still hold.
- **random** — ``random`` seeds at import. If children inherited one seed, all
  ~190 would draw the same first-poll jitter and poll TikTok in lockstep.
- **the guard** — the stdlib forkserver swallows only ImportError from a
  preload; anything else kills it, and with it every monitor spawn.
- **memory** (Linux only) — the point of the change. Without it, a new import
  outside the preload list would silently give the 2.8 GB back.

🚨 **Every child reports whether it inherited the recorder** (``preloaded``) and
each test asserts it. Without that, all of these pass with the preload off —
they would be testing today's behaviour and reporting green.
"""

import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("forkserver_driver.py")


def drive(tmp_path, *args):
    out = subprocess.run(
        [sys.executable, str(DRIVER), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def assert_all_preloaded(result):
    assert result["preload_enabled"] is True
    assert [c["preloaded"] for c in result["children"]] == [True] * len(
        result["children"]
    )


def test_importing_the_preload_starts_no_thread_and_freezes(tmp_path):
    result = drive(tmp_path, "import", "-")
    assert result["python_threads"] == 1
    assert result["native_threads"] == 1
    assert result["loaded"] == result["modules"]
    assert result["freeze_count"] > 0


def test_preload_survives_a_module_that_raises(tmp_path, monkeypatch):
    (tmp_path / "boom_on_import.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    from utils.forkserver_preload import _preload

    try:
        loaded = _preload(["boom_on_import", "no_such_module_at_all", "json"])
    finally:
        gc.unfreeze()
    assert loaded == ["json"]


@pytest.fixture
def self_signed_cert(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "1",
         "-subj", "/CN=127.0.0.1"],
        check=True, capture_output=True,
    )
    return str(cert), str(key)


def test_preloaded_children_complete_tls_requests(tmp_path, self_signed_cert):
    result = drive(tmp_path, "tls", "preload", *self_signed_cert)
    assert_all_preloaded(result)
    for child in result["children"]:
        assert child["curl_status"] == 200
        assert child["curl_body"] == "ok-from-test-server"
        assert child["requests_status"] == 200


def test_worker_log_files_still_split_under_preload(tmp_path):
    result = drive(tmp_path, "logs", "preload")
    assert_all_preloaded(result)
    alice = (tmp_path / "logs" / "monitor-alice.log").read_text()
    bob = (tmp_path / "logs" / "monitor-bob.log").read_text()
    assert "marker-alice" in alice and "marker-bob" not in alice
    assert "marker-bob" in bob and "marker-alice" not in bob
    main_log = tmp_path / "tiktok-recorder.log"
    shared = main_log.read_text() if main_log.exists() else ""
    assert "marker-" not in shared
    for child in result["children"]:
        # None off Linux, where there is no /proc to count descriptors through.
        assert child["main_log_fds"] in (0, None)


def test_children_draw_different_first_poll_jitter(tmp_path):
    result = drive(tmp_path, "random", "preload")
    assert_all_preloaded(result)
    draws = [c["draw"] for c in result["children"]]
    assert len(set(draws)) == len(draws), draws


@pytest.mark.skipif(
    not Path("/proc/self/smaps_rollup").exists(), reason="needs Linux smaps_rollup"
)
def test_preload_cuts_each_childs_private_memory_by_more_than_half(tmp_path):
    with_preload = drive(tmp_path, "memory", "preload")
    without = drive(tmp_path, "memory", "none")
    assert_all_preloaded(with_preload)
    assert [c["preloaded"] for c in without["children"]] == [False] * 3
    mean = lambda r: sum(c["private_dirty_kb"] for c in r["children"]) / 3
    # Measured on prod 2026-10-15: 20.4 MB without, 4.5 MB with.
    assert mean(with_preload) < 0.5 * mean(without), (mean(with_preload), mean(without))
