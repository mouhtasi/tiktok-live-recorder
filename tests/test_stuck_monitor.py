"""
§94 — a monitor that is alive but stuck watches nobody.

Found 2026-09-24 by tiktak's late-join measure on its first event: @_bbylola_,
polled every 5 minutes, was joined 80 minutes into a broadcast. Her monitor had
been blocked for **six days** inside `download_live_stream()` — a stream that
stalled without closing its socket, read with no timeout. Nine monitors were
stuck that way when a routine restart healed them by accident; the supervisor
reported every one alive and healthy, because "alive" was all it could see.

Two fixes, each pinned here:

1. The stream read has a timeout, so a stall raises into start_recording()'s
   existing "network hiccup" path: re-check the room, reconnect or finish and
   convert.
2. Every monitor publishes a *deadline* — the time by which it promises to act
   again (poll, write, convert). The supervisor treats a live worker past its
   deadline as stuck, ends it, and respawns it at once, counting it apart from
   deaths. That covers any hang the timeouts miss.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest
import requests

from core.supervisor import RecorderSupervisor
from core.tiktok_api import TikTokAPI
from core.tiktok_recorder import TikTokRecorder


# --- 1. the stream read can no longer block forever -------------------------------


def test_the_stream_is_read_with_a_finite_timeout():
    api = TikTokAPI.__new__(TikTokAPI)
    api._http_client_stream = Mock()
    api._http_client_stream.get.return_value.iter_content.return_value = iter([b"x"])

    list(api.download_live_stream("http://cdn/x.flv"))

    kwargs = api._http_client_stream.get.call_args.kwargs
    connect, read = kwargs["timeout"]
    assert 0 < connect <= 30 and 0 < read <= 120


def _recorder(tmp_path, deadline=None):
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "tester"
    rec.duration = None
    rec.output = str(tmp_path)
    rec.bitrate = None
    rec.ffmpeg_path = None
    rec.mode = None
    rec._stop_now_event = None
    rec._deadline = deadline
    rec.should_stop_now = lambda: False
    rec.tiktok = Mock()
    rec.tiktok.get_live_url.return_value = "http://cdn/x.flv"
    rec.tiktok.last_room_created_at = None
    return rec


def test_a_stalled_stream_ends_the_recording_and_keeps_what_was_written(tmp_path):
    """The read timeout surfaces as requests' ConnectionError mid-iteration."""
    def stalls():
        yield b"x" * (600 * 1024)
        raise requests.exceptions.ConnectionError("Read timed out.")

    rec = _recorder(tmp_path)
    rec.tiktok.is_room_alive.side_effect = [True, False]   # gone after the stall
    rec.tiktok.download_live_stream.return_value = stalls()

    with patch("core.tiktok_recorder.VideoManagement") as vm, patch(
        "core.tiktok_recorder.time.sleep"
    ):
        rec.start_recording("tester", "room-1")

    [capture] = tmp_path.glob("TK_tester_*_flv.mp4")
    assert capture.stat().st_size >= 600 * 1024
    assert vm.convert_flv_to_mp4.call_count == 1


# --- 2. the monitor publishes a deadline ------------------------------------------


class _Value:
    """multiprocessing.Value's interface, without a process."""
    def __init__(self):
        self.value = 0.0


def test_promise_sets_the_deadline_and_is_a_no_op_without_one():
    rec = _recorder(None, deadline=_Value())
    before = time.time()
    rec._promise(120)
    assert before + 120 <= rec._deadline.value <= time.time() + 120

    bare = TikTokRecorder.__new__(TikTokRecorder)   # a hand-built double
    bare._promise(120)                              # must not raise


def test_waiting_for_the_next_poll_promises_past_the_wait(tmp_path):
    """A cold monitor sleeping 20 minutes is not stuck; its promise says so."""
    rec = _recorder(tmp_path, deadline=_Value())
    rec._stop_event = Mock()
    rec._wait_for_next_poll(1200)
    assert rec._deadline.value >= time.time() + 1200


class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


class _StopEvent:
    """Never set; each wait advances the fake clock by the full timeout."""
    def __init__(self, clock):
        self.clock = clock
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        self.clock.t += timeout
        return False


def test_a_retier_mid_wait_lands_within_a_slice(tmp_path):
    """2026-09-24: @yumehime555's stream ended 05:41 and her monitor began a
    60-minute cold wait; ingest made her hot two minutes later, and she was
    live again at 05:42:50 — joined 26 minutes in, and only because a restart
    happened. The interval is re-read between slices, so it lands in ≤5 min."""
    clock = _Clock()
    rec = _recorder(tmp_path)
    rec._stop_event = _StopEvent(clock)
    tiers = iter([5])                       # hot from the first re-read on
    rec._poll_interval_minutes = lambda: next(tiers, 5)

    with patch("core.tiktok_recorder.time.monotonic", clock.monotonic):
        rec._wait_for_next_poll(3600)

    assert clock.t <= 300


def test_an_unchanged_tier_waits_the_whole_interval(tmp_path):
    clock = _Clock()
    rec = _recorder(tmp_path)
    rec._stop_event = _StopEvent(clock)
    rec._poll_interval_minutes = lambda: 20

    with patch("core.tiktok_recorder.time.monotonic", clock.monotonic):
        rec._wait_for_next_poll(1200)

    assert clock.t == 1200


def test_recording_keeps_the_deadline_moving(tmp_path):
    deadline = _Value()
    seen = []

    def chunks():
        for _ in range(3):
            yield b"x" * (600 * 1024)
            seen.append(deadline.value)

    rec = _recorder(tmp_path, deadline=deadline)
    rec.tiktok.is_room_alive.side_effect = [True, False]
    rec.tiktok.download_live_stream.return_value = chunks()
    with patch("core.tiktok_recorder.VideoManagement"):
        rec.start_recording("tester", "room-1")

    assert seen and all(v > time.time() for v in seen)


# --- 3. the supervisor ends a stuck worker and replaces it ------------------------


class _Worker:
    def __init__(self, username):
        self.username = username
        self.deadline = 0.0
        self._alive = True
        self.terminated = False

    def is_alive(self):
        return self._alive

    def seconds_overdue(self, now):
        if not self.deadline:
            return None
        return max(0.0, now - self.deadline)

    def terminate(self):
        self.terminated = True
        self._alive = False

    def request_stop(self):
        pass

    def stop_recording_now(self):
        pass

    def join(self, timeout=None):
        pass


def _supervisor(tmp_path, wall):
    (tmp_path / "users.txt").write_text("alice\nbob\n")
    spawned = []

    def spawn(username, stagger=True):
        w = _Worker(username)
        spawned.append(w)
        return w

    sup = RecorderSupervisor(
        watchlist_path=tmp_path / "users.txt",
        spawn_worker=spawn,
        health_path=tmp_path / "health.json",
        now=lambda: 10_000.0,
        wall=lambda: wall[0],
    )
    return sup, spawned


def test_a_worker_past_its_deadline_is_ended_and_respawned(tmp_path):
    wall = [1_000_000.0]
    sup, spawned = _supervisor(tmp_path, wall)
    sup.reconcile()
    alice, bob = spawned
    alice.deadline = wall[0] + 60     # promised to act within a minute
    bob.deadline = wall[0] + 3600

    wall[0] += 600                    # ten minutes later: alice is overdue
    sup.reconcile()
    assert alice.terminated and not bob.terminated

    sup.reconcile()                   # the next pass replaces her
    assert sup.workers["alice"] is not alice

    health = json.loads((tmp_path / "health.json").read_text())
    assert health["stuck_total"] == 1
    assert health["deaths_total"] == 0      # a stuck kill is not a crash
    assert [s["user"] for s in health["stuck_recent"]] == ["alice"]


def test_a_worker_that_has_not_promised_anything_is_not_judged(tmp_path):
    """Deadline 0 = not reported yet (starting up, or an older monitor)."""
    wall = [1_000_000.0]
    sup, spawned = _supervisor(tmp_path, wall)
    sup.reconcile()
    wall[0] += 86_400
    sup.reconcile()
    assert not any(w.terminated for w in spawned)
    assert json.loads((tmp_path / "health.json").read_text())["stuck_total"] == 0


def _promise_then_block(config):
    """A monitor that keeps no promise: one short deadline, then a long block."""
    config.deadline.value = time.time() + 0.5
    time.sleep(3600)


def test_a_real_blocked_process_is_ended_and_replaced(tmp_path):
    """The fakes cannot prove the two things that matter across a process
    boundary: that the child's promise reaches the parent through the shared
    Value, and that terminate() ends a child blocked in a call."""
    import multiprocessing
    from main import _Worker
    from utils.recorder_config import RecorderConfig
    from utils.enums import Mode

    (tmp_path / "users.txt").write_text("alice\n")
    procs = []

    def spawn(username, stagger=True):
        config = RecorderConfig(mode=Mode.AUTOMATIC, user=username)
        config.deadline = multiprocessing.Value("d", 0.0)
        proc = multiprocessing.Process(target=_promise_then_block, args=(config,))
        proc.start()
        procs.append(proc)
        return _Worker(username, proc, multiprocessing.Event(),
                       multiprocessing.Event(), config.deadline)

    sup = RecorderSupervisor(watchlist_path=tmp_path / "users.txt",
                             spawn_worker=spawn, health_path=tmp_path / "h.json")
    try:
        sup.reconcile()
        first = procs[0]
        end = time.time() + 10
        while time.time() < end and first.is_alive():
            time.sleep(0.2)
            sup.reconcile()
        assert not first.is_alive()
        assert len(procs) >= 2 and procs[1].is_alive()
        assert json.loads((tmp_path / "h.json").read_text())["stuck_total"] >= 1
    finally:
        for p in procs:
            p.kill()
            p.join(5)


def test_a_worker_without_the_deadline_protocol_is_left_alone(tmp_path):
    """The existing fakes, and a handle from before §94, have no seconds_overdue."""
    (tmp_path / "users.txt").write_text("alice\n")

    class Old:
        def is_alive(self):
            return True

    sup = RecorderSupervisor(
        watchlist_path=tmp_path / "users.txt",
        spawn_worker=lambda u, stagger=True: Old(),
        wall=lambda: 1e12,
    )
    sup.reconcile()
    sup.reconcile()
    assert "alice" in sup.workers
