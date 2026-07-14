"""
Tests for cooperative worker shutdown (PLAN §37) — how a per-user monitor is
asked to go away without wrecking anything.

## The rule

**A stop request must never truncate a recording.** `automatic_mode()` is a
`while True` of: resolve room-id -> `manual_mode()` -> (if live)
`start_recording()`, which **blocks for the entire broadcast** -> sleep. So the
stop flag is checked at the **top of the loop**, i.e. at a poll boundary, which
means a worker asked to stop mid-broadcast finishes writing its file first and
exits afterwards. Cooperative, not pre-emptive: no signal, no terminate(), no
half-written FLV.

This is what lets §37 retire the §35 deferral guard. Today a watch-list change
restarts the ONE recorder subprocess, truncating *every* recording in flight —
so tiktak has to defer the change until nothing is recording. With per-worker
cooperative stop, removing @a cannot truncate @b (different process, untouched)
and cannot even truncate @a (it finishes first). Nothing to defer.

## Why not just terminate() the worker?

Because the worker holds the open output file. SIGTERM mid-stream leaves a
truncated FLV that our ingest then has to guess about — and §29 already taught
us what "recorder processes we didn't cleanly account for" costs (orphaned trees
double-writing → corruption). Killing is what we're getting *away* from.
"""

from unittest.mock import Mock, patch

import pytest

from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import UserLiveError
from utils.enums import TimeOut


class _BreakLoop(Exception):
    """Raised from a patched time.sleep to escape the infinite poll loop."""


def _make_recorder(should_stop=None, interval=5, user="tester"):
    rec = TikTokRecorder.__new__(TikTokRecorder)  # bypass __init__/network
    rec.user = user
    rec.automatic_interval = interval
    rec.tiktok = Mock()
    rec.should_stop = should_stop or (lambda: False)
    return rec


# NOTE ON TRIPWIRES: these tests assert that automatic_mode() *returns*. Against
# code with no stop check it would instead loop forever and hang the suite — a
# useless red. So each one arms a tripwire (_BreakLoop) on the path the loop would
# take if it kept going, turning "no stop check" into a fast, legible failure.


def test_stop_requested_exits_at_the_poll_boundary():
    # Flag already set: the loop must exit immediately, WITHOUT polling TikTok.
    recorder = _make_recorder(should_stop=lambda: True)
    recorder.tiktok.get_room_id_from_user.side_effect = UserLiveError("not live")

    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop):
        recorder.automatic_mode()  # returns, rather than looping forever

    recorder.tiktok.get_room_id_from_user.assert_not_called()


def test_stop_mid_recording_lets_the_recording_finish():
    """THE test. The stop arrives while start_recording() is blocked writing the
    stream. The recording must complete and only then may the worker exit."""
    events = []
    stop = {"requested": False}

    recorder = _make_recorder(should_stop=lambda: stop["requested"])
    recorder.tiktok.get_room_id_from_user.return_value = "room-1"

    def fake_manual_mode():
        if stop["requested"]:  # tripwire: a second recording must never start
            raise _BreakLoop("worker kept polling after being told to stop")
        # Stand in for start_recording()'s long blocking write. The supervisor
        # asks us to stop half-way through the broadcast...
        events.append("recording_started")
        stop["requested"] = True
        # ...and we keep going to the end of the stream regardless.
        events.append("recording_finished")

    recorder.manual_mode = fake_manual_mode

    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop):
        recorder.automatic_mode()

    # The recording ran to completion — not cut short — and then the worker left.
    assert events == ["recording_started", "recording_finished"]
    # And it did not start a *new* poll cycle after being told to stop.
    assert recorder.tiktok.get_room_id_from_user.call_count == 1


def test_stop_while_sleeping_between_polls_exits_without_another_poll():
    # The common case: the user isn't live, we're idling between polls, and the
    # account gets toggled off. Next time round the loop we simply leave.
    stop = {"requested": False}
    sleeps = []
    recorder = _make_recorder(should_stop=lambda: stop["requested"])
    recorder.tiktok.get_room_id_from_user.side_effect = UserLiveError("not live")

    def fake_sleep(_seconds):
        if stop["requested"]:  # tripwire: we should never idle a second time
            raise _BreakLoop("worker kept polling after being told to stop")
        sleeps.append(_seconds)
        stop["requested"] = True  # toggled off while we were idling

    with patch("core.tiktok_recorder.time.sleep", side_effect=fake_sleep):
        recorder.automatic_mode()

    assert recorder.tiktok.get_room_id_from_user.call_count == 1
    assert len(sleeps) == 1


def test_without_a_stop_flag_the_loop_still_runs_forever():
    """Regression guard: the default (no stop requested) behaviour is unchanged —
    a monitor must not quietly exit just because §37 added a check. A monitor
    that silently stops looping IS §28."""
    recorder = _make_recorder(should_stop=lambda: False)
    recorder.tiktok.get_room_id_from_user.side_effect = UserLiveError("not live")

    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop) as slp:
        with pytest.raises(_BreakLoop):  # only our sentinel breaks it out
            recorder.automatic_mode()

    slp.assert_called_once_with(5 * TimeOut.ONE_MINUTE)


def test_transient_error_does_not_exit_the_worker():
    """§28 again, in the new structure: a transient poll error must still be
    caught and retried, not mistaken for a reason to leave the loop."""
    recorder = _make_recorder(should_stop=lambda: False)
    recorder.tiktok.get_room_id_from_user.side_effect = RuntimeError("boom")

    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop) as slp:
        with pytest.raises(_BreakLoop):
            recorder.automatic_mode()

    slp.assert_called_once_with(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)


# --- force stop: interrupt the recording NOW, but still finalize it ----------
#
# "Stop recording @a right now, even though they're mid-broadcast" — without
# touching @b, and without leaving a mess behind.
#
# The naive way is to SIGKILL the worker. That is wrong: the file on disk is raw
# **FLV bytes** (the `.mp4` suffix is a lie until convert_flv_to_mp4() runs at the
# end of start_recording), so killing mid-stream leaves an unconverted FLV that
# our ingest still matches with glob("TK_*.mp4") — we would archive a mislabelled
# file. It also puts us back in the business of terminating recorder processes,
# which is §29's neighbourhood.
#
# So force-stop breaks out of the *chunk loop* instead. The worker then falls
# through the existing `finally` (flush) -> closes the file -> converts -> exits:
# the same path KeyboardInterrupt already takes today. Everything recorded up to
# that moment is kept, converted, and ingestable.


def _make_recording_recorder(tmp_path, chunks, should_stop_now):
    rec = _make_recorder()
    rec.should_stop_now = should_stop_now
    rec.duration = None
    rec.output = str(tmp_path)
    rec.bitrate = None
    rec.ffmpeg_path = None
    rec.mode = None
    rec.tiktok.get_live_url.return_value = "http://cdn.example/live.flv"
    # Tripwire: if the force-stop check is missing, the download loop drains the
    # whole stream; is_room_alive then goes False so the test FAILS on the
    # assertions rather than spinning forever on an exhausted iterator.
    rec.tiktok.is_room_alive.side_effect = [True, True, False]
    rec.tiktok.download_live_stream.return_value = iter(chunks)
    return rec


def test_force_stop_interrupts_the_recording_but_still_converts(tmp_path):
    """THE force-stop test. The interrupt lands mid-broadcast; the bytes we have
    must be flushed, the file closed, and the conversion run — not abandoned."""
    seen = []
    # Stop once we've taken two chunks; the stream would otherwise keep going.
    stop_after = 2

    def should_stop_now():
        return len(seen) >= stop_after

    def chunks():
        for i in range(100):  # a stream that does NOT end on its own
            seen.append(i)
            yield b"x" * (600 * 1024)  # > the 512 KB buffer, so writes happen

    rec = _make_recording_recorder(tmp_path, chunks(), should_stop_now)

    with patch("core.tiktok_recorder.VideoManagement") as vm:
        rec.start_recording("tester", "room-1")

    # It stopped promptly — it did not drain the whole stream.
    assert len(seen) < 10
    # The recording was finalized, not abandoned: converted exactly once.
    assert vm.convert_flv_to_mp4.call_count == 1
    # And the partial recording is on disk with real bytes in it.
    written = list(tmp_path.glob("TK_tester_*_flv.mp4"))
    assert len(written) == 1
    assert written[0].stat().st_size > 0


def test_graceful_stop_does_not_interrupt_an_in_flight_recording(tmp_path):
    """The graceful flag must NOT reach into the download loop — that's the whole
    distinction. Only should_stop_now() interrupts; should_stop() is read at the
    poll boundary, so a graceful removal lets the broadcast finish."""
    chunk_count = {"n": 0}

    def chunks():
        for _ in range(5):  # a stream that ends on its own
            chunk_count["n"] += 1
            yield b"x" * (600 * 1024)

    rec = _make_recording_recorder(tmp_path, chunks(), should_stop_now=lambda: False)
    rec.should_stop = lambda: True  # graceful stop is pending the whole time
    # After the stream drains, is_room_alive goes False so the loop ends.
    rec.tiktok.is_room_alive.side_effect = [True, False]

    with patch("core.tiktok_recorder.VideoManagement") as vm:
        rec.start_recording("tester", "room-1")

    assert chunk_count["n"] == 5  # the full broadcast was recorded
    assert vm.convert_flv_to_mp4.call_count == 1
