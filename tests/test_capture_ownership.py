"""
§104 — the monitor must notice when its capture is taken, and must not hold a
silent room forever.

2026-10-01, a watched account: the broadcast ended but TikTok's room/info kept saying
"live" for 5.8 minutes, during which the stream gave nothing and the capture's
mtime stood still. tiktak's ingest sweep reads "no write for 300 s" as "the
writer is gone", remuxed the file, archived it (777 MB, fine) and unlinked it —
19 s before the monitor, still holding the file open, finished. Then:

  * `VideoManagement.wait_for_file_release()` opens the path with mode "ab",
    which CREATES a missing file. That is where the 0-byte stub came from, and
    ffmpeg then failed with "moov atom not found" on it.
  * Had the stream resumed instead, every later byte would have gone to the
    unlinked file, silently.

So the monitor checks, at each pass top and each flush, that the path still names
the file it holds open. If not it ends the recording (the capture was archived by
someone else), skips the convert, and the immediate re-poll starts a NEW file.
The sweep stays the single authority for "idle 300 s"; nothing changes in tiktak.

Separately, a bound: 15 minutes with no bytes ends the recording. Without it a
room that stays "live" forever would hold its capture forever with a healthy
watchdog promise (the §39 phantom). The sweep cannot help there — a 0-byte
capture is "not FLV", and it leaves it alone.
"""

import json
from unittest.mock import Mock, patch

from core.tiktok_recorder import TikTokRecorder
from utils.video_management import VideoManagement


class _Clock:
    """Passes cost no time; only the backoff sleeps move this clock."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _recorder(tmp_path, *, stream, alive):
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "tester"
    rec.duration = None
    rec.output = str(tmp_path)
    rec.bitrate = None
    rec.ffmpeg_path = None
    rec.mode = None
    rec._stop_now_event = None
    rec._deadline = None
    rec.should_stop_now = lambda: False
    rec.tiktok = Mock()
    rec.tiktok.get_live_url.return_value = "http://cdn/x.flv"
    rec.tiktok.last_room_created_at = None
    rec.tiktok.last_stream_status = 200
    rec.tiktok.last_stream_length = None
    rec.tiktok.is_room_alive.side_effect = alive
    rec.tiktok.download_live_stream.side_effect = stream
    return rec


def _run(rec, clock=None):
    """Run one recording; return (convert mock, log lines)."""
    clock = clock or _Clock()
    lines = []
    log = Mock()
    for level in ("info", "warning", "error"):
        getattr(log, level).side_effect = lambda msg, *a, **k: lines.append(str(msg))
    with patch("core.tiktok_recorder.VideoManagement") as vm, patch(
        "core.tiktok_recorder.time.sleep", clock.sleep
    ), patch("core.tiktok_recorder.time.monotonic", clock.monotonic), patch(
        "core.tiktok_recorder.logger", log
    ):
        rec.start_recording("tester", "room-1")
    return vm.convert_flv_to_mp4, lines


def _captures(tmp_path):
    return sorted(tmp_path.glob("TK_tester_*_flv.mp4"))


def _unlink_capture(tmp_path):
    for p in _captures(tmp_path):
        p.unlink()


# --- (e) the monitor notices the sweep took its capture -----------------------


def test_a_capture_adopted_while_open_ends_the_recording_without_a_convert(tmp_path):
    def stream(url):
        yield b"a" * 100
        _unlink_capture(tmp_path)      # the sweep remuxes it and unlinks it

    rec = _recorder(tmp_path, stream=stream, alive=[True, True, True])
    convert, lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 1   # no second pass
    assert convert.call_count == 0
    assert any("adopted while open" in l for l in lines)
    assert _captures(tmp_path) == []                          # no stub re-created


def test_adoption_is_noticed_at_the_flush_too_not_only_at_the_pass_top(tmp_path):
    consumed = []

    def stream(url):
        yield b"x" * 10
        _unlink_capture(tmp_path)
        for i in range(3):
            consumed.append(i)
            yield b"y" * (300 * 1024)   # the second of these crosses the 512 KB flush

    rec = _recorder(tmp_path, stream=stream, alive=[True, True, True])
    convert, lines = _run(rec)

    assert consumed == [0, 1]           # the flush saw it; the third chunk was never read
    assert convert.call_count == 0
    assert any("adopted while open" in l for l in lines)


def test_a_capture_renamed_to_needs_review_is_detected_like_an_unlink(tmp_path):
    """The sweep's other way to take a capture: `src.rename(".needs-review")`."""
    def stream(url):
        yield b"a" * 100
        for p in _captures(tmp_path):
            p.rename(p.with_name(p.name + ".needs-review"))

    rec = _recorder(tmp_path, stream=stream, alive=[True, True, True])
    convert, lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 1
    assert convert.call_count == 0
    assert any("adopted while open" in l for l in lines)


def test_adoption_is_noticed_before_any_request_in_the_next_pass(tmp_path):
    """A monitor in a long backoff must see it without one more room/info call."""
    def stream(url):
        _unlink_capture(tmp_path)    # nothing buffered: only the pass top can notice
        return iter(())

    rec = _recorder(tmp_path, stream=stream, alive=[True, True, True])
    _, lines = _run(rec)

    assert rec.tiktok.is_room_alive.call_count == 1
    assert any("0 buffered bytes" in l for l in lines)


def test_bytes_still_buffered_when_a_pass_ends_are_checked_and_counted(tmp_path):
    """The pass-end flush wrote to a taken file unchecked, and the next pass top
    then claimed "0 buffered bytes are lost" — a number it never measured."""
    def stream(url):
        yield b"a" * 100                # under 512 KB: stays in the buffer
        _unlink_capture(tmp_path)

    rec = _recorder(tmp_path, stream=stream, alive=[True, True, True])
    _, lines = _run(rec)

    [warning] = [l for l in lines if "adopted while open" in l]
    assert "100 buffered bytes" in warning
    assert rec.tiktok.is_room_alive.call_count == 1


def test_a_capture_taken_between_the_checks_at_the_end_does_not_raise(tmp_path):
    """exists() then getsize() are two calls; the sweep can act between them."""
    rec = _recorder(tmp_path, stream=lambda url: iter(()), alive=[False])
    with patch("core.tiktok_recorder.os.path.getsize", side_effect=FileNotFoundError):
        convert, lines = _run(rec)

    assert convert.call_count == 0
    assert any("already" in l and "adopted" in l for l in lines)


def test_the_warning_for_a_capture_taken_at_a_flush_counts_the_lost_bytes(tmp_path):
    def stream(url):
        _unlink_capture(tmp_path)
        yield b"y" * (600 * 1024)

    rec = _recorder(tmp_path, stream=stream, alive=[True, True])
    _, lines = _run(rec)

    [warning] = [l for l in lines if "adopted while open" in l]
    assert f"{600 * 1024} buffered bytes" in warning


def test_the_next_recording_after_an_adoption_gets_a_new_file(tmp_path):
    def first(url):
        yield b"OLD" * 50
        _unlink_capture(tmp_path)

    _run(_recorder(tmp_path, stream=first, alive=[True, True]))

    second = _recorder(tmp_path, stream=lambda url: iter([b"NEW" * 50]),
                       alive=[True, False])
    _run(second)

    [capture] = _captures(tmp_path)
    assert capture.read_bytes() == b"NEW" * 50


def test_a_capture_taken_during_the_last_check_is_not_converted_or_recreated(tmp_path):
    """The sweep can win between the pass-top check and the room check."""
    def room_ends_after_the_sweep(_room):
        _unlink_capture(tmp_path)
        return False

    rec = _recorder(tmp_path, stream=lambda url: iter(()),
                    alive=room_ends_after_the_sweep)
    convert, lines = _run(rec)

    assert convert.call_count == 0
    assert _captures(tmp_path) == []
    assert any("already" in l and "adopted" in l for l in lines)


def test_convert_never_creates_the_file_it_was_asked_to_convert(tmp_path):
    """`open(path, "ab")` creates a missing file: the 0-byte stub of 2026-10-01."""
    missing = tmp_path / "TK_tester_2026.10.01_21-33-20_flv.mp4"
    with patch("utils.video_management.ffmpeg") as ff:
        VideoManagement.convert_flv_to_mp4(str(missing))
    assert not missing.exists()
    assert ff.input.call_count == 0


# --- C: a 0-byte capture leaves nothing behind --------------------------------


def test_a_zero_byte_capture_is_deleted_with_its_sidecar_and_not_converted(tmp_path):
    rec = _recorder(tmp_path, stream=lambda url: iter(()), alive=[False])
    convert, lines = _run(rec)

    assert convert.call_count == 0
    assert _captures(tmp_path) == []
    assert list(tmp_path.glob("*.room.json")) == []
    assert any("no bytes" in l for l in lines)


def test_a_capture_with_bytes_is_still_converted_and_keeps_its_sidecar(tmp_path):
    rec = _recorder(tmp_path, stream=lambda url: iter([b"x" * 50]),
                    alive=[True, False])
    convert, _ = _run(rec)

    assert convert.call_count == 1
    [sidecar] = tmp_path.glob("*.room.json")
    assert json.loads(sidecar.read_text())["room_id"] == "room-1"


# --- D: fifteen minutes of silence is a bound, not a pause --------------------


def _forever(limit=200):
    """is_room_alive that is always True, but fails the test if it is asked too often."""
    calls = {"n": 0}

    def alive(_room):
        calls["n"] += 1
        assert calls["n"] <= limit, "the silence floor never ended the recording"
        return True
    return alive, calls


def test_fifteen_minutes_of_silence_ends_a_zero_byte_recording_and_leaves_no_trace(tmp_path):
    alive, _ = _forever()
    clock = _Clock()
    start = clock.t
    rec = _recorder(tmp_path, stream=lambda url: iter(()), alive=alive)
    convert, lines = _run(rec, clock)

    assert 900 <= clock.t - start < 900 + 100     # the floor, within one more backoff
    assert convert.call_count == 0
    assert _captures(tmp_path) == []
    assert list(tmp_path.glob("*.room.json")) == []
    assert sum("silent" in l for l in lines) == 1  # one WARNING, not one per pass


def test_silence_after_bytes_ends_the_recording_and_keeps_what_was_written(tmp_path):
    alive, _ = _forever()
    passes = iter([[b"A" * 1000]])
    rec = _recorder(tmp_path, stream=lambda url: iter(next(passes, [])), alive=alive)
    convert, lines = _run(rec)

    assert convert.call_count == 1
    [capture] = _captures(tmp_path)
    assert capture.read_bytes() == b"A" * 1000
    assert any("silent" in l for l in lines)


def test_bytes_returning_after_four_minutes_continue_the_same_file_intact(tmp_path):
    """Four minutes is below every bound: the sweep's 300 s and the 15 min floor."""
    clock = _Clock()
    start = clock.t
    passes = iter([[b"A" * 1000]] + [[]] * 12 + [[b"B" * 1000]])
    rec = _recorder(tmp_path, stream=lambda url: iter(next(passes)),
                    alive=[True] * 14 + [False])
    convert, lines = _run(rec, clock)

    assert clock.t - start >= 240                 # the silence really lasted 4 min
    assert convert.call_count == 1
    [capture] = _captures(tmp_path)
    assert capture.read_bytes() == b"A" * 1000 + b"B" * 1000
    assert not any("silent" in l for l in lines)
