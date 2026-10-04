"""
§104 — a stream pass that ends with zero bytes must be visible and must back off.

Found 2026-10-04 while chasing a watched account's "0-byte recording" of 2026-10-01
(it turned out to be archived fine). The finding was in the loop: when
`download_live_stream()` ends cleanly with nothing, `start_recording()` re-enters
at network speed — about 3-4 s per pass, ~100 `room/info` calls — and logs nothing
at all. That recording's tail was 5.8 minutes of it, and the log read "Started
recording" then "User is no longer live" with no line between (the 88 minutes
before the tail were one healthy pass). Nothing could say whether a stream was
healthy, empty, or looping.

The fix is evidence and politeness, not a new ending rule:

1. Every pass end gets one line: bytes, seconds, HTTP status, Content-Length.
   A long empty run is rate-limited (first, then every 10th) so it cannot flood.
2. A zero-byte pass backs off 2, 4, 8, 16, 30, 30 … s. A pass that wrote bytes
   reconnects at once, as before, and resets the backoff — a healthy stream that
   the CDN recycles every minute must not slow down.
"""

from unittest.mock import Mock, patch

import pytest

from core.tiktok_api import TikTokAPI
from core.tiktok_recorder import TikTokRecorder


def _recorder(tmp_path, passes):
    """A recorder whose Nth stream pass yields `passes[N]`, then the room ends."""
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
    rec.tiktok.last_stream_status = None
    rec.tiktok.last_stream_length = None
    script = iter(passes)
    rec.tiktok.is_room_alive.side_effect = [True] * len(passes) + [False]

    def answer(url):
        # Like the real download_live_stream: status is known once get() returns.
        rec.tiktok.last_stream_status = 200
        rec.tiktok.last_stream_length = "0"
        return iter(next(script))

    rec.tiktok.download_live_stream.side_effect = answer
    return rec


def _run(rec):
    """Run the loop; return (sleeps, log lines)."""
    sleeps, lines = [], []
    log = Mock()
    for level in ("info", "warning", "error"):
        getattr(log, level).side_effect = lambda msg, *a, **k: lines.append(str(msg))
    with patch("core.tiktok_recorder.VideoManagement"), patch(
        "core.tiktok_recorder.time.sleep", lambda s: sleeps.append(s)
    ), patch("core.tiktok_recorder.logger", log):
        rec.start_recording("tester", "room-1")
    return sleeps, lines


def _pass_lines(lines):
    return [l for l in lines if "Stream pass ended" in l]


def test_empty_passes_back_off_and_the_backoff_is_capped(tmp_path):
    sleeps, _ = _run(_recorder(tmp_path, [[]] * 7))
    assert sleeps == [2, 4, 8, 16, 30, 30, 30]


def test_a_pass_with_bytes_reconnects_at_once_and_resets_the_backoff(tmp_path):
    passes = [[], [], [b"x" * 100], [b"y" * 100], []]
    sleeps, _ = _run(_recorder(tmp_path, passes))
    # empty, empty -> 2, 4; two byte passes -> no sleep; the next empty restarts at 2
    assert sleeps == [2, 4, 2]


def test_every_pass_end_says_bytes_seconds_status_and_content_length(tmp_path):
    _, lines = _run(_recorder(tmp_path, [[b"a" * 300], []]))
    first, second = _pass_lines(lines)
    assert "300 bytes" in first and "HTTP 200" in first and "Content-Length 0" in first
    assert "0 bytes" in second


def test_a_long_empty_run_logs_the_first_and_every_tenth_pass_only(tmp_path):
    _, lines = _run(_recorder(tmp_path, [[]] * 25))
    assert len(_pass_lines(lines)) == 3  # passes 1, 10, 20


def test_bytes_after_a_long_empty_run_continue_the_same_file_intact(tmp_path):
    """The recording must survive silence: same capture, nothing lost or reordered."""
    passes = [[b"A" * 1000]] + [[]] * 6 + [[b"B" * 1000]]
    rec = _recorder(tmp_path, passes)
    _run(rec)
    [capture] = tmp_path.glob("TK_tester_*_flv.mp4")
    assert capture.read_bytes() == b"A" * 1000 + b"B" * 1000


def _raising_recorder(tmp_path, exc, passes):
    """Every stream pass raises `exc`; the room ends after `passes` of them."""
    rec = _recorder(tmp_path, [])
    rec.tiktok.is_room_alive.side_effect = [True] * passes + [False]
    rec.tiktok.download_live_stream.side_effect = exc
    return rec


def test_a_pass_that_raises_builtin_connection_error_logs_and_backs_off(tmp_path):
    """Prod runs Mode.MANUAL (`-mode` is absent from its command line), so the
    builtin-ConnectionError branch is silent and sleeps nothing. Left alone it is
    a tight silent loop; the pass report and backoff must sit where it reaches."""
    sleeps, lines = _run(_raising_recorder(tmp_path, ConnectionResetError("reset"), 5))
    assert sleeps == [2, 4, 8, 16, 30]
    [first] = _pass_lines(lines)
    assert "0 bytes" in first and "ConnectionResetError" in first


def test_a_pass_that_raises_a_network_error_is_reported_too(tmp_path):
    import requests
    _, lines = _run(_raising_recorder(
        tmp_path, requests.exceptions.ConnectionError("Read timed out."), 2))
    [first] = _pass_lines(lines)
    assert "ConnectionError" in first


def test_bytes_then_an_exception_do_not_count_as_an_empty_pass(tmp_path):
    def half():
        yield b"x" * 100
        raise ConnectionResetError("reset")

    rec = _raising_recorder(tmp_path, lambda url: half(), 1)
    sleeps, lines = _run(rec)
    assert sleeps == []
    assert "100 bytes" in _pass_lines(lines)[0]


def test_a_pass_that_got_no_response_does_not_report_the_previous_status(tmp_path):
    """Evidence must not lie: pass 2 never got an answer, so it cannot say HTTP 200."""
    rec = _recorder(tmp_path, [[b"x" * 10]])
    answer = rec.tiktok.download_live_stream.side_effect
    calls = {"n": 0}

    def second_connect_fails(url):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionResetError("connect failed")   # before any response
        return answer(url)

    rec.tiktok.download_live_stream.side_effect = second_connect_fails
    rec.tiktok.is_room_alive.side_effect = [True, True, False]
    _, lines = _run(rec)

    first, second = _pass_lines(lines)
    assert "HTTP 200" in first
    assert "HTTP 200" not in second and "HTTP None" in second


def test_a_403_with_a_body_is_an_empty_pass_and_writes_nothing_to_the_capture(tmp_path):
    """The adverse condition: a CDN that answers the tail with an error PAGE.

    `iter_content` yields that body whatever the status, so without this it counts
    as "bytes": no backoff, a log line per pass, and HTML appended to the FLV.
    """
    api = TikTokAPI.__new__(TikTokAPI)
    response = Mock(status_code=403, headers={"Content-Length": "19"})
    response.iter_content.return_value = iter([b"<html>denied</html>"])
    api._http_client_stream = Mock()
    api._http_client_stream.get.return_value = response
    api.get_live_url = Mock(return_value="http://cdn/x.flv")
    api.last_room_created_at = None
    api.is_room_alive = Mock(side_effect=[True, True, True, False])

    rec = _recorder(tmp_path, [])
    rec.tiktok = api
    sleeps, lines = _run(rec)

    assert sleeps == [2, 4, 8]
    assert "HTTP 403" in _pass_lines(lines)[0]
    assert response.close.called
    [capture] = tmp_path.glob("TK_tester_*_flv.mp4")
    assert capture.read_bytes() == b""                        # no error page in the FLV


def test_a_force_stop_during_an_empty_run_ends_it_after_one_pass(tmp_path):
    """`should_stop_now()` was read only inside the chunk loop, so an empty run
    never saw it — and the event's wait() returns at once, so the backoff spun."""
    rec = _recorder(tmp_path, [[]] * 5)
    rec.should_stop_now = lambda: True
    sleeps, lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 1
    assert sleeps == []
    assert any("Stop requested" in l for l in lines)


def test_download_live_stream_closes_and_yields_nothing_for_an_error_status():
    api = TikTokAPI.__new__(TikTokAPI)
    response = Mock(status_code=404, headers={})
    response.iter_content.return_value = iter([b"not found"])
    api._http_client_stream = Mock()
    api._http_client_stream.get.return_value = response

    assert list(api.download_live_stream("http://cdn/x.flv")) == []

    assert response.close.called
    assert api.last_stream_status == 404


def test_download_live_stream_exposes_status_and_content_length():
    api = TikTokAPI.__new__(TikTokAPI)
    response = Mock(status_code=403, headers={"Content-Length": "0"})
    response.iter_content.return_value = iter(())
    api._http_client_stream = Mock()
    api._http_client_stream.get.return_value = response

    assert list(api.download_live_stream("http://cdn/x.flv")) == []

    assert api.last_stream_status == 403
    assert api.last_stream_length == "0"


def test_a_missing_content_length_is_reported_as_unknown():
    api = TikTokAPI.__new__(TikTokAPI)
    response = Mock(status_code=200, headers={})
    response.iter_content.return_value = iter([b"x"])
    api._http_client_stream = Mock()
    api._http_client_stream.get.return_value = response

    list(api.download_live_stream("http://cdn/x.flv"))

    assert api.last_stream_length is None
