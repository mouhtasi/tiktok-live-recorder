"""
A paused broadcast is not an ended one.

room/info's `status` has three values that matter: 2 live, 3 paused, 4 ended
(upstream Michele0303/tiktok-live-recorder#463). `is_room_alive()` answered
"status == 2", and the recording loop ended the recording on any other answer.
So a creator who paused for a moment ended our file, and the monitor went back
to its poll interval before it looked again.

Measured on prod, 2026-09-24 to 2026-10-09: 3 of 134 broadcasts were recorded as
two files, with 4.2, 5.1 and 5.3 minutes missing between them — one hot-tier
poll each. The log of the newest (@goldzjay, 2026-10-05 06:13) reads "Stream
pass ended", "User is no longer live", and five minutes later "Joined
@goldzjay's broadcast 87 minutes after it started": the same room. 🚨 That log
did not record the status value, so status 3 is the likely cause and not a
proven one; the stop line now names the status so the next case settles it.

What must hold:

* a paused room never STARTS a recording — `is_room_alive()` stays "status 2";
* an open recording stays open through a pause, in the same file;
* every other status still ends it at once (§39: a stale room is status 4);
* a pause cannot hold the loop forever — the §104 silence floor still ends it.
"""

from unittest.mock import Mock, patch

from core.tiktok_api import TikTokAPI
from core.tiktok_recorder import TikTokRecorder

from test_is_room_alive import (
    ROOM_ID, STATUS_ENDED, STATUS_LIVE, WAF_STATUS_CODE, _check_alive, _make_api,
    _room_info,
)

STATUS_PAUSED = 3


# --- the API: a pause is not "alive", and the status is kept -----------------


def test_a_paused_room_does_not_start_a_recording():
    api = _make_api(room_info=_room_info(status=STATUS_PAUSED))
    assert api.is_room_alive(ROOM_ID) is False
    assert api.last_room_status == STATUS_PAUSED


def test_the_status_of_each_answer_is_kept():
    api = _make_api(room_info=_room_info(status=STATUS_LIVE))
    assert api.is_room_alive(ROOM_ID) is True
    assert api.last_room_status == STATUS_LIVE

    api = _make_api(room_info=_room_info(status=STATUS_ENDED))
    assert api.is_room_alive(ROOM_ID) is False
    assert api.last_room_status == STATUS_ENDED


def test_a_waf_answer_never_leaves_the_last_status_in_place():
    """Under the WAF there is no status. A 3 left over from the answer before
    would keep a recording open on a room nobody can see."""
    api = _make_api(
        room_info=_room_info(status_code=WAF_STATUS_CODE),
        check_alive=_check_alive(False),
    )
    api.last_room_status = STATUS_PAUSED
    assert api.is_room_alive(ROOM_ID) is False
    assert api.last_room_status is None


# --- the recording loop ------------------------------------------------------


def _recorder(tmp_path, statuses, passes):
    """A recorder whose Nth liveness check answers `statuses[N]` and whose Nth
    stream pass yields `passes[N]`."""
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
    answers, script = iter(statuses), iter(passes)

    def alive(room_id):
        rec.tiktok.last_room_status = next(answers)
        return rec.tiktok.last_room_status == STATUS_LIVE

    def stream(url):
        rec.tiktok.last_stream_status = 200
        return iter(next(script))

    rec.tiktok.is_room_alive.side_effect = alive
    rec.tiktok.download_live_stream.side_effect = stream
    return rec


def _run(rec):
    lines = []
    log = Mock()
    for level in ("info", "warning", "error"):
        getattr(log, level).side_effect = lambda msg, *a, **k: lines.append(str(msg))
    with patch("core.tiktok_recorder.VideoManagement"), patch(
        "core.tiktok_recorder.time.sleep", lambda s: None
    ), patch("core.tiktok_recorder.logger", log):
        rec.start_recording("tester", "room-1")
    return lines


def _captures(tmp_path):
    return sorted(tmp_path.glob("*_flv.mp4"))


def test_a_pause_keeps_the_recording_open_in_one_file(tmp_path):
    """THE case: live, paused (the stream gives nothing), live again, ended.
    Before the fix the second answer ended the file and `bbbb` was never read."""
    rec = _recorder(
        tmp_path,
        statuses=[STATUS_LIVE, STATUS_PAUSED, STATUS_LIVE, STATUS_ENDED],
        passes=[[b"aaaa"], [], [b"bbbb"]],
    )
    lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 3
    (capture,) = _captures(tmp_path)
    assert capture.read_bytes() == b"aaaabbbb"
    assert sum("paused the broadcast" in line for line in lines) == 1
    assert any("resumed" in line for line in lines)


def test_a_long_pause_is_reported_once(tmp_path):
    rec = _recorder(
        tmp_path,
        statuses=[STATUS_LIVE] + [STATUS_PAUSED] * 4 + [STATUS_ENDED],
        passes=[[b"aaaa"], [], [], [], []],
    )
    lines = _run(rec)
    assert sum("paused the broadcast" in line for line in lines) == 1


def test_an_ended_room_stops_at_once_and_the_line_names_the_status(tmp_path):
    rec = _recorder(tmp_path, statuses=[STATUS_LIVE, STATUS_ENDED], passes=[[b"aaaa"]])
    lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 1
    assert any("no longer live (room status 4)" in line for line in lines)


def test_an_unknown_status_is_not_a_pause(tmp_path):
    """Only 3 keeps a recording open. Status 1, a missing status and the WAF's
    None all end it, as before: §39's phantom recordings came from trusting an
    answer that was not "live"."""
    for status, shown in ((1, "1"), (None, "unknown")):
        rec = _recorder(tmp_path, statuses=[STATUS_LIVE, status], passes=[[b"aaaa"]])
        lines = _run(rec)
        assert rec.tiktok.download_live_stream.call_count == 1
        assert any(f"no longer live (room status {shown})" in line for line in lines)


def test_a_pause_that_never_ends_is_ended_by_the_silence_floor(tmp_path):
    """The loop, not the step: a room that reads "paused" forever must not hold
    the monitor. Nothing new bounds it — §104's silence floor already counts
    empty passes, and a paused pass that gives no bytes is one."""
    def forever():
        while True:
            yield STATUS_PAUSED

    rec = _recorder(tmp_path, statuses=forever(), passes=iter(lambda: [], None))
    with patch.object(TikTokRecorder, "SILENCE_FLOOR_S", 0):
        lines = _run(rec)

    assert rec.tiktok.download_live_stream.call_count == 1
    assert any("silent for" in line for line in lines)


def test_the_api_constant_is_the_value_the_loop_compares():
    assert TikTokAPI.ROOM_PAUSED == STATUS_PAUSED
