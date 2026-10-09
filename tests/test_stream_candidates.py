"""
More than one stream URL for a broadcast.

room/info lists several renditions, and for each a `main` and a `backup` CDN.
`get_live_url()` took the best level's `main` FLV and nothing else, so a URL
that gave no data had no alternative: the loop asked the same URL again until
the room ended. Upstream tries the next URL (898c232, "try alternate live stream
urls"); this is that idea, held to two rules of ours:

* **FLV only.** Upstream also offers HLS, written as `.ts`. tiktak's ingest
  knows `_flv.mp4` captures and nothing else.
* **A capture never mixes two renditions.** A reconnect to the same URL already
  appends a second FLV to the file, and the remux copes. A second rendition can
  change the codec in the middle of a file. So the loop moves to the next URL
  only while the capture is still empty.

🚨 How often this can act is small: prod logged 3 empty stream passes in the
four days after §104 began to log them (2026-10-05 to 10-09).

Also pinned: the audio-only entry (`ao`, no level) is never a candidate. A
"recording" of it would be a live with no picture.
"""

import json
from unittest.mock import Mock, patch

from core.tiktok_api import TikTokAPI
from core.tiktok_recorder import TikTokRecorder

ROOM_ID = "7660702476433820429"


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _api(payload):
    api = TikTokAPI.__new__(TikTokAPI)
    api.WEBCAST_URL = "https://webcast.tiktok.com"
    api.http_client = Mock()
    api.http_client.get.return_value = _Response(payload)
    return api


def _sdk_payload(entries, qualities):
    return {"status_code": 0, "data": {"status": 2, "stream_url": {
        "live_core_sdk_data": {"pull_data": {
            "stream_data": json.dumps({"data": entries}),
            "options": {"qualities": qualities},
        }},
    }}}


QUALITIES = [
    {"sdk_key": "ld", "level": 1, "name": "360p"},
    {"sdk_key": "hd", "level": 3, "name": "720p"},
]
# The shape upstream issue #456 printed for a real room, 2026-07-15.
ENTRIES = {
    "ld": {"main": {"flv": "http://cdn/ld.flv"}, "backup": {"flv": "http://cdn-b/ld.flv"}},
    "ao": {"main": {"flv": "http://cdn/x.flv?only_audio=1"}},
    "hd": {"main": {"flv": "http://cdn/hd.flv"}, "backup": {"flv": "http://cdn-b/hd.flv"}},
}


# --- the API -----------------------------------------------------------------


def test_candidates_are_best_level_first_main_before_backup():
    api = _api(_sdk_payload(ENTRIES, QUALITIES))
    assert api.get_live_url(ROOM_ID) == "http://cdn/hd.flv"     # unchanged
    assert api.last_stream_candidates == [
        ("720p main", "http://cdn/hd.flv"),
        ("720p backup", "http://cdn-b/hd.flv"),
        ("360p main", "http://cdn/ld.flv"),
        ("360p backup", "http://cdn-b/ld.flv"),
    ]


def test_the_audio_only_entry_is_never_a_candidate():
    api = _api(_sdk_payload(ENTRIES, QUALITIES))
    api.get_live_url(ROOM_ID)
    assert not any("only_audio" in url for _, url in api.last_stream_candidates)


def test_a_best_level_with_no_flv_gives_the_next_one():
    """Before: the best level's missing FLV was returned as None, and the
    recorder raised "unable to retrieve live streaming url" on a room that had
    a 360p stream to give."""
    entries = dict(ENTRIES, hd={"main": {"hls": "http://cdn/hd.m3u8"}})
    api = _api(_sdk_payload(entries, QUALITIES))
    assert api.get_live_url(ROOM_ID) == "http://cdn/ld.flv"


def test_a_backup_that_repeats_main_is_listed_once():
    entries = {"hd": {"main": {"flv": "http://cdn/hd.flv"},
                      "backup": {"flv": "http://cdn/hd.flv"}}}
    api = _api(_sdk_payload(entries, QUALITIES))
    api.get_live_url(ROOM_ID)
    assert api.last_stream_candidates == [("720p main", "http://cdn/hd.flv")]


def test_legacy_urls_keep_their_order():
    payload = {"status_code": 0, "data": {"status": 2, "stream_url": {
        "flv_pull_url": {"SD1": "http://cdn/sd.flv", "HD1": "http://cdn/hd.flv"},
        "rtmp_pull_url": "http://cdn/hd.flv",
    }}}
    api = _api(payload)
    assert api.get_live_url(ROOM_ID) == "http://cdn/hd.flv"
    assert api.last_stream_candidates == [
        ("HD1", "http://cdn/hd.flv"), ("SD1", "http://cdn/sd.flv"),
    ]


def test_candidates_of_the_last_room_never_survive_a_new_call():
    api = _api(_sdk_payload(ENTRIES, QUALITIES))
    api.get_live_url(ROOM_ID)
    api.http_client.get.return_value = _Response(_sdk_payload(ENTRIES, []))
    assert api.get_live_url(ROOM_ID) is None
    assert api.last_stream_candidates == []


# --- the recording loop ------------------------------------------------------


def _recorder(tmp_path, candidates, passes):
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
    rec.tiktok.get_live_url.return_value = candidates[0][1]
    rec.tiktok.last_stream_candidates = candidates
    rec.tiktok.last_room_created_at = None
    rec.tiktok.last_stream_status = None
    rec.tiktok.last_stream_length = None
    rec.tiktok.last_room_status = 4
    rec.tiktok.is_room_alive.side_effect = [True] * len(passes) + [False]
    script = iter(passes)

    def stream(url):
        rec.tiktok.last_stream_status = 200
        return iter(next(script))

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


def _asked(rec):
    return [call.args[0] for call in rec.tiktok.download_live_stream.call_args_list]


TWO = [("720p main", "http://cdn/hd.flv"), ("720p backup", "http://cdn-b/hd.flv")]


def test_an_empty_first_stream_moves_to_the_next_one(tmp_path):
    rec = _recorder(tmp_path, TWO, passes=[[], [b"aaaa"]])
    lines = _run(rec)

    assert _asked(rec) == ["http://cdn/hd.flv", "http://cdn-b/hd.flv"]
    (capture,) = tmp_path.glob("*_flv.mp4")
    assert capture.read_bytes() == b"aaaa"
    assert any("720p backup" in line and "2/2" in line for line in lines)


def test_the_streams_are_tried_in_a_ring_while_the_capture_is_empty(tmp_path):
    rec = _recorder(tmp_path, TWO, passes=[[], [], [b"aaaa"]])
    _run(rec)
    assert _asked(rec) == ["http://cdn/hd.flv", "http://cdn-b/hd.flv", "http://cdn/hd.flv"]


def test_a_capture_that_holds_bytes_stays_on_its_stream(tmp_path):
    """🚨 The rule that keeps one rendition per file. After `aaaa` an empty pass
    is the ordinary end-of-pass reconnect, and it goes to the same URL."""
    rec = _recorder(tmp_path, TWO, passes=[[b"aaaa"], [], [b"bbbb"]])
    _run(rec)
    assert _asked(rec) == ["http://cdn/hd.flv"] * 3


def test_one_stream_is_asked_again_as_before(tmp_path):
    rec = _recorder(tmp_path, TWO[:1], passes=[[], [b"aaaa"]])
    _run(rec)
    assert _asked(rec) == ["http://cdn/hd.flv"] * 2


def test_the_streams_on_offer_are_logged_once_at_the_start(tmp_path):
    rec = _recorder(tmp_path, TWO, passes=[[b"aaaa"]])
    lines = _run(rec)
    offer = [line for line in lines if "on offer" in line]
    assert len(offer) == 1
    assert "720p main" in offer[0] and "720p backup" in offer[0]
