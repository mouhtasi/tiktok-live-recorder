"""
The room sidecar: how late did we join this broadcast?

On 2026-09-24 @yumehime555 went live at 04:51 and the recording began at 05:26 —
her monitor was on tiktak's hourly cold tier. Nothing recorded that 35-minute
loss; the user noticed it by watching the stream. The recording's own start time
is in its filename, but the broadcast's start was not kept anywhere.

`webcast/room/info`, which `get_live_url()` already fetches right before every
recording, carries the room's `create_time`. So the fork keeps it and writes it
beside the capture as `TK_<user>_<ts>.room.json`, for tiktak's ingest to read.
No extra request: an extra request would muddy the §93 poll-rate trial it ships
alongside.

🚨 The sidecar is bookkeeping. A failure to write it must never cost the
recording it describes.
"""

import json
from unittest.mock import Mock, patch

from core.tiktok_api import TikTokAPI
from core.tiktok_recorder import TikTokRecorder

ROOM_ID = "7689020295423806221"   # @yumehime555's room, 2026-09-24
CREATED = 1790239860              # its create_time: 04:51:00 EDT


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


def _live_payload(create_time=CREATED):
    data = {"status": 2, "stream_url": {"flv_pull_url": {"HD1": "http://cdn/x.flv"}}}
    if create_time is not None:
        data["create_time"] = create_time
    return {"status_code": 0, "data": data}


# --- the API keeps create_time from the response it already fetched ----------


def test_get_live_url_keeps_the_rooms_create_time():
    api = _api(_live_payload())
    assert api.get_live_url(ROOM_ID) == "http://cdn/x.flv"
    assert api.last_room_created_at == CREATED
    assert api.http_client.get.call_count == 1   # no extra request


def test_a_response_without_create_time_clears_the_previous_value():
    """A stale value from the last broadcast would be read as this one's start."""
    api = _api(_live_payload())
    api.get_live_url(ROOM_ID)
    api.http_client.get.return_value = _Response(_live_payload(create_time=None))
    api.get_live_url(ROOM_ID)
    assert api.last_room_created_at is None


# --- the recorder writes it beside the capture --------------------------------


def _recorder(tmp_path, created):
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "tester"
    rec.duration = None
    rec.output = str(tmp_path)
    rec.bitrate = None
    rec.ffmpeg_path = None
    rec.mode = None
    rec._stop_now_event = None
    rec.should_stop_now = lambda: False
    rec.tiktok = Mock()
    rec.tiktok.get_live_url.return_value = "http://cdn/x.flv"
    rec.tiktok.last_room_created_at = created
    rec.tiktok.is_room_alive.side_effect = [True, False]
    rec.tiktok.download_live_stream.return_value = iter([b"x" * 1024])
    return rec


def test_recording_writes_a_sidecar_named_for_the_final_mp4(tmp_path):
    rec = _recorder(tmp_path, CREATED)
    with patch("core.tiktok_recorder.VideoManagement"):
        rec.start_recording("tester", ROOM_ID)

    [capture] = tmp_path.glob("TK_tester_*_flv.mp4")
    sidecar = tmp_path / capture.name.replace("_flv.mp4", ".room.json")
    body = json.loads(sidecar.read_text())
    assert body["room_id"] == ROOM_ID
    assert body["room_created_at"] == CREATED
    assert isinstance(body["joined_at"], int)
    # Nothing half-written is left beside it.
    assert not list(tmp_path.glob("*.tmp"))


def test_an_unknown_create_time_is_written_as_null_not_dropped(tmp_path):
    """The WAF fallback path has no room/info data. "Unknown" must reach ingest
    as unknown, so the page cannot render it as a zero-minute join."""
    rec = _recorder(tmp_path, None)
    with patch("core.tiktok_recorder.VideoManagement"):
        rec.start_recording("tester", ROOM_ID)
    [sidecar] = tmp_path.glob("TK_tester_*.room.json")
    assert json.loads(sidecar.read_text())["room_created_at"] is None


def test_a_non_integer_create_time_is_treated_as_unknown(tmp_path):
    """A Mock, a string, a bool: none of them is a timestamp."""
    rec = _recorder(tmp_path, Mock())
    with patch("core.tiktok_recorder.VideoManagement"):
        rec.start_recording("tester", ROOM_ID)
    [sidecar] = tmp_path.glob("TK_tester_*.room.json")
    assert json.loads(sidecar.read_text())["room_created_at"] is None


def test_a_sidecar_that_cannot_be_written_does_not_stop_the_recording(tmp_path):
    rec = _recorder(tmp_path, CREATED)
    with patch("core.tiktok_recorder.VideoManagement") as vm, patch(
        "core.tiktok_recorder.os.replace", side_effect=OSError("disk full")
    ):
        rec.start_recording("tester", ROOM_ID)

    [capture] = tmp_path.glob("TK_tester_*_flv.mp4")
    assert capture.stat().st_size > 0
    assert vm.convert_flv_to_mp4.call_count == 1
    assert not list(tmp_path.glob("*.room.json*"))
