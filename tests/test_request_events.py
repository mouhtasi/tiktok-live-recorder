"""
Regression tests for ``TikTokAPI``'s request-event instrumentation.

## Why this exists

tiktak (the downstream project) could only ever see the *scraper's* half of its
TikTok request volume in ``request_log`` — the recorder's liveness polling was
invisible, despite generating far more requests (every automatic-mode poll
makes at least two direct TikTok calls: room-id resolution and ``is_room_alive``).
This closes that gap, but the fork must stay decoupled from tiktak's database —
so it only ever appends a JSON line per request to an optional file path handed
in via ``-events-file``. tiktak drains that file on its own schedule.

Two things this must guarantee, because a diagnostics feature that breaks the
thing it measures is worse than no diagnostics at all (the exact shape of the
§54.1 challenge-jar bug elsewhere in this project's history):

1. With no ``events_file`` configured (the default), behavior is byte-for-byte
   identical to before this patch — every existing test in this suite must
   still pass unmodified.
2. A failure to WRITE the event (bad path, disk full, permissions) must never
   propagate and must never suppress or alter the real request's result.
"""

import json
from unittest.mock import Mock

import pytest

from core.tiktok_api import TikTokAPI


class _Response:
    def __init__(self, *, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _make_api(*, events_file=None, user="tester"):
    api = TikTokAPI.__new__(TikTokAPI)
    api.BASE_URL = "https://www.tiktok.com"
    api.WEBCAST_URL = "https://webcast.tiktok.com"
    api.API_URL = "https://www.tiktok.com/api-live/user/room/"
    api.TIKREC_API = "https://tikrec.com"
    api._events_file = events_file
    api._user = user
    api.http_client = Mock()
    return api


def _read_events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- disabled by default: no behavior change, nothing written ----------------


def test_no_events_file_writes_nothing(tmp_path):
    api = _make_api(events_file=None)
    api.http_client.get.return_value = _Response(status_code=200)

    response = api._get("https://webcast.tiktok.com/x", "webcast/room/info")

    assert response.status_code == 200
    assert not (tmp_path / "events.jsonl").exists()


def test_missing_events_file_attribute_is_safe():
    """__new__-bypassed instances (as every other test in this suite builds)
    must not AttributeError just because _events_file was never set."""
    api = TikTokAPI.__new__(TikTokAPI)
    api.http_client = Mock()
    api.http_client.get.return_value = _Response(status_code=200)

    response = api._get("https://webcast.tiktok.com/x", "webcast/room/info")

    assert response.status_code == 200


# --- enabled: one line per call, correct shape --------------------------------


def test_successful_call_writes_one_event_line(tmp_path):
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file), user="lo3tus")
    api.http_client.get.return_value = _Response(status_code=200)

    api._get("https://webcast.tiktok.com/webcast/room/info/?room_id=1", "webcast/room/info")

    events = _read_events(events_file)
    assert len(events) == 1
    ev = events[0]
    assert ev["subsystem"] == "recorder-liveness"
    assert ev["endpoint"] == "webcast/room/info"
    assert ev["status"] == 200
    assert ev["username"] == "lo3tus"
    assert isinstance(ev["latency_ms"], int) and ev["latency_ms"] >= 0
    assert "ts" in ev


def test_subsystem_override_for_tikrec_calls(tmp_path):
    """tikrec.com is a third-party signer, not TikTok — it must not be counted
    under the same subsystem tiktak uses to reason about TikTok's own request
    volume, or a tikrec slowdown would look like a TikTok one."""
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))
    api.http_client.get.return_value = _Response(status_code=200)

    api._get(
        "https://tikrec.com/tiktok/room/api/sign",
        "tikrec/sign",
        subsystem="recorder-tikrec",
    )

    events = _read_events(events_file)
    assert events[0]["subsystem"] == "recorder-tikrec"


def test_endpoint_label_is_fixed_not_url_derived(tmp_path):
    """The endpoint column must stay low-cardinality (request_log is grouped by
    it) — never derive it from a URL, which carries a room_id or signature."""
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))
    api.http_client.get.return_value = _Response(status_code=200)

    api._get(
        "https://webcast.tiktok.com/webcast/room/info/?aid=1988&room_id=7489823057092168456",
        "webcast/room/info",
    )

    assert _read_events(events_file)[0]["endpoint"] == "webcast/room/info"


# --- a failed call is still recorded, with status=None ------------------------


def test_call_that_raises_still_emits_an_event_with_null_status(tmp_path):
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))
    api.http_client.get.side_effect = OSError("connection refused")

    with pytest.raises(OSError):
        api._get("https://webcast.tiktok.com/x", "webcast/room/info")

    events = _read_events(events_file)
    assert len(events) == 1
    assert events[0]["status"] is None


# --- the instrument must never break the call it measures ---------------------


def test_unwritable_events_path_does_not_break_the_real_call(tmp_path):
    # A directory that doesn't exist — open() for append will raise.
    bad_path = tmp_path / "does" / "not" / "exist" / "events.jsonl"
    api = _make_api(events_file=str(bad_path))
    api.http_client.get.return_value = _Response(status_code=200, payload={"ok": True})

    response = api._get("https://webcast.tiktok.com/x", "webcast/room/info")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_unwritable_events_path_does_not_mask_the_real_exception(tmp_path):
    bad_path = tmp_path / "does" / "not" / "exist" / "events.jsonl"
    api = _make_api(events_file=str(bad_path))
    api.http_client.get.side_effect = OSError("connection refused")

    with pytest.raises(OSError):
        api._get("https://webcast.tiktok.com/x", "webcast/room/info")


# --- real call sites route through _get with the right label ------------------


def test_is_room_alive_uses_recorder_liveness_subsystem(tmp_path):
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))
    api.http_client.get.return_value = _Response(payload={"data": {"status": 2}})

    assert api.is_room_alive("123") is True

    events = _read_events(events_file)
    assert len(events) == 1
    assert events[0] == {**events[0], "subsystem": "recorder-liveness", "endpoint": "webcast/room/info"}


def test_get_live_url_uses_recorder_liveness_subsystem(tmp_path):
    """get_live_url() hits the same webcast/room/info endpoint as
    is_room_alive() — it fires once per broadcast start, at the highest-load
    moment, and must be tagged the same way or that endpoint's total looks
    complete while quietly missing exactly its busiest calls."""
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))
    api.http_client.get.return_value = _Response(payload={
        "data": {"stream_url": {}},
    })

    api.get_live_url("123", user="tester")

    events = _read_events(events_file)
    assert len(events) == 1
    assert events[0]["subsystem"] == "recorder-liveness"
    assert events[0]["endpoint"] == "webcast/room/info"


def test_room_id_resolution_via_tikrec_splits_subsystems(tmp_path):
    """The happy path makes two calls: tikrec's signer (recorder-tikrec) and
    the signed fetch, which still lands on tiktok.com (recorder-liveness)."""
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))

    def _route(url, **kwargs):
        if "tikrec.com" in url:
            return _Response(payload={"signed_path": "/api-live/user/room/?signed=1"})
        return _Response(payload={"data": {"user": {"roomId": "999"}}})

    api.http_client.get.side_effect = _route

    assert api.get_room_id_from_user("tester") == "999"

    events = _read_events(events_file)
    assert [e["subsystem"] for e in events] == ["recorder-tikrec", "recorder-liveness"]
    assert [e["endpoint"] for e in events] == ["tikrec/sign", "tiktok/room-signed"]


def test_room_id_resolution_direct_fallback_uses_recorder_liveness(tmp_path):
    events_file = tmp_path / "events.jsonl"
    api = _make_api(events_file=str(events_file))

    def _route(url, **kwargs):
        if "tikrec.com" in url:
            return _Response(status_code=522, text="<html>error</html>", payload=None)
        return _Response(payload={"data": {"user": {"roomId": "999"}}})

    def _route_with_bad_json(url, **kwargs):
        if "tikrec.com" in url:
            r = Mock()
            r.raise_for_status = Mock()
            r.json = Mock(side_effect=ValueError("not json"))
            r.status_code = 522
            return r
        return _Response(payload={"data": {"user": {"roomId": "999"}}})

    api.http_client.get.side_effect = _route_with_bad_json

    assert api.get_room_id_from_user("tester") == "999"

    events = _read_events(events_file)
    labels = [(e["subsystem"], e["endpoint"]) for e in events]
    assert ("recorder-tikrec", "tikrec/sign") in labels
    assert ("recorder-liveness", "tiktok/api-live-room") in labels
