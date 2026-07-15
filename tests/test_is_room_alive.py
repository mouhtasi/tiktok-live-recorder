"""
Regression tests for ``TikTokAPI.is_room_alive()`` — the liveness gate.

## Why this exists

``is_room_alive`` decides whether the recorder starts recording. Every watched
account is polled through it; a false ``True`` starts a phantom recording of a
room that isn't broadcasting.

On **2026-07-15 ~06:45 UTC** TikTok changed what ``webcast/room/check_alive``
returns: its ``alive`` flag began answering ``true`` for rooms that had **ended**
(room ``status == 4``), including rooms created *months* earlier. Because
``get_room_id_from_user()`` deliberately returns an offline user's *last*
room_id (liveness is a separate decision), the old ``is_room_alive`` — which
trusted ``check_alive.alive`` alone — reported **every** watched account as live.
The recorder then resolved each stale room's flv URL and wrote keepalive
``{"message":"pong"}`` frames to disk indefinitely; the viewer, which calls a
file "live" while it is being written, lit up the whole roster.

Ground truth captured on prod that day:

    @lulu83245  (genuinely live) -> check_alive true, room status **2**, created today
    @montysmom  (offline months) -> check_alive true, room status **4**, created 2025-04-05
    @sihui.wu5 / @lucky77776 / ... -> check_alive true, room status **4**

So the authoritative liveness signal is room/info ``status`` (2 == ongoing,
4 == ended), **not** ``check_alive.alive``. These tests pin that: the montysmom
case (status 4 + alive true) MUST read as not-live.

A WAF challenge (``status_code == 4003110``) is the one case room/info can't give
us a status; there we fall back to the legacy ``alive`` flag rather than newly
going blind, so get_live_url()'s page-scrape WAF path can still fire.
"""

import json
from unittest.mock import Mock

import pytest

from core.tiktok_api import TikTokAPI
from utils.custom_exceptions import UserLiveError
from utils.enums import TikTokError

ROOM_ID = "7489823057092168456"  # montysmom's real stale room from the incident

STATUS_LIVE = 2
STATUS_ENDED = 4
WAF_STATUS_CODE = 4003110


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _room_info(*, status=None, status_code=0):
    data = {} if status is None else {"status": status}
    return {"status_code": status_code, "data": data}


def _check_alive(alive):
    return {"data": [{"alive": alive, "room_id": int(ROOM_ID)}], "status_code": 0}


def _make_api(*, room_info=None, check_alive=None):
    """A TikTokAPI whose http_client routes room/info vs check_alive by URL.

    ``__new__`` bypasses ``__init__`` so no real HTTP stack is built.
    """
    api = TikTokAPI.__new__(TikTokAPI)
    api.WEBCAST_URL = "https://webcast.tiktok.com"

    def _get(url, **kwargs):
        if "/webcast/room/info/" in url:
            assert room_info is not None, f"unexpected room/info call: {url}"
            return _Response(room_info)
        if "/webcast/room/check_alive/" in url:
            assert check_alive is not None, f"unexpected check_alive call: {url}"
            return _Response(check_alive)
        raise AssertionError(f"unexpected request to {url}")

    api.http_client = Mock()
    api.http_client.get.side_effect = _get
    return api


def _urls(api):
    return [c.args[0] for c in api.http_client.get.call_args_list]


# --- the fix: liveness is room status, not the check_alive flag --------------


def test_live_room_status_2_is_alive():
    api = _make_api(room_info=_room_info(status=STATUS_LIVE))
    assert api.is_room_alive(ROOM_ID) is True


def test_ended_room_status_4_is_not_alive_even_when_check_alive_says_true():
    """THE incident. montysmom: room ended (status 4) but check_alive lies True.

    Must read as not-live. If this ever regresses, the recorder starts phantom
    recordings again and the whole roster shows fake-live.
    """
    api = _make_api(
        room_info=_room_info(status=STATUS_ENDED),
        check_alive=_check_alive(True),
    )
    assert api.is_room_alive(ROOM_ID) is False


def test_does_not_trust_check_alive_in_the_normal_path():
    """Liveness is decided from room/info alone when there's no WAF block —
    check_alive must not even be consulted (it is the unreliable signal)."""
    api = _make_api(room_info=_room_info(status=STATUS_ENDED))  # no check_alive route
    assert api.is_room_alive(ROOM_ID) is False
    assert not any("/check_alive/" in u for u in _urls(api))


# --- edges -------------------------------------------------------------------


def test_missing_room_id_raises_not_currently_live():
    api = _make_api()
    with pytest.raises(UserLiveError) as excinfo:
        api.is_room_alive("")
    assert excinfo.value.args[0] is TikTokError.USER_NOT_CURRENTLY_LIVE


def test_room_info_without_status_is_not_alive():
    api = _make_api(room_info=_room_info(status=None))
    assert api.is_room_alive(ROOM_ID) is False


# --- WAF: room/info can't give a status, fall back to the legacy flag ---------


def test_waf_block_falls_back_to_check_alive_true():
    api = _make_api(
        room_info=_room_info(status=None, status_code=WAF_STATUS_CODE),
        check_alive=_check_alive(True),
    )
    assert api.is_room_alive(ROOM_ID) is True


def test_waf_block_falls_back_to_check_alive_false():
    api = _make_api(
        room_info=_room_info(status=None, status_code=WAF_STATUS_CODE),
        check_alive=_check_alive(False),
    )
    assert api.is_room_alive(ROOM_ID) is False
