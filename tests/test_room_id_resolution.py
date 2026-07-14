"""
Regression tests for ``TikTokAPI.get_room_id_from_user()`` — username -> room_id.

Why this matters: room-id resolution is the gate on *all* recording. The primary
path runs through **tikrec.com**, a free third-party URL-signing service, and on
2026-07-12 it went down (Cloudflare 522, an HTML error page). The unguarded
``.json()`` on that HTML blew up for every user on every poll, so recording
stopped completely even though TikTok itself was up. A signer outage must never
again take the recorder blind.

Upstream (#448) *appears* to fix this: it raises ``TikRecUnavailableError`` when
signing fails and falls back to ``_old_get_room_id_from_user()``. But that
fallback calls **tiktok.eulerstream.com with an empty ``x-api-key``**, which
answers ``HTTP 401 — "requires the Webcast Premium add-on"`` (a paid product).
Verified live 2026-07-13. So upstream's fallback is dead code for anyone without
a subscription: tikrec down -> 401 -> ``UserLiveError`` -> recorder blind, i.e.
the same outage. **Hence ``_direct_get_room_id_from_user()``** (TikTok's public
``api-live`` endpoint, which currently answers unsigned) is the fallback we use,
and ``test_eulerstream_is_never_called`` guards that it stays that way.

The fallback must trigger on *every* way the tikrec path can fail to produce a
room_id, not just an unreachable signer:

1. signer unreachable / HTTP error            -> ``TikRecUnavailableError``
2. signer returns non-JSON (the 522 HTML)     -> ``TikRecUnavailableError``
3. signer returns JSON but no ``signed_path`` -> ``TikRecUnavailableError``
4. signer OK, but the *signed fetch* yields no ``roomId``   <- (4) and (5) are
5. signer OK, but the *signed fetch* returns non-JSON          post-signing, so
                                                               they raise no
                                                               TikRecUnavailableError

Cases 4 and 5 are the subtle ones: ``TikRecUnavailableError`` is raised only by
the *signing* call, so a narrow ``except TikRecUnavailableError`` never sees
them and the user silently resolves to ``None`` (or dies on a ValueError) —
even though the direct endpoint would have answered fine.

A genuine WAF block is *not* a signer failure and must still propagate as
``UserLiveError``, never be silently retried through the fallback.
"""

import json
from unittest.mock import Mock

import pytest

from core.tiktok_api import TikTokAPI
from utils.custom_exceptions import UserLiveError
from utils.enums import TikTokError

SIGN_URL = "https://tikrec.com/tiktok/room/api/sign"
SIGNED_PATH = "/api-live/user/room/?signed=1&_signature=abc"
DIRECT_URL = "https://www.tiktok.com/api-live/user/room/"
EULER_URL = "https://tiktok.eulerstream.com"

ROOM_ID_VIA_TIKREC = "7111111111111111111"
ROOM_ID_VIA_DIRECT = "7222222222222222222"

# The literal body tikrec served during the 2026-07-12 outage.
CLOUDFLARE_522_HTML = "<html><body>error code: 522</body></html>"
# TikTok's WAF interstitial.
WAF_HTML = "<html><body>Please wait...</body></html>"

_BAD_JSON = object()  # sentinel: .json() raises, as it does on an HTML body


class _Response:
    """Minimal stand-in for an HTTP response: ``.text``, ``.json()``,
    ``.raise_for_status()``.

    ``text`` defaults to the serialized payload, because the code under test
    reads ``.text`` *before* ``.json()`` and treats an empty body as a WAF
    block — a fake that leaves ``.text`` empty on a JSON response would trip
    that check and test nothing real.
    """

    def __init__(self, *, text=None, payload=None, http_error=None):
        if text is None:
            text = "" if payload is None or payload is _BAD_JSON else json.dumps(payload)
        self.text = text
        self._payload = payload
        self._http_error = http_error

    def raise_for_status(self):
        if self._http_error is not None:
            raise self._http_error

    def json(self):
        if self._payload is _BAD_JSON:
            raise json.JSONDecodeError("Expecting value", self.text or "", 0)
        return self._payload


def _room_payload(room_id):
    """The shape both the signed and the direct endpoint return."""
    return {"data": {"user": {"roomId": room_id}}}


def _make_api(*, sign, signed_fetch=None, direct=None):
    """Build a TikTokAPI whose ``http_client`` routes by URL.

    ``__new__`` bypasses ``__init__`` so no real HTTP stack is constructed; the
    URL constants are set exactly as ``__init__`` sets them.
    """
    api = TikTokAPI.__new__(TikTokAPI)
    api.BASE_URL = "https://www.tiktok.com"
    api.WEBCAST_URL = "https://webcast.tiktok.com"
    api.API_URL = DIRECT_URL
    api.EULER_API = EULER_URL
    api.TIKREC_API = "https://tikrec.com"

    def _get(url, **kwargs):
        if url.startswith(SIGN_URL):
            if isinstance(sign, Exception):
                raise sign
            return sign
        # The signed URL is BASE_URL + signed_path, so it shares a prefix with the
        # direct endpoint — distinguish them by the signature query, which only
        # the signed URL carries (the direct call passes its params separately).
        if "signed=1" in url:
            return signed_fetch
        if url.startswith(DIRECT_URL):
            return direct
        raise AssertionError(f"unexpected request to {url}")

    api.http_client = Mock()
    api.http_client.get.side_effect = _get
    return api


def _requested_urls(api):
    return [call.args[0] for call in api.http_client.get.call_args_list]


def _direct_calls(api):
    """Requests to the direct api-live endpoint — i.e. the fallback firing.

    The *signed* URL is ``BASE_URL + signed_path`` and so also starts with the
    direct endpoint's URL; only the signed one carries the signature query, so
    exclude it or every signed fetch reads as a fallback.
    """
    return [
        u
        for u in _requested_urls(api)
        if u.startswith(DIRECT_URL) and "signed=1" not in u
    ]


# --- the tikrec path works: no fallback, no direct call ----------------------


def test_signed_path_returns_room_id_without_falling_back():
    api = _make_api(
        sign=_Response(payload={"signed_path": SIGNED_PATH}),
        signed_fetch=_Response(payload=_room_payload(ROOM_ID_VIA_TIKREC)),
    )

    assert api.get_room_id_from_user("tester") == ROOM_ID_VIA_TIKREC
    # The direct endpoint is more WAF-prone under load, so it must stay a
    # fallback — never used while the signer is healthy.
    assert _direct_calls(api) == []


# --- every way the tikrec path can fail must reach the direct endpoint -------


@pytest.mark.parametrize(
    "sign, signed_fetch, why",
    [
        # (1) signer unreachable / HTTP error.
        (
            _Response(http_error=OSError("connection refused")),
            None,
            "signer unreachable",
        ),
        # (2) THE 2026-07-12 OUTAGE: Cloudflare 522, an HTML body -> .json() blows up.
        (
            _Response(text=CLOUDFLARE_522_HTML, payload=_BAD_JSON),
            None,
            "signer returned a 522 HTML page",
        ),
        # (3) signer answers JSON but omits signed_path (down / overloaded).
        (
            _Response(payload={}),
            None,
            "signer returned no signed_path",
        ),
        # (4) signer OK, but the signed fetch yields no roomId. Raises no
        #     TikRecUnavailableError, so a narrow handler misses it entirely.
        (
            _Response(payload={"signed_path": SIGNED_PATH}),
            _Response(payload={"data": {"user": {}}}),
            "signed fetch returned no roomId",
        ),
        # (5) signer OK, but the signed fetch returns non-JSON. Ditto — and this
        #     one doesn't just return None, it raises ValueError out of the API.
        (
            _Response(payload={"signed_path": SIGNED_PATH}),
            _Response(text="<html>nope</html>", payload=_BAD_JSON),
            "signed fetch returned non-JSON",
        ),
    ],
)
def test_falls_back_to_direct_endpoint(sign, signed_fetch, why):
    api = _make_api(
        sign=sign,
        signed_fetch=signed_fetch,
        direct=_Response(payload=_room_payload(ROOM_ID_VIA_DIRECT)),
    )

    assert api.get_room_id_from_user("tester") == ROOM_ID_VIA_DIRECT, why
    assert _direct_calls(api), why


def test_eulerstream_is_never_called():
    """The regression guard on upstream's fallback. ``_old_get_room_id_from_user``
    hits eulerstream with an empty API key -> HTTP 401 "requires the Webcast
    Premium add-on". Routing the fallback there instead of to the direct
    endpoint reintroduces the outage, silently — the code still *looks* like it
    has a fallback."""
    api = _make_api(
        sign=_Response(text=CLOUDFLARE_522_HTML, payload=_BAD_JSON),
        direct=_Response(payload=_room_payload(ROOM_ID_VIA_DIRECT)),
    )

    assert api.get_room_id_from_user("tester") == ROOM_ID_VIA_DIRECT
    assert not any(u.startswith(EULER_URL) for u in _requested_urls(api))


# --- a WAF block is not a signer failure: it must propagate ------------------


def test_waf_block_on_signed_fetch_propagates():
    # Retrying a WAF block through the direct endpoint would just earn a second
    # block from the same IP; the caller needs to see it.
    api = _make_api(
        sign=_Response(payload={"signed_path": SIGNED_PATH}),
        signed_fetch=_Response(text=WAF_HTML),
    )

    with pytest.raises(UserLiveError) as excinfo:
        api.get_room_id_from_user("tester")
    assert excinfo.value.args[0] is TikTokError.WAF_BLOCKED
    assert _direct_calls(api) == []


def test_waf_block_on_direct_fallback_propagates():
    api = _make_api(
        sign=_Response(text=CLOUDFLARE_522_HTML, payload=_BAD_JSON),
        direct=_Response(text=WAF_HTML),
    )

    with pytest.raises(UserLiveError) as excinfo:
        api.get_room_id_from_user("tester")
    assert excinfo.value.args[0] is TikTokError.WAF_BLOCKED


# --- offline users still resolve (liveness is a separate call) ---------------


def test_offline_user_still_resolves_a_room_id():
    # Both endpoints return a roomId for an offline user; is_room_alive() decides
    # liveness. If resolution treated "offline" as failure, automatic mode could
    # never pick a stream back up.
    api = _make_api(
        sign=_Response(payload={"signed_path": SIGNED_PATH}),
        signed_fetch=_Response(payload=_room_payload(ROOM_ID_VIA_TIKREC)),
    )

    assert api.get_room_id_from_user("offline_user") == ROOM_ID_VIA_TIKREC
