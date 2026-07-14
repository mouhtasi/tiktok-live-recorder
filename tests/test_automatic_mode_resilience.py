"""
Regression tests for ``TikTokRecorder.automatic_mode()``'s resilience to
transient errors during the poll/record cycle.

Why this matters: in multi-user automatic mode each username is monitored in
its own OS process (``main.run_recordings`` spawns one
``multiprocessing.Process`` per user). The parent only ``join()``s those
children — it never respawns a dead one — and ``record_user`` catches
exceptions merely to log them before the process exits. So if
``automatic_mode()`` lets an exception escape its loop, that user is silently
dropped until the entire recorder is restarted. The loop must therefore survive
*any* transient per-iteration error.

The guard was originally ``except ConnectionError`` (builtin only), later
widened upstream to ``except (ConnectionError, RequestException,
HTTPException)``. That still misses real transport errors:

* the ``is_room_alive`` / main-API path uses **curl_cffi**, whose exceptions
  derive from ``curl_cffi.CurlError -> OSError``, *not*
  ``requests.RequestException``; and
* a WAF / HTML response makes ``response.json()`` raise the stdlib
  ``json.JSONDecodeError``.

Both escape the old tuple and kill the monitor process. The fix replaces the
tuple with a broad ``except Exception`` that logs and retries after the
connection-closed backoff (``KeyboardInterrupt`` / ``SystemExit`` are
``BaseException`` and still propagate, so graceful shutdown is unaffected).
"""

import json
from http.client import HTTPException
from unittest.mock import Mock, patch

import pytest
from requests import RequestException
from requests.exceptions import ConnectionError as RequestsConnectionError
from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError

from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import LiveNotFound, UserLiveError
from utils.enums import TimeOut

# The exception set the previous (upstream `develop`) handler caught. Anything
# outside this tuple escaped the loop and killed the per-user monitor process.
OLD_HANDLER = (ConnectionError, RequestException, HTTPException)


class _BreakLoop(Exception):
    """Sentinel raised from a patched ``time.sleep`` to escape the ``while True``."""


def _make_recorder(interval=5, user="tester"):
    # Bypass __init__ so no real HTTP client / network stack is constructed.
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = user
    rec.automatic_interval = interval
    rec.tiktok = Mock()
    return rec


def _run_until_first_sleep(recorder):
    """Run ``automatic_mode()`` until its first ``time.sleep``, which we turn
    into ``_BreakLoop`` so the infinite loop terminates. Returns the sleep mock
    so callers can assert on the backoff that was requested. If the loop instead
    lets the original exception escape (the pre-fix behaviour), that exception —
    not ``_BreakLoop`` — propagates and the enclosing ``pytest.raises`` fails."""
    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop) as slp:
        with pytest.raises(_BreakLoop):
            recorder.automatic_mode()
    return slp


@pytest.mark.parametrize(
    "exc, caught_by_old_handler",
    [
        # The error that actually hit prod on 2026-07-12 (came via plain
        # `requests`) — develop's widened tuple would have caught this one.
        (RequestsConnectionError("connection reset by peer"), True),
        # curl_cffi transport error (the is_room_alive / main-API path) —
        # NOT a requests.RequestException, so develop's tuple misses it.
        (CurlConnectionError("connection reset by peer"), False),
        # WAF / HTML response -> response.json() raises stdlib JSONDecodeError.
        (json.JSONDecodeError("Expecting value", "", 0), False),
        # Any other surprise must not take down the monitor either.
        (RuntimeError("unexpected"), False),
    ],
)
def test_transient_poll_error_does_not_kill_monitor(exc, caught_by_old_handler):
    # Premise check: the "improvement" cases really are outside the old handler's
    # reach, so the pre-fix code would have propagated them and killed the user's
    # monitor process. This is what our broad handler now prevents.
    assert isinstance(exc, OLD_HANDLER) is caught_by_old_handler

    recorder = _make_recorder()
    recorder.tiktok.get_room_id_from_user.side_effect = exc

    slp = _run_until_first_sleep(recorder)

    # The loop caught the error and backed off instead of dying.
    slp.assert_called_once_with(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)


def test_keyboardinterrupt_is_not_swallowed():
    # KeyboardInterrupt (BaseException) must still propagate so the parent's
    # Ctrl-C / graceful-shutdown path works; `except Exception` must not eat it.
    recorder = _make_recorder()
    recorder.tiktok.get_room_id_from_user.side_effect = KeyboardInterrupt

    with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop):
        with pytest.raises(KeyboardInterrupt):
            recorder.automatic_mode()


@pytest.mark.parametrize("not_live_exc", [UserLiveError("nope"), LiveNotFound("nope")])
def test_user_not_live_still_uses_recheck_interval(not_live_exc):
    # The normal "not live" path is unchanged: it waits the full
    # automatic_interval, not the shorter connection-closed backoff — proving
    # the broad handler didn't collapse the two distinct wait behaviours.
    recorder = _make_recorder(interval=7)
    recorder.tiktok.get_room_id_from_user.side_effect = not_live_exc

    slp = _run_until_first_sleep(recorder)

    slp.assert_called_once_with(7 * TimeOut.ONE_MINUTE)
