"""
Regression tests for the 2026-08-29 respawn storm: ``_setup()`` runs OUTSIDE
``automatic_mode()``'s retry loop, so a network error during it kills the
monitor process instead of being retried.

## The incident

``run()`` calls ``self._setup()`` and *then* dispatches to ``automatic_mode()``.
``_setup()` makes three to four network calls (country blacklist probe, room-id
resolution, and a liveness probe that is embedded in a log message's f-string).
None of them are inside the loop's ``try``.

On 2026-08-29 TikTok's WAF began refusing the direct room-id endpoint. The
traceback recovered from prod:

    File "src/core/tiktok_recorder.py", line 151, in run
        self._setup()
    File "src/core/tiktok_recorder.py", line 122, in _setup
        self.room_id = self.tiktok.get_room_id_from_user(self.user)
    File "src/core/tiktok_api.py", line 276, in _direct_get_room_id_from_user
        raise UserLiveError(TikTokError.WAF_BLOCKED)

``UserLiveError`` is the class ``automatic_mode()`` **already** handles — it
means "not live right now", and the loop's response is to wait one poll interval
and try again. Raised one frame earlier, in ``_setup()``, the identical
exception escapes ``run()``, ``record_user`` logs it, and the process exits.
The supervisor then respawned all 119 monitors every 5 seconds: 2,244 respawns
and 2,205 WAF tracebacks in the 3.5 minutes of log that survived rotation,
~63,000 requests/hour at TikTok, and the box at load 38.

So the bug is not the exception and not the WAF. It is that a *retryable*
condition was raised in the one place where nothing retries.

## What these tests pin down

The fix must make setup failures take the same path as poll failures. That is a
property of the *shape* — every network call in setup — not of the single line
that happened to throw, so the tests cover each call site in ``_setup()``
separately. Testing only ``get_room_id_from_user`` would pass against a fix that
left ``check_country_blacklisted`` unguarded, which is the same class of partial
fix that let this bug survive §28.
"""

from unittest.mock import Mock, patch

import pytest

from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import TikTokRecorderError, UserLiveError
from utils.enums import Mode


class _BreakLoop(BaseException):
    """Raised from a patched wait to escape the otherwise-infinite loop.

    Derives from BaseException, not Exception, and that is load-bearing: the
    handler under test is a deliberate catch-all, so an Exception-derived
    sentinel raised from inside the try block would be *swallowed by the very
    code being tested* and the test would hang instead of failing. Only
    KeyboardInterrupt/SystemExit-style exceptions can be relied on to escape it —
    which is also the property that lets the real recorder still shut down
    cleanly on Ctrl-C.
    """


def _make_recorder(user="tester"):
    """A recorder wired for watch-list automatic mode, with no network stack."""
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = user
    rec.url = None
    rec.room_id = None
    rec.mode = Mode.AUTOMATIC
    rec.automatic_interval = 5
    rec.tiktok = Mock()
    rec.sec_uid = None
    rec._proxy = None
    rec._cookies = None
    rec._stop_event = None
    rec._stop_now_event = None
    rec.watchlist_path = None
    # Healthy defaults; each test breaks exactly one of these.
    rec.tiktok.is_country_blacklisted.return_value = False
    rec.tiktok.get_room_id_from_user.return_value = "7671634437293067025"
    rec.tiktok.is_room_alive.return_value = False
    return rec


def _run_until_first_wait(recorder):
    """Run ``run()`` until the loop's first wait, which we turn into
    ``_BreakLoop``.

    If setup errors are handled, the loop reaches a wait and ``_BreakLoop``
    propagates. If they escape (the pre-fix behaviour), the *original* exception
    propagates instead and ``pytest.raises(_BreakLoop)`` fails — which is
    precisely the regression.
    """
    with patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=_BreakLoop):
        with patch("core.tiktok_recorder.time.sleep", side_effect=_BreakLoop):
            with pytest.raises(_BreakLoop):
                recorder.run()


# --- the exact incident ----------------------------------------------------


def test_waf_block_resolving_the_room_id_does_not_kill_the_monitor():
    """The 2026-08-29 storm, reproduced at its source.

    A WAF block during room-id resolution must become a wait-and-retry, exactly
    as it already does when raised from inside the loop. Before the fix this
    raises UserLiveError out of run() and the monitor process exits.
    """
    rec = _make_recorder()
    rec.tiktok.get_room_id_from_user.side_effect = UserLiveError(
        "Your IP is blocked by TikTok WAF. Please change your IP address."
    )

    _run_until_first_wait(rec)


def test_country_blacklist_probe_failure_does_not_kill_the_monitor():
    """`check_country_blacklisted()` is the FIRST network call in setup.

    It was not the line in the prod traceback, which is exactly why it is
    tested: a fix aimed only at the observed line leaves this one fatal, and the
    next outage looks brand new.
    """
    rec = _make_recorder()
    rec.tiktok.is_country_blacklisted.side_effect = ConnectionError("dns is down")

    _run_until_first_wait(rec)


def test_liveness_probe_failure_during_setup_does_not_kill_the_monitor():
    """The third call — and the one that is easiest to miss, because it is
    invoked from inside an f-string in a logger.info() argument."""
    rec = _make_recorder()
    rec.tiktok.is_room_alive.side_effect = ConnectionError("connection reset")

    _run_until_first_wait(rec)


# --- the shape, not the instance -------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        UserLiveError("not live"),
        TikTokRecorderError("country blacklisted in automatic mode"),
        ConnectionError("transport"),
        ValueError("json decode"),
        RuntimeError("something nobody predicted"),
    ],
    ids=["userlive", "recordererror", "connection", "valueerror", "unforeseen"],
)
def test_no_setup_exception_escapes_the_monitor_loop(exc):
    """Any exception from setup must be retried, not fatal.

    The catch-all matters for the same reason it did in §28: the process model
    turns *any* escaping exception into a silently unmonitored account, and now
    also into a hot respawn loop. Enumerating survivable exception types is what
    failed twice already.
    """
    rec = _make_recorder()
    rec.tiktok.get_room_id_from_user.side_effect = exc

    _run_until_first_wait(rec)


def test_a_recovering_setup_goes_on_to_record():
    """The retry must actually re-run setup, not just swallow the error.

    A guard that caught the exception but never retried would pass every test
    above while leaving the monitor alive and permanently useless — the §28
    'blind recorder' failure wearing the fix's clothes.
    """
    rec = _make_recorder()

    # Fail once, then succeed for good. A plain two-element list would break on
    # the loop's own re-resolution right after setup, which is a property of the
    # test's mocking rather than of the behaviour under test.
    calls = {"n": 0}

    def flaky(_user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UserLiveError("Your IP is blocked by TikTok WAF.")
        return "7671634437293067025"

    rec.tiktok.get_room_id_from_user.side_effect = flaky
    rec.tiktok.is_room_alive.return_value = True

    with patch.object(TikTokRecorder, "start_recording") as rec_start:
        with patch.object(TikTokRecorder, "_wait_for_next_poll"):
            with patch("core.tiktok_recorder.time.sleep"):
                # Stop after the recording so the loop terminates.
                rec_start.side_effect = _BreakLoop
                with pytest.raises(_BreakLoop):
                    rec.run()

    assert rec_start.called, "the monitor never retried setup after recovering"


def test_setup_failure_waits_before_retrying():
    """A failing setup must back off, never spin.

    Without a wait this loop becomes the respawn storm again, just relocated
    inside one process instead of across 119 of them.
    """
    rec = _make_recorder()

    # Fail twice, then break out so the assertions can run.
    calls = {"n": 0}

    def always_waf(_user):
        calls["n"] += 1
        if calls["n"] > 2:
            raise _BreakLoop()
        raise UserLiveError("WAF")

    rec.tiktok.get_room_id_from_user.side_effect = always_waf

    slept = []
    with patch.object(
        TikTokRecorder, "_wait_for_next_poll", side_effect=lambda s: slept.append(s)
    ):
        with patch(
            "core.tiktok_recorder.time.sleep", side_effect=lambda s: slept.append(s)
        ):
            with pytest.raises(_BreakLoop):
                rec.run()

    assert slept, "setup failed in a tight loop with no delay"
    assert all(s > 0 for s in slept), f"non-positive backoff: {slept}"
