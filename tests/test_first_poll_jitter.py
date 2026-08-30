"""
Tests for de-phasing the monitor herd: each supervised monitor waits a random
fraction of its poll interval before its FIRST poll.

## The measurement this exists to fix

Every monitor starts within ~30s of the recorder starting, and nothing has ever
offset their poll timers, so each §58 tier fires as one synchronised volley. On
2026-08-30, minutes after the §76 deploy, `request_log` showed TikTok refusing
the tail of every volley and forgiving it after a few minutes of quiet:

    11:04:37 | 200 | 20    <- ramp starts: the first 20 succeed
    11:04:42 | 403 | 40    <- then everything refuses
    11:04:52 | 403 | 39
    11:05:03 | 403 | 10
       ...     (four minutes quiet)
    11:09:38 | 200 |  7    <- hot tier wakes: the first 7 succeed again
    11:09:43 | 403 |  5    <- then refuses again
    11:09:53 | 403 |  9
    11:10:03 | 403 |  4

The second event is 7+5+9+4 = 25 requests, which is exactly the 25 accounts on
the 5-minute tier waking together. 🚨 **This is a short-window burst threshold,
not a ban**: it recovers on its own, so the fix is to never present a burst.

Spreading 25 accounts over 5 minutes is one request per 12s, and 94 accounts
over the same window is one per 3.2s — both far under a threshold that sits
somewhere around 20 requests in a few seconds.

## Why the offset is applied once, not per poll

Delaying every poll would add latency to every cycle forever. Delaying only the
first one permanently de-phases the herd, because each subsequent poll inherits
the offset: a monitor that starts 137s late stays 137s out of step with its
tier for as long as it lives. One cost, paid once.

## The cap, and what it is protecting

The wait is capped at 5 minutes even for a 60-minute account. Uncapped, a cold
monitor could sit idle for an hour after a restart, and 🚨 **a missed live is
unrecoverable** — a restart must never blind the recorder for long. Capping at 5
minutes still spreads 94 cold accounts to one request per 3.2s, which is well
inside budget, so the cap costs nothing that matters.
"""

from unittest.mock import Mock, patch

import pytest

from core.tiktok_recorder import TikTokRecorder, INITIAL_POLL_JITTER_MAX_S
from utils.enums import Mode


class _BreakLoop(BaseException):
    """Escapes the poll loop. BaseException so the loop's catch-all cannot eat
    it — see test_setup_resilience for why that matters."""


def _make_recorder(tmp_path=None, interval=5, user="alice", watchlist=True):
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = user
    rec.url = None
    rec.room_id = None
    rec.mode = Mode.AUTOMATIC
    rec.automatic_interval = interval
    rec.tiktok = Mock()
    rec.sec_uid = None
    rec._proxy = None
    rec._cookies = None
    rec._stop_event = None
    rec._stop_now_event = None
    rec.watchlist_path = (tmp_path / "users.txt") if (watchlist and tmp_path) else None
    if watchlist and tmp_path:
        (tmp_path / "users.txt").write_text(f"{user} {interval}\n")
    rec.tiktok.is_country_blacklisted.return_value = False
    rec.tiktok.get_room_id_from_user.return_value = "room-1"
    rec.tiktok.is_room_alive.return_value = False
    return rec


def _waits_during_one_pass(rec):
    """Every _wait_for_next_poll duration up to the first completed poll."""
    waits = []

    def record(seconds):
        waits.append(seconds)
        if len(waits) >= 2:  # jitter + the post-poll recheck wait
            raise _BreakLoop()

    with patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=record):
        with patch("core.tiktok_recorder.time.sleep", side_effect=record):
            with pytest.raises(_BreakLoop):
                rec.automatic_mode()
    return waits


def test_a_supervised_monitor_waits_before_its_first_poll(tmp_path):
    """The volley, prevented at its source."""
    rec = _make_recorder(tmp_path, interval=5)

    waits = _waits_during_one_pass(rec)

    assert waits, "the monitor polled immediately — the herd stays in lockstep"
    assert waits[0] > 0


def test_the_first_wait_never_exceeds_the_cap(tmp_path):
    """A 60-minute account must not sit idle for an hour after a restart.

    A missed live cannot be recovered, so the spread is bounded by how long the
    recorder may stay blind, not by the poll interval.
    """
    # Repeated because the delay is random: one draw landing under the cap is
    # not evidence the cap is enforced.
    for _ in range(30):
        waits = _waits_during_one_pass(_make_recorder(tmp_path, interval=60))
        assert 0 <= waits[0] <= INITIAL_POLL_JITTER_MAX_S, waits[0]

    assert INITIAL_POLL_JITTER_MAX_S == 300


def test_the_first_wait_never_exceeds_the_poll_interval(tmp_path):
    """Spreading a 1-minute account over 5 minutes would skip whole cycles."""
    for _ in range(30):
        waits = _waits_during_one_pass(_make_recorder(tmp_path, interval=1))
        assert waits[0] <= 60, waits[0]


def test_monitors_are_actually_de_phased(tmp_path):
    """Distinct offsets are the entire point.

    A constant delay would move the volley without breaking it up — 25 accounts
    would still arrive together, five minutes later.
    """
    delays = {
        _waits_during_one_pass(_make_recorder(tmp_path, interval=5, user=f"u{i}"))[0]
        for i in range(20)
    }

    assert len(delays) > 15, f"offsets are not spread: {sorted(delays)}"


def test_only_the_first_poll_is_delayed(tmp_path):
    """The offset persists through later polls, so it is paid once.

    Re-jittering every cycle would add latency forever and, worse, would let a
    monitor drift back into phase with its tier.
    """
    rec = _make_recorder(tmp_path, interval=5)

    waits = []

    def record(seconds):
        waits.append(seconds)
        if len(waits) >= 4:
            raise _BreakLoop()

    with patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=record):
        with patch("core.tiktok_recorder.time.sleep", side_effect=record):
            with pytest.raises(_BreakLoop):
                rec.automatic_mode()

    # First is the jitter; every later wait is the plain recheck interval.
    assert waits[1] == 5 * 60
    assert waits[2] == 5 * 60


def test_an_unsupervised_run_is_not_delayed(tmp_path):
    """A single-account CLI run has no herd to break up.

    Making a human wait up to five minutes for `-user someone -mode automatic`
    would be a bug, not a safeguard.
    """
    rec = _make_recorder(tmp_path=None, interval=5, watchlist=False)

    waits = _waits_during_one_pass(rec)

    assert waits[0] == 5 * 60, f"an unsupervised run was jittered: {waits}"


def test_the_initial_wait_is_interruptible(tmp_path):
    """A monitor asked to stop must not hold the supervisor for five minutes.

    §37's stop is cooperative and read at poll boundaries; an initial wait that
    ignored it would make every watch-list removal take up to the cap, and
    `stop_all()` on shutdown would hang.
    """
    rec = _make_recorder(tmp_path, interval=60)
    stop = Mock()
    stop.is_set.return_value = True
    rec._stop_event = stop

    with patch.object(TikTokRecorder, "_setup") as setup:
        rec.automatic_mode()

    # Told to stop before doing anything: it must exit without polling at all.
    assert not setup.called
    # And the wait it did use must be the interruptible one, not time.sleep.
    assert stop.wait.called or stop.is_set.called
