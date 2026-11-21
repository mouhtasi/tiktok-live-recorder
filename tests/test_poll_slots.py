"""
Tests for wall-clock poll slots (tiktak §140).

## The measurement this exists to fix

On 2026-11-20, 145 of 178 monitors were on the 20-minute tier, and 58–67 % of
every day's signed room lookups fell inside ONE ~6-minute band of each
20-minute cycle (`request_log`, every day since 11-07). TikTok answered 403 to
15–28 % of the lookups inside the band and to 0.1–2.5 % outside it. Five
accounts sat in the band's core and were blocked on 90 %+ of their polls —
18 to 21 hours blind in a row, with `/status` green the whole time.

Two things made the band:
- the first-poll stagger spreads a tier over at most 300 s, so a restart packs
  all 145 cold monitors into 5 of their 20 minutes, and every later poll
  inherited that phase;
- each cycle was `interval + the poll's own duration`, so the phases drifted
  1–2 minutes a day, and an account's place inside the band moved with them.

## What replaces it

Each supervised monitor polls at a fixed offset inside its interval, derived
from its username (`zlib.crc32`, the same in every process and across
restarts). After each poll it waits for its next slot. A tier then covers its
whole interval no matter when the monitors started, and poll durations cannot
accumulate into drift. The first poll after a start keeps the 300 s stagger, so
a restart blinds no account for longer than it did before.
"""

import os
import subprocess
import sys
import threading
import zlib
from unittest.mock import Mock, patch

import pytest

from core.tiktok_recorder import TikTokRecorder
from utils.enums import Mode


class _BreakLoop(BaseException):
    """Escapes the poll loop; BaseException so the loop's catch-all cannot eat it."""


def _make_recorder(tmp_path=None, interval=20, user="alice", stop_event=None):
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
    rec._stop_event = stop_event
    rec._stop_now_event = None
    rec._session_on = False
    rec.watchlist_path = (tmp_path / "users.txt") if tmp_path else None
    if tmp_path:
        (tmp_path / "users.txt").write_text(f"{user} {interval}\n")
    rec.tiktok.is_country_blacklisted.return_value = False
    rec.tiktok.get_room_id_from_user.return_value = "room-1"
    rec.tiktok.is_room_alive.return_value = False
    return rec


# --- where an account's slot is --------------------------------------------


def test_the_slot_is_the_same_in_another_process():
    """🚨 `hash()` is salted per process (PYTHONHASHSEED), so a slot built on it
    would move at every restart and differ between the monitor and anything
    that tries to predict it. crc32 is not salted."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from core.tiktok_recorder import TikTokRecorder as T;"
        "r = T.__new__(T); r.user = 'some.account_42';"
        "print(repr(r._slot_offset_s(1200)))"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(out) == 1, out
    here = _make_recorder(user="some.account_42")._slot_offset_s(1200)
    assert out == {repr(here)}


def test_the_offset_lies_inside_the_interval():
    for i in range(500):
        off = _make_recorder(user=f"user{i}")._slot_offset_s(1200)
        assert 0 <= off < 1200


def test_a_tier_covers_its_whole_interval():
    """The band, reproduced and then removed.

    145 accounts on one 20-minute tier, as on 2026-11-20. Under the old 300 s
    stagger every one of them polled inside the same 5 minutes. Spread by slot,
    the busiest 6 minutes hold close to the uniform share (30 %).
    """
    interval = 1200
    offsets = sorted(
        _make_recorder(user=f"acct_{i:03d}")._slot_offset_s(interval) for i in range(145)
    )
    window = 360
    densest = max(
        sum(1 for o in offsets if (o - start) % interval < window) for start in offsets
    )
    assert densest / len(offsets) < 0.45, f"{densest} of 145 in one 6-min window"


# --- how long the recheck waits ----------------------------------------------


@pytest.mark.parametrize("interval_min", [1, 5, 20, 60])
def test_the_recheck_ends_on_the_slot(tmp_path, interval_min):
    rec = _make_recorder(tmp_path, interval=interval_min, user="bob")
    interval = interval_min * 60
    off = rec._slot_offset_s(interval)
    for now in (1_763_700_000.0, 1_763_700_123.4, 1_763_700_999.9, 1_763_701_777.7):
        with patch("core.tiktok_recorder.time.time", return_value=now):
            delay = rec._recheck_delay_s(interval_min)
        landed = (now + delay) % interval
        assert landed == pytest.approx(off, abs=1e-6) or landed == pytest.approx(
            off + interval, abs=1e-6
        ) or landed == pytest.approx(off - interval, abs=1e-6)


@pytest.mark.parametrize("interval_min", [1, 5, 20, 60])
def test_the_recheck_is_bounded(tmp_path, interval_min):
    """Never a double poll right after a poll, and never more than one interval
    plus that gap — so no account waits much longer than its tier promises."""
    rec = _make_recorder(tmp_path, interval=interval_min, user="carol")
    interval = interval_min * 60
    gap = rec._slot_gap_s(interval)
    assert 0 < gap <= 60
    assert gap <= interval / 4
    for k in range(400):
        now = 1_763_700_000.0 + k * 7.31
        with patch("core.tiktok_recorder.time.time", return_value=now):
            delay = rec._recheck_delay_s(interval_min)
        assert gap <= delay < interval + gap, (now, delay)


def test_poll_time_does_not_drift_the_phase(tmp_path):
    """The old cycle was `interval + poll time`, so 100 polls of ~1.5 s moved an
    account 150 s through its tier. A slot does not move."""
    rec = _make_recorder(tmp_path, interval=20, user="dave")
    interval = 1200
    off = rec._slot_offset_s(interval)
    now = 1_763_700_000.0
    for k in range(100):
        with patch("core.tiktok_recorder.time.time", return_value=now):
            now += rec._recheck_delay_s(20)
        assert (now % interval) == pytest.approx(off, abs=1e-6)
        now += 0.5 + (k % 7) * 0.4  # the poll itself takes time


def test_an_unsupervised_run_keeps_the_plain_interval(tmp_path):
    """A single-account CLI run has no herd; it waits exactly its interval."""
    rec = _make_recorder(None, interval=5, user="erin")
    assert rec._recheck_delay_s(5) == 300


def test_automatic_mode_waits_for_the_slot(tmp_path):
    """Wired in: the wait after a not-live answer is the slot wait, not the
    plain interval."""
    rec = _make_recorder(tmp_path, interval=20, user="frank")
    waits = []

    def record(seconds):
        waits.append(seconds)
        if len(waits) >= 2:  # the first-poll stagger, then the recheck
            raise _BreakLoop()

    now = 1_763_700_000.0
    with patch("core.tiktok_recorder.time.time", return_value=now):
        with patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=record), \
             pytest.raises(_BreakLoop):
            rec.automatic_mode()
        expected = rec._recheck_delay_s(20)

    assert waits[1] == pytest.approx(expected)
    assert waits[1] != 1200


# --- the wait itself -----------------------------------------------------------


def _run_wait(rec, seconds, retier_to=None, retier_after=0.0):
    """Drive _wait_for_next_poll on a fake monotonic clock; return how long it waited."""
    clock = {"t": 0.0}

    def fake_wait(timeout):
        clock["t"] += timeout
        if retier_to is not None and clock["t"] >= retier_after:
            rec.watchlist_path.write_text(f"{rec.user} {retier_to}\n")
        return False

    with patch.object(rec._stop_event, "wait", side_effect=fake_wait), \
         patch("core.tiktok_recorder.time.monotonic", lambda: clock["t"]):
        rec._wait_for_next_poll(seconds)
    return clock["t"]


def test_an_unchanged_interval_never_cuts_a_slot_wait(tmp_path):
    """🚨 The double poll. A slot wait can be up to `interval + gap`. The old
    re-tier cap, `min(seconds, interval)`, would end it `gap` seconds early; the
    poll then lands just before its slot and the next slot is under `gap` away,
    so the account polls twice in a minute, every cycle."""
    rec = _make_recorder(tmp_path, interval=20, user="gina", stop_event=threading.Event())
    assert _run_wait(rec, 1250) == pytest.approx(1250)


def test_a_retier_to_hot_still_cuts_the_wait(tmp_path):
    """§94's mid-wait re-tier must survive: a cold account promoted to the hot
    tier two minutes into a 20-minute wait polls within its new interval."""
    rec = _make_recorder(tmp_path, interval=20, user="hana", stop_event=threading.Event())
    waited = _run_wait(rec, 1250, retier_to=5, retier_after=120)
    assert waited <= 5 * 60 + rec._slot_gap_s(300)
