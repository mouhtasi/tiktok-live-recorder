"""
Tests for per-account liveness poll intervals (tiktak §58, candidate 2).

## Why this exists

Every monitor currently rechecks on the same global `-automatic_interval`
(5 minutes). Measured on tiktak's roster 2026-08-05: **81 of 98 watched accounts
have never gone live once**, and they cost the same 12 polls/hr as the 17 that
stream daily. Tiering the cold ones to 60 minutes cuts the recorder's TikTok
traffic ~2.8x, and the recorder is ~83% of that traffic now that §57 shipped.

## The design constraint that shapes every test here

🚨 **A missed live is unrecoverable.** A missed enumerate only delays discovery —
the post is still there next walk. A broadcast nobody recorded is gone forever.
So every fallback in this module must resolve **toward faster polling**: whenever
the watch-list cannot answer, the monitor uses the configured default, never a
slower guess.

## Where the interval comes from, and why not from the supervisor

The monitor re-reads its **own** interval from the watch-list file at each poll
boundary. Two reasons it is not passed at spawn:

* a tier change must not respawn the monitor — §36/§37 made a watch-list change
  stop restarting the recorder, and a respawn truncates any recording in flight;
* the supervisor's job is *who* runs, not *how fast*. Keeping the interval out of
  `spawn_worker()` leaves §37's reconcile contract untouched.

## The bug the long interval creates, which 5 minutes hid

🚨 `automatic_mode()` checks `should_stop()` only at the **loop top**, then sleeps
the whole interval. At 5 minutes a cooperative stop (§37 — de-listing an account)
lands within 5 minutes and nobody notices. **At 60 minutes it would take up to an
hour**, and §38b's "N retiring" status would read green-but-wrong for that hour.
So the wait must be `stop_event.wait(...)`, not `time.sleep(...)`.
`test_stop_interrupts_a_long_wait` is the guard, and it is the whole reason this
change is not a one-line constant edit.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from core.supervisor import read_watchlist, read_watchlist_entries
from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import LiveNotFound


# --- parsing the extended watch-list ----------------------------------------


def test_watchlist_entry_carries_an_optional_interval(tmp_path):
    p = tmp_path / "users.txt"
    p.write_text("alice 60\nbob 5\n")

    entries = read_watchlist_entries(p)

    assert [(e.username, e.interval_min) for e in entries] == [
        ("alice", 60),
        ("bob", 5),
    ]


def test_bare_username_line_has_no_interval(tmp_path):
    """The old one-name-per-line format must keep working unchanged.

    tiktak may publish a plain list at any time (and did for every release before
    §58). `None` means "no instruction", which the monitor resolves to its
    configured default — not to some slower fallback.
    """
    p = tmp_path / "users.txt"
    p.write_text("alice\nbob 30\n")

    entries = read_watchlist_entries(p)

    assert entries[0].username == "alice"
    assert entries[0].interval_min is None
    assert entries[1].interval_min == 30


@pytest.mark.parametrize("junk", ["abc", "0", "-5", "3.5", "", "5x", "99999999999"])
def test_malformed_interval_reads_as_no_instruction(tmp_path, junk):
    """A garbled interval must never crash the parse or slow the poll.

    The watch-list is machine-written, so junk here means tiktak has a bug — and
    the safe response to *our* bug is to keep polling at the default rate, not to
    stop watching the account or to park it on a nonsense interval. `0` and
    negative values are rejected for the same reason: they are not "poll fast",
    they are "spin".
    """
    p = tmp_path / "users.txt"
    p.write_text(f"alice {junk}\n")

    entries = read_watchlist_entries(p)

    assert len(entries) == 1
    assert entries[0].username == "alice"
    assert entries[0].interval_min is None


def test_parsing_still_tolerates_untidy_files(tmp_path):
    p = tmp_path / "users.txt"
    p.write_text("\n  # a comment\n\n  alice   60  \n\nbob\n# trailing\n")

    entries = read_watchlist_entries(p)

    assert [(e.username, e.interval_min) for e in entries] == [
        ("alice", 60),
        ("bob", None),
    ]


def test_duplicate_username_keeps_the_first_entry(tmp_path):
    """A user listed twice must still get exactly one monitor (§29: two
    processes recording one account double-write the file → corruption). The
    interval of the first line wins; the point is that the count stays one."""
    p = tmp_path / "users.txt"
    p.write_text("alice 60\nalice 5\n")

    entries = read_watchlist_entries(p)

    assert len(entries) == 1
    assert entries[0].interval_min == 60


def test_read_watchlist_still_returns_plain_usernames(tmp_path):
    """`read_watchlist` is also the parser for the *stop-now command file*, which
    has no interval column and must not grow one. Keeping its signature means the
    §37 supervisor and its tests are untouched by §58."""
    p = tmp_path / "users.txt"
    p.write_text("alice 60\nbob\n")

    assert read_watchlist(p) == ["alice", "bob"]


def test_missing_watchlist_still_raises(tmp_path):
    """Unchanged from §37 and deliberately so: an *empty* list means "record
    nobody", a *missing* file means "no information — change nothing"."""
    with pytest.raises(FileNotFoundError):
        read_watchlist_entries(tmp_path / "nope.txt")


# --- the monitor resolving its own interval ----------------------------------


def _make_recorder(tmp_path=None, interval=5, user="alice", stop_event=None):
    """A recorder with no network stack, as in test_automatic_mode_resilience."""
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = user
    rec.automatic_interval = interval
    rec.tiktok = Mock()
    rec._stop_event = stop_event
    rec._stop_now_event = None
    rec.watchlist_path = (tmp_path / "users.txt") if tmp_path else None
    return rec


def test_monitor_uses_its_own_interval_from_the_watchlist(tmp_path):
    (tmp_path / "users.txt").write_text("alice 60\nbob 5\n")
    rec = _make_recorder(tmp_path, interval=5, user="alice")

    assert rec._poll_interval_minutes() == 60


def test_each_monitor_reads_only_its_own_row(tmp_path):
    (tmp_path / "users.txt").write_text("alice 60\nbob 5\n")

    assert _make_recorder(tmp_path, user="alice")._poll_interval_minutes() == 60
    assert _make_recorder(tmp_path, user="bob")._poll_interval_minutes() == 5


def test_monitor_absent_from_watchlist_uses_the_default(tmp_path):
    """A monitor mid-retirement is no longer listed. It must keep polling at the
    default until it exits, not slow down or speed up on the way out."""
    (tmp_path / "users.txt").write_text("bob 60\n")
    rec = _make_recorder(tmp_path, interval=5, user="alice")

    assert rec._poll_interval_minutes() == 5


@pytest.mark.parametrize("breakage", ["missing", "unreadable", "no_path"])
def test_unresolvable_watchlist_uses_the_default(tmp_path, breakage):
    """🚨 Bias fast on every failure. A filesystem blip must not silently park a
    real streamer on an hourly poll — that loses broadcasts, and it reports
    green while doing it."""
    rec = _make_recorder(tmp_path, interval=5, user="alice")

    if breakage == "missing":
        pass  # never created
    elif breakage == "unreadable":
        (tmp_path / "users.txt").write_text("alice 60\n")
        # Patched where it is *used*, not where it is defined — tiktok_recorder
        # binds the name at import, so patching core.supervisor would miss.
        with patch(
            "core.tiktok_recorder.read_watchlist_entries", side_effect=OSError("boom")
        ):
            assert rec._poll_interval_minutes() == 5
        return
    elif breakage == "no_path":
        rec.watchlist_path = None

    assert rec._poll_interval_minutes() == 5


def test_interval_change_is_picked_up_without_respawn(tmp_path):
    """The whole reason the interval is re-read per poll instead of fixed at
    spawn: re-tiering an account must not restart its monitor."""
    wl = tmp_path / "users.txt"
    wl.write_text("alice 60\n")
    rec = _make_recorder(tmp_path, interval=5, user="alice")

    assert rec._poll_interval_minutes() == 60

    wl.write_text("alice 5\n")  # promoted to the hot tier

    assert rec._poll_interval_minutes() == 5


# --- the stop must not wait out a long interval ------------------------------


def test_supervised_wait_uses_the_stop_event_not_sleep(tmp_path):
    """When supervised, the recheck delay must be an interruptible wait."""
    (tmp_path / "users.txt").write_text("alice 60\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, interval=5, user="alice", stop_event=stop)

    with patch.object(stop, "wait", return_value=False) as waited:
        with patch("core.tiktok_recorder.time.sleep") as slept:
            rec._wait_for_next_poll(42)

    waited.assert_called_once_with(42)
    slept.assert_not_called()


def test_unsupervised_wait_still_sleeps(tmp_path):
    """Single-user manual runs have no stop event; they must still work, and the
    §28/§37 resilience tests drive exactly this path."""
    rec = _make_recorder(tmp_path, interval=5, user="alice", stop_event=None)

    with patch("core.tiktok_recorder.time.sleep") as slept:
        rec._wait_for_next_poll(42)

    slept.assert_called_once_with(42)


def test_stop_interrupts_a_long_wait(tmp_path):
    """🚨 The guard for the bug a 60-minute tier introduces.

    A monitor asked to retire must exit at its next poll boundary — which, with
    an hourly interval, must NOT mean "in up to an hour". Before §58 this was
    `time.sleep(interval)`, so de-listing an account on the cold tier would leave
    its monitor alive for an hour and the status page showing "1 retiring" the
    whole time (§38b: a routine opt-out that reads as a stuck system).

    Driven with a real Event across a real thread, because the failure mode is a
    blocking call that ignores it. A mocked wait cannot fail this test.
    """
    (tmp_path / "users.txt").write_text("alice 600\n")  # ten hours
    stop = threading.Event()
    rec = _make_recorder(tmp_path, interval=600, user="alice", stop_event=stop)
    # Never live, so every iteration goes straight to the recheck wait.
    rec.tiktok.get_room_id_from_user.side_effect = LiveNotFound("not live")

    thread = threading.Thread(target=rec.automatic_mode, daemon=True)
    thread.start()
    time.sleep(0.3)  # let it reach the wait
    assert thread.is_alive(), "monitor exited before it could be asked to stop"

    stop.set()
    thread.join(timeout=5)

    assert not thread.is_alive(), (
        "monitor did not wake on the stop event — it is sleeping out its full "
        "interval, which at the cold tier is an hour"
    )
