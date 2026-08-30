"""
Tests for the supervisor's crash-loop guard and start-up pacing.

## What went wrong on 2026-08-29

`reconcile()` respawned a dead monitor unconditionally and immediately. When
TikTok's WAF began refusing every request, all 119 monitors died within a second
or two of starting, and the supervisor rebuilt all 119 on the next 5-second
pass — forever. The surviving 3.5 minutes of log hold **2,244 respawns**. Each
one re-resolved DNS and re-hit tikrec and TikTok from cold, which is where
~63,000 requests/hour and a wedged pihole came from.

The §28 fix (setup inside the retry loop) removes tonight's *trigger*. These
tests exist because the next trigger will be a different one: any cause that
makes a monitor exit promptly reproduces the storm exactly, and the supervisor
is the only place that can bound it. A respawn policy with no notion of "this
keeps failing" is not a supervisor, it is an amplifier.

## The two properties, and the tension between them

**Bound the loop.** A monitor that dies quickly, repeatedly, must be retried
more slowly each time, up to a ceiling.

**Never stop trying.** A missed live is unrecoverable — there is no backfill for
a broadcast nobody recorded. So the backoff has a *ceiling*, never a give-up,
and any monitor that recovers must return to normal scheduling immediately. A
guard that quarantines a flapping account forever would trade a loud outage for
a silent one, which is the failure mode this project keeps paying for.

Start-up pacing (`max_spawns_per_pass`) is a separate concern that happens to
live in the same loop: even a perfectly healthy start currently fires 119
monitors simultaneously, and that burst alone is what exhausted pihole's
rate limiter — 3,182 DNS queries/minute against a 1,000/minute ceiling.
"""

import json

from core.supervisor import RecorderSupervisor


class FakeWorker:
    def __init__(self, username):
        self.username = username
        self.stop_requested = False
        self.force_stopped = False
        self._alive = True

    def is_alive(self):
        return self._alive

    def request_stop(self):
        self.stop_requested = True

    def stop_recording_now(self):
        self.force_stopped = True

    def join(self, timeout=None):
        pass

    def die(self):
        self._alive = False


class FakeClock:
    """Monotonic time under the test's control — no sleeping, no flakiness."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_supervisor(tmp_path, users, **kwargs):
    watchlist = tmp_path / "users.txt"
    watchlist.write_text("".join(f"{u}\n" for u in users))

    spawned = []

    def spawn(username):
        w = FakeWorker(username)
        spawned.append(w)
        return w

    clock = FakeClock()
    kwargs.setdefault("now", clock)
    # No jitter by default so the backoff schedule is exactly assertable; the
    # jitter itself is tested separately.
    kwargs.setdefault("jitter", lambda: 1.0)
    sup = RecorderSupervisor(
        watchlist_path=watchlist,
        stop_now_path=tmp_path / "stop_now.txt",
        spawn_worker=spawn,
        **kwargs,
    )
    return sup, spawned, clock


def _live(spawned, username):
    """The most recent worker spawned for `username`."""
    return [w for w in spawned if w.username == username][-1]


# --- the crash loop ---------------------------------------------------------


def test_a_monitor_that_dies_instantly_is_not_respawned_immediately(tmp_path):
    """The storm, in one assertion.

    Pre-fix this spawns a replacement on every pass forever. The replacement
    must instead be held back, because a monitor that lived under a second is
    reporting a condition that another attempt will not fix.
    """
    sup, spawned, clock = make_supervisor(tmp_path, ["alice"])
    sup.reconcile()
    assert len(spawned) == 1

    clock.advance(1)  # died almost instantly
    _live(spawned, "alice").die()
    result = sup.reconcile()

    assert len(spawned) == 1, "a fast-exiting monitor was respawned with no delay"
    assert "alice" in result.deferred


def test_the_backoff_grows_and_then_stops_growing(tmp_path):
    """Exponential, with a ceiling.

    The ceiling is the point: unbounded growth would eventually park a monitor
    for hours, and an account nobody is polling cannot be recorded.
    """
    sup, spawned, clock = make_supervisor(
        tmp_path, ["alice"], backoff_base=5, backoff_max=40, fast_exit_seconds=30
    )

    delays = []
    for _ in range(6):
        sup.reconcile()
        clock.advance(1)
        _live(spawned, "alice").die()
        sup.reconcile()
        delays.append(sup.retry_delay_for("alice"))
        # Wait out the backoff so the next pass actually respawns.
        clock.advance(delays[-1])

    assert delays == [5, 10, 20, 40, 40, 40], delays


def test_a_monitor_is_respawned_once_its_backoff_expires(tmp_path):
    """Backoff delays a retry. It must never cancel one."""
    sup, spawned, clock = make_supervisor(tmp_path, ["alice"], backoff_base=5)
    sup.reconcile()
    clock.advance(1)
    _live(spawned, "alice").die()
    sup.reconcile()
    assert len(spawned) == 1

    clock.advance(5)
    result = sup.reconcile()

    assert len(spawned) == 2, "the monitor was never retried after its backoff"
    assert "alice" in result.respawned


def test_a_monitor_that_ran_healthily_is_respawned_at_once(tmp_path):
    """A long-lived monitor that exits is not flapping — it is an ordinary
    death, and delaying its replacement costs recordings for no benefit."""
    sup, spawned, clock = make_supervisor(tmp_path, ["alice"], fast_exit_seconds=30)
    sup.reconcile()

    clock.advance(3600)  # ran for an hour
    _live(spawned, "alice").die()
    result = sup.reconcile()

    assert len(spawned) == 2
    assert "alice" in result.respawned
    assert "alice" not in result.deferred


def test_recovery_clears_the_backoff(tmp_path):
    """The escalation must be forgotten once the monitor proves healthy again.

    Without this a monitor that flapped at 08:00 and then ran cleanly all day
    would still be treated as a repeat offender the next time it restarts —
    the guard slowly poisoning accounts it was meant to protect.
    """
    sup, spawned, clock = make_supervisor(
        tmp_path, ["alice"], backoff_base=5, fast_exit_seconds=30
    )
    # Two fast exits: escalate. The delay is read *before* waiting it out —
    # reading it afterwards would always see 0 and assert nothing.
    delay = 0
    for _ in range(2):
        sup.reconcile()
        clock.advance(1)
        _live(spawned, "alice").die()
        sup.reconcile()
        delay = sup.retry_delay_for("alice")
        clock.advance(delay)
    assert delay == 10, "the second fast exit did not escalate"

    # Now a healthy run, then a death.
    sup.reconcile()
    clock.advance(3600)
    _live(spawned, "alice").die()
    sup.reconcile()

    assert sup.retry_delay_for("alice") == 0, "backoff survived a healthy run"


def test_one_flapping_account_does_not_delay_the_others(tmp_path):
    """Backoff is per-account.

    §37's whole premise is that one account's trouble must not touch anyone
    else's monitor; a global backoff would rebuild the coupling it removed.
    """
    sup, spawned, clock = make_supervisor(tmp_path, ["alice", "bob"], backoff_base=5)
    sup.reconcile()

    clock.advance(1)
    _live(spawned, "alice").die()
    _live(spawned, "bob").die()
    # Bob has been alive long enough to count as healthy.
    sup.reconcile()

    clock.advance(120)
    sup.reconcile()
    sup.reconcile()

    assert sup.retry_delay_for("bob") == 0
    bob_workers = [w for w in spawned if w.username == "bob"]
    assert len(bob_workers) >= 2, "bob's monitor was held back by alice's backoff"


def test_jitter_is_applied_so_monitors_do_not_retry_in_lockstep(tmp_path):
    """119 monitors failing together would otherwise retry together, forever.

    Synchronised retries are the storm in slow motion: the same simultaneous
    burst, just every N seconds instead of continuously.
    """
    sup, spawned, clock = make_supervisor(
        tmp_path, ["alice"], backoff_base=100, jitter=lambda: 0.5
    )
    sup.reconcile()
    clock.advance(1)
    _live(spawned, "alice").die()
    sup.reconcile()

    assert sup.retry_delay_for("alice") == 50, "jitter was not applied to the backoff"


def test_backoff_never_becomes_a_permanent_giveup(tmp_path):
    """There is no failure count that stops retrying. A missed live cannot be
    recovered, so the only safe floor is 'keep trying, slowly'."""
    sup, spawned, clock = make_supervisor(
        tmp_path, ["alice"], backoff_base=5, backoff_max=60
    )
    for _ in range(50):
        sup.reconcile()
        clock.advance(1)
        _live(spawned, "alice").die()
        sup.reconcile()
        clock.advance(sup.retry_delay_for("alice"))

    clock.advance(60)
    result = sup.reconcile()
    assert "alice" in result.respawned, "the supervisor gave up on an account"


# --- start-up pacing --------------------------------------------------------


def test_a_cold_start_does_not_spawn_every_monitor_at_once(tmp_path):
    """The 119-monitor thundering herd.

    Each monitor's first act is a DNS resolution and two or three TikTok calls.
    Starting all of them in one pass produced 3,182 DNS queries/minute against
    pihole's 1,000/minute limit — which then answered REFUSED, which the
    resolver retried, which were themselves queries. The burst has to be spread.
    """
    users = [f"user{i}" for i in range(50)]
    sup, spawned, _ = make_supervisor(tmp_path, users, max_spawns_per_pass=10)

    result = sup.reconcile()

    assert len(spawned) == 10, f"spawned {len(spawned)} in one pass"
    assert len(result.queued) == 40


def test_pacing_still_starts_everyone_over_successive_passes(tmp_path):
    """Spreading the start must not drop anyone. Slower is acceptable; a
    permanently unmonitored account is not."""
    users = [f"user{i}" for i in range(50)]
    sup, spawned, _ = make_supervisor(tmp_path, users, max_spawns_per_pass=10)

    for _ in range(5):
        sup.reconcile()

    assert sorted(w.username for w in spawned) == sorted(users)


def test_pacing_is_off_by_default_for_small_watchlists(tmp_path):
    """A handful of accounts must still start promptly — the pacing exists for
    the 119-account case and should be invisible below it."""
    sup, spawned, _ = make_supervisor(tmp_path, ["alice", "bob", "carol"])
    sup.reconcile()
    assert len(spawned) == 3


# --- the health file (what tiktak reads) ------------------------------------


def test_the_health_file_reports_the_storm_shape(tmp_path):
    """`/api/lives/recorder/status` said `healthy` throughout the incident.

    It could only see the parent process, which was alive and busy the whole
    time. The supervisor is the only component that knows a monitor died, so it
    has to publish that — respawn counts and who is in backoff — or the status
    page keeps reporting green through the next one.
    """
    health = tmp_path / "health.json"
    sup, spawned, clock = make_supervisor(
        tmp_path, ["alice", "bob"], health_path=health, backoff_base=5
    )
    sup.reconcile()

    clock.advance(1)
    _live(spawned, "alice").die()
    sup.reconcile()

    data = json.loads(health.read_text())
    assert data["monitors_alive"] == 1
    assert data["watched"] == 2
    # Deaths, not respawns, are the leading indicator. During the storm's
    # backoff phase a monitor can be dying repeatedly while respawns are being
    # withheld on purpose — reporting only respawns would show the guard working
    # and the failure not happening.
    assert data["deaths_total"] >= 1
    assert "alice" in data["backing_off"]
    assert data["backing_off"]["alice"] > 0


def test_a_health_file_that_cannot_be_written_does_not_stop_supervision(tmp_path):
    """Telemetry must never be able to take down the thing it measures."""
    sup, spawned, _ = make_supervisor(
        tmp_path, ["alice"], health_path=tmp_path / "no" / "such" / "dir" / "h.json"
    )
    sup.reconcile()  # must not raise
    assert len(spawned) == 1
