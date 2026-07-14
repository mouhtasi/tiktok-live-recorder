"""
Tests for the reconciling supervisor (PLAN §37) — the parent loop that owns the
per-user monitor processes.

## Why this exists

`main.run_recordings()` already spawns **one process per user** and then does
nothing but `join()` them. Per-account isolation is therefore *already there* —
two things are missing, and neither is a process:

1. the parent never supervises, so **a dead monitor is never respawned** (this is
   §28's root cause, sitting in upstream: one uncaught exception took the
   recorder blind to a user for 3.5h); and
2. the watch-list is fixed at argv, so adding/removing a user means **restarting
   the whole recorder** — which truncates every recording in flight, including
   for the users the change doesn't touch (the harm §35's deferral guard works
   around downstream).

The supervisor replaces fire-and-join with a reconcile loop against a watch-list
**file** that tiktak writes atomically. A file (not a socket) because it is
atomic, inspectable, survives restarts, and keeps the recorder ignorant of our
SQLite schema.

## The two failure modes these tests exist to prevent

Both of the incidents this code rewrites failed **silently**, so "it looked fine"
is not evidence:

* **§28 (blind recorder):** a worker dies and nobody notices. Guarded by
  `test_dead_worker_in_watchlist_is_respawned`.
* **§29 (orphaned/duplicated recorders):** two processes recording the same user
  concurrently, double-writing the same file → corruption. Guarded by
  `test_reconcile_is_idempotent` and `test_live_worker_is_never_respawned` — a
  supervisor that respawns a worker that is actually still alive *creates* §29.

Plus the one the whole feature is for:
`test_adding_a_user_does_not_disturb_existing_workers` — no other worker may be
touched, because touching one truncates its recording.

## Contract assumed of a worker handle (so tests can fake it)

`is_alive()`, `request_stop()` (cooperative — the worker exits at its next poll
boundary, never mid-recording; see test_worker_stop.py), `join(timeout=...)`.
"""

import pytest

from core.supervisor import RecorderSupervisor, read_watchlist


class FakeWorker:
    """Stand-in for a per-user monitor process."""

    def __init__(self, username):
        self.username = username
        self.stop_requested = False
        self.force_stopped = False
        self.joined = False
        self._alive = True

    # -- the handle protocol the supervisor is allowed to rely on --
    def is_alive(self):
        return self._alive

    def request_stop(self, force=False):
        self.stop_requested = True
        if force:
            self.force_stopped = True

    def join(self, timeout=None):
        self.joined = True

    # -- test affordances --
    def die(self):
        """Simulate the §28 failure: the process is gone, nobody was told."""
        self._alive = False

    def finish(self):
        """Simulate a cooperative exit completing after request_stop()."""
        self._alive = False


def make_supervisor(tmp_path, users):
    """Supervisor with an injected spawner, so no real processes are created."""
    watchlist = tmp_path / "users.txt"
    write_watchlist(watchlist, users)

    spawned = []

    def spawn(username):
        w = FakeWorker(username)
        spawned.append(w)
        return w

    sup = RecorderSupervisor(
        watchlist_path=watchlist,
        stop_now_path=tmp_path / "stop_now.txt",
        spawn_worker=spawn,
    )
    return sup, watchlist, spawned


def write_watchlist(path, users):
    path.write_text("".join(f"{u}\n" for u in users))


# --- reading the watch-list -------------------------------------------------


def test_watchlist_parsing_tolerates_untidy_files(tmp_path):
    p = tmp_path / "users.txt"
    p.write_text("\n alice \n\n# a comment\nbob\nalice\n\n")

    # Blank lines, surrounding whitespace and comments are ignored; duplicates
    # collapse (a user listed twice must not get two recorders — that is §29).
    assert read_watchlist(p) == ["alice", "bob"]


def test_missing_watchlist_raises_rather_than_reading_as_empty(tmp_path):
    # An empty list legitimately means "record nobody", so a *missing* file must
    # not be silently indistinguishable from it — see the reconcile test below.
    with pytest.raises(FileNotFoundError):
        read_watchlist(tmp_path / "nope.txt")


# --- first reconcile: one worker per user -----------------------------------


def test_first_reconcile_spawns_one_worker_per_user(tmp_path):
    sup, _, spawned = make_supervisor(tmp_path, ["alice", "bob", "carol"])

    result = sup.reconcile()

    assert sorted(result.spawned) == ["alice", "bob", "carol"]
    assert sorted(w.username for w in spawned) == ["alice", "bob", "carol"]
    assert len(spawned) == 3  # exactly one each — never two (that is §29)


def test_reconcile_is_idempotent(tmp_path):
    # Nothing changed => nothing happens. A supervisor that churns here would
    # restart healthy workers and truncate their recordings.
    sup, _, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()

    result = sup.reconcile()

    assert result.spawned == [] and result.stopped == [] and result.respawned == []
    assert len(spawned) == 2  # no new processes


# --- the point of the feature: per-account add/remove -----------------------


def test_adding_a_user_does_not_disturb_existing_workers(tmp_path):
    """THE test. Today, adding an account restarts the whole recorder and
    truncates every recording in flight. After §37 it must spawn exactly one new
    worker and leave every other worker *untouched* — same object, not stopped,
    not re-spawned."""
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()
    before = {w.username: w for w in spawned}

    write_watchlist(watchlist, ["alice", "bob", "carol"])
    result = sup.reconcile()

    assert result.spawned == ["carol"]
    assert result.stopped == [] and result.respawned == []
    # alice and bob were not touched at all.
    for name in ("alice", "bob"):
        assert before[name].stop_requested is False
        assert before[name].is_alive()
        assert sup.workers[name] is before[name]


def test_removing_a_user_stops_only_that_worker(tmp_path):
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()
    before = {w.username: w for w in spawned}

    write_watchlist(watchlist, ["bob"])
    result = sup.reconcile()

    assert result.stopped == ["alice"]
    assert before["alice"].stop_requested is True
    # Disabling @alice must not touch @bob — that is the whole reason §35 needed
    # a deferral guard, and the reason §37 makes it unnecessary.
    assert before["bob"].stop_requested is False
    assert before["bob"].is_alive()


def test_stopped_worker_is_reaped_once_it_exits(tmp_path):
    # request_stop() is cooperative: the worker keeps running (possibly still
    # recording) until its next poll boundary. It must stay tracked until then,
    # and must NOT be asked to stop twice or double-counted.
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()
    alice = next(w for w in spawned if w.username == "alice")

    write_watchlist(watchlist, ["bob"])
    sup.reconcile()
    assert "alice" in sup.workers  # still finishing its recording

    result = sup.reconcile()
    assert result.stopped == []  # not re-stopped
    assert "alice" in sup.workers

    alice.finish()  # the recording ended; the worker exited at the poll boundary
    result = sup.reconcile()

    assert result.reaped == ["alice"]
    assert alice.joined is True  # joined, so no zombie is left behind
    assert "alice" not in sup.workers


def test_empty_watchlist_stops_everything(tmp_path):
    # "Record nobody" is a legitimate state (every account paused). It is only
    # safe to honour because tiktak writes the file atomically — a half-written
    # file must never be readable as empty. See the tiktak-side atomicity test.
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()

    write_watchlist(watchlist, [])
    result = sup.reconcile()

    assert sorted(result.stopped) == ["alice", "bob"]
    assert all(w.stop_requested for w in spawned)


# --- §28, fixed at the source ----------------------------------------------


def test_dead_worker_in_watchlist_is_respawned(tmp_path):
    """§28: a monitor died from an uncaught exception and nobody respawned it —
    the recorder was blind to that user for 3.5h. The supervisor must notice and
    respawn *only* that user."""
    sup, _, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()
    alice, bob = spawned[0], spawned[1]

    alice.die()
    result = sup.reconcile()

    assert result.respawned == ["alice"]
    assert alice.joined is True  # the corpse is reaped, not leaked
    assert sup.workers["alice"] is not alice  # a genuinely new worker
    assert sup.workers["alice"].is_alive()
    # Healing @alice must not have touched @bob. Our current watchdog heals by
    # restarting ALL 14 monitors; that is exactly what this replaces.
    assert sup.workers["bob"] is bob
    assert bob.stop_requested is False


def test_live_worker_is_never_respawned(tmp_path):
    """The §29 hazard, inverted: respawning a worker that is actually still
    alive gives that user TWO recorders writing the same file concurrently ->
    corruption. Only genuinely dead workers may be respawned."""
    sup, _, spawned = make_supervisor(tmp_path, ["alice"])
    sup.reconcile()

    for _ in range(5):
        result = sup.reconcile()
        assert result.respawned == []

    assert len(spawned) == 1


def test_dead_worker_not_in_watchlist_is_reaped_not_respawned(tmp_path):
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice"])
    sup.reconcile()
    alice = spawned[0]

    write_watchlist(watchlist, [])
    alice.die()
    result = sup.reconcile()

    assert result.respawned == []
    assert result.reaped == ["alice"]
    assert len(spawned) == 1  # no resurrection
    assert "alice" not in sup.workers


# --- force stop: "stop recording @a NOW, mid-broadcast" ---------------------
#
# The watch-list is *desired state*; "stop now" is a *command*, so it gets its own
# file rather than being smuggled into the state file. The supervisor consumes it:
# read -> signal those workers -> delete. The worker interrupts its download loop
# but still flushes + converts (see test_worker_stop.py) — force means "stop
# promptly and finalize", never "kill".
#
# Normal use is paired: tiktak removes @a from the watch-list AND issues stop-now.
# The worker then interrupts the recording, converts it, returns to the top of its
# poll loop, sees it is no longer wanted, and exits.


def write_stop_now(tmp_path, users):
    (tmp_path / "stop_now.txt").write_text("".join(f"{u}\n" for u in users))


def test_force_stop_signals_only_the_named_worker(tmp_path):
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()
    before = {w.username: w for w in spawned}

    # tiktak: drop alice from the watch-list AND cut her recording immediately.
    write_watchlist(watchlist, ["bob"])
    write_stop_now(tmp_path, ["alice"])
    result = sup.reconcile()

    assert result.stopped == ["alice"]
    assert before["alice"].force_stopped is True
    # @bob is mid-broadcast and must not notice any of this.
    assert before["bob"].stop_requested is False
    assert before["bob"].force_stopped is False
    assert before["bob"].is_alive()


def test_graceful_removal_does_not_force(tmp_path):
    # The default path: remove without a stop-now command => the recording is
    # allowed to finish. force_stopped must stay False.
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice"])
    sup.reconcile()
    alice = spawned[0]

    write_watchlist(watchlist, [])
    sup.reconcile()

    assert alice.stop_requested is True
    assert alice.force_stopped is False


def test_stop_now_command_is_consumed_exactly_once(tmp_path):
    # A command file that isn't consumed would re-interrupt the *next* recording
    # every reconcile — a user could never record again.
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()

    write_stop_now(tmp_path, ["alice"])
    sup.reconcile()
    assert not (tmp_path / "stop_now.txt").exists()

    alice = next(w for w in spawned if w.username == "alice")
    alice.force_stopped = False  # if it fires again, we'll see it
    sup.reconcile()
    assert alice.force_stopped is False


def test_stop_now_for_an_unknown_user_is_ignored(tmp_path):
    # A stale or bogus command must not crash the supervisor — that would take
    # down every monitor on the box.
    sup, _, spawned = make_supervisor(tmp_path, ["alice"])
    sup.reconcile()

    write_stop_now(tmp_path, ["nobody"])
    result = sup.reconcile()

    assert result.stopped == []
    assert spawned[0].force_stopped is False


# --- safety: a bad watch-list must never take recording down ----------------


def test_unreadable_watchlist_changes_nothing(tmp_path):
    """If the watch-list disappears or can't be read, the safe move is to keep
    running exactly as we are. Treating it as "no users" would stop every
    recorder on the box because of a transient filesystem problem — a §28-class
    outage with extra steps."""
    sup, watchlist, spawned = make_supervisor(tmp_path, ["alice", "bob"])
    sup.reconcile()

    watchlist.unlink()
    result = sup.reconcile()

    assert result.stopped == [] and result.spawned == [] and result.respawned == []
    assert set(sup.workers) == {"alice", "bob"}
    assert all(w.stop_requested is False for w in spawned)
    assert all(w.is_alive() for w in spawned)
