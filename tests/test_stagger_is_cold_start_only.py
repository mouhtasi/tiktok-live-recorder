"""
Tests that the first-poll stagger applies to a *batch* of monitors starting
together, and never to a lone respawn.

## The incident this comes from

The stagger was introduced to break up synchronised poll volleys, and it was
applied on every monitor start without distinction. Minutes later, on prod:

    07:20:01  @shellykimm got through the WAF, went live, started recording
    07:20:05  the recorder was restarted (to deploy the stagger)
    07:20:27  her replacement monitor drew a 292s stagger
    07:25:21  she was re-detected and recording resumed

Nearly five minutes blind on the one account that had been recording seconds
earlier — and for no benefit at all. 🚨 **A lone respawn has no herd to break
up.** The other 118 monitors were already de-phased and kept their offsets; the
one that restarted was the only one whose timing was not a problem.

So the stagger has to be conditional on *why* the monitor is starting.

## Why "how many are starting at once" and not "is the supervisor young"

A time-based rule ("stagger for the first N seconds of the supervisor's life")
gets the cold start right and the other two cases wrong: it would not stagger a
bulk watch-list addition an hour in — 50 new accounts polling in lockstep is the
same volley — and it *would* stagger a crash-respawn that happened to land
inside the window.

Counting the batch is a direct expression of the actual hazard: many monitors
about to make their first request at the same moment. It gets all three right.
"""

from core.supervisor import RecorderSupervisor, STAGGER_BATCH_THRESHOLD


class FakeWorker:
    def __init__(self, username, stagger):
        self.username = username
        self.stagger = stagger
        self._alive = True

    def is_alive(self):
        return self._alive

    def request_stop(self):
        pass

    def stop_recording_now(self):
        pass

    def join(self, timeout=None):
        pass

    def die(self):
        self._alive = False


class FakeClock:
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

    def spawn(username, stagger=True):
        w = FakeWorker(username, stagger)
        spawned.append(w)
        return w

    clock = FakeClock()
    kwargs.setdefault("now", clock)
    kwargs.setdefault("jitter", lambda: 1.0)
    sup = RecorderSupervisor(
        watchlist_path=watchlist,
        stop_now_path=tmp_path / "stop_now.txt",
        spawn_worker=spawn,
        **kwargs,
    )
    return sup, spawned, clock


def _latest(spawned, username):
    return [w for w in spawned if w.username == username][-1]


def test_a_cold_start_staggers_every_monitor(tmp_path):
    """The case the stagger exists for: a full roster polling in lockstep."""
    users = [f"user{i}" for i in range(30)]
    sup, spawned, _ = make_supervisor(tmp_path, users, max_spawns_per_pass=30)

    sup.reconcile()

    assert len(spawned) == 30
    assert all(w.stagger for w in spawned), "a cold start went unstaggered"


def test_a_lone_respawn_is_not_staggered(tmp_path):
    """@shellykimm's five blind minutes, prevented.

    One monitor restarting among 29 healthy ones has no herd to break up: the
    others kept their offsets, so delaying this one buys nothing and costs
    exactly the time an account that was recording seconds ago stays unwatched.
    """
    users = [f"user{i}" for i in range(30)]
    sup, spawned, clock = make_supervisor(
        tmp_path, users, max_spawns_per_pass=30, fast_exit_seconds=0
    )
    sup.reconcile()

    clock.advance(3600)  # everyone healthy for an hour
    _latest(spawned, "user7").die()
    sup.reconcile()

    replacement = _latest(spawned, "user7")
    assert replacement.stagger is False, "a lone respawn was needlessly delayed"


def test_a_single_new_account_is_not_staggered(tmp_path):
    """Adding one account to a running recorder is a respawn-shaped event.

    The existing monitors are already spread, so the newcomer can poll at once —
    and should, because the operator who just added it is watching for it.
    """
    users = [f"user{i}" for i in range(30)]
    sup, spawned, clock = make_supervisor(tmp_path, users, max_spawns_per_pass=30)
    sup.reconcile()

    clock.advance(3600)
    (tmp_path / "users.txt").write_text("".join(f"{u}\n" for u in users + ["newcomer"]))
    sup.reconcile()

    assert _latest(spawned, "newcomer").stagger is False


def test_a_bulk_addition_is_staggered(tmp_path):
    """Fifty accounts added at once is a volley, whenever it happens.

    This is the case a "stagger only while the supervisor is young" rule would
    get wrong, and it is why the condition counts the batch instead.
    """
    users = ["existing"]
    sup, spawned, clock = make_supervisor(tmp_path, users, max_spawns_per_pass=100)
    sup.reconcile()

    clock.advance(3600)
    newcomers = [f"new{i}" for i in range(50)]
    (tmp_path / "users.txt").write_text("".join(f"{u}\n" for u in users + newcomers))
    sup.reconcile()

    assert all(_latest(spawned, u).stagger for u in newcomers)


def test_the_threshold_is_the_boundary(tmp_path):
    """Just over the threshold staggers; at or under it does not."""
    over = [f"u{i}" for i in range(STAGGER_BATCH_THRESHOLD + 1)]
    sup, spawned, _ = make_supervisor(tmp_path, over, max_spawns_per_pass=100)
    sup.reconcile()
    assert all(w.stagger for w in spawned)

    under = [f"v{i}" for i in range(STAGGER_BATCH_THRESHOLD)]
    sup2, spawned2, _ = make_supervisor(tmp_path, under, max_spawns_per_pass=100)
    sup2.reconcile()
    assert not any(w.stagger for w in spawned2)


def test_a_budgeted_cold_start_still_staggers_every_batch(tmp_path):
    """The spawn budget splits a cold start across passes.

    Each of those passes is still a batch of 20 first-polls, so every one of
    them must stagger — otherwise the tail of a 119-account roster comes up
    unstaggered and re-creates the volley the budget was only ever smoothing.
    """
    users = [f"user{i}" for i in range(60)]
    sup, spawned, _ = make_supervisor(tmp_path, users, max_spawns_per_pass=20)

    for _ in range(3):
        sup.reconcile()

    assert len(spawned) == 60
    assert all(w.stagger for w in spawned)


def test_a_respawn_alongside_a_cold_start_batch_is_staggered(tmp_path):
    """A respawn that genuinely coincides with a batch is part of the volley.

    The rule is about how many first-polls are about to land together, not about
    the reason any individual one is starting.
    """
    users = [f"user{i}" for i in range(30)]
    sup, spawned, clock = make_supervisor(
        tmp_path, users, max_spawns_per_pass=5, fast_exit_seconds=0
    )
    sup.reconcile()  # spawns 5, leaves 25 queued

    clock.advance(3600)
    _latest(spawned, "user0").die()
    sup.reconcile()  # a big batch is still pending, so this pass is a volley

    assert _latest(spawned, "user0").stagger is True
