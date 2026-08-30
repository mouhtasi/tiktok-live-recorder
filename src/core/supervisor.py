"""Reconciling supervisor for the per-user monitor processes.

`main.run_recordings()` already spawns one process per user — it just never
supervises them. It calls `join()` and nothing else, which means:

* a monitor that dies is never respawned (the recorder goes blind for that user
  until the whole thing is restarted — the root cause of the 3.5h outage), and
* the watch-list is fixed at argv, so adding or removing a user requires
  restarting the recorder, truncating every recording in flight.

This module replaces fire-and-join with a reconcile loop against a watch-list
file. The file is the contract: it is atomic to swap, inspectable, survives
restarts, and keeps the recorder ignorant of whatever produced it.

Two things it must never do, both of which fail silently:

* respawn a worker that is **actually alive** — that gives one user two recorders
  writing the same output file concurrently, i.e. corruption; and
* stop everything because the watch-list could not be read — a filesystem blip
  must not take recording down.
"""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger_manager import logger

# A monitor that exits sooner than this after being spawned did not do any real
# work — it failed during start-up. That is the signature of the 2026-08-29
# storm, where every monitor died in one to two seconds and was rebuilt on the
# next 5-second pass, 119 at a time.
FAST_EXIT_SECONDS = 30

# The backoff schedule for repeat fast exits: 5s, 10s, 20s ... capped.
#
# The cap is not a tuning detail. There is no give-up threshold anywhere in this
# module, because a missed live cannot be recovered — an account nobody polls is
# an account nobody records. The policy is "retry forever, but at a rate that
# cannot flood", so the ceiling has to be short enough to still catch a
# broadcast and long enough to be harmless when 119 accounts are all failing.
BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 300

# How many monitors may be started in a single reconcile pass.
#
# Every monitor's first act is a DNS resolution plus two or three TikTok calls,
# so starting 119 at once is a synchronised burst. On 2026-08-29 a restart put
# 3,182 DNS queries/minute through pihole against a 1,000/minute limit; pihole
# answered REFUSED, the resolver read that as failure and retried, and the
# retries were themselves queries. At 20 per 5-second pass a full 119-account
# roster takes about 30 seconds to come up, which is nothing against a broadcast
# but is the difference between a burst and a ramp.
MAX_SPAWNS_PER_PASS = 20

# How many monitors must be starting together before their first polls are
# staggered.
#
# 🚨 A lone respawn must NOT be staggered. On 2026-08-30 @shellykimm was
# recording when the recorder was restarted; her replacement monitor drew a 292s
# stagger and the account went unwatched for nearly five minutes — with no
# benefit, because the other 118 monitors had kept their offsets and there was
# no herd to break up.
#
# The condition counts the batch rather than asking whether the supervisor is
# young, because a time-based rule gets two of the three cases wrong: it would
# leave a bulk watch-list addition unstaggered (50 new accounts in lockstep is
# the same volley, whenever it happens) and would stagger a crash-respawn that
# merely landed inside the window. What matters is how many first requests are
# about to arrive together, not why any one of them is starting.
STAGGER_BATCH_THRESHOLD = 5


@dataclass(frozen=True)
class WatchEntry:
    """One watch-list row: who to monitor, and optionally how often to poll.

    `interval_min is None` means the row gave no instruction, which the monitor
    resolves to its configured default. It is deliberately distinct from any
    numeric value: "no opinion" and "poll hourly" must never collapse together,
    because one of them is safe to guess and the other is not.
    """

    username: str
    interval_min: int | None = None


# A poll interval this long is certainly a bug in whatever wrote the file rather
# than an instruction: a day between liveness polls would miss every broadcast.
_MAX_INTERVAL_MIN = 24 * 60


def _parse_interval(token: str) -> int | None:
    """A row's interval, or None if it does not state a usable one.

    Every rejection resolves to None — i.e. "use the default" — and never to a
    slower value or an exception. The watch-list is machine-written, so junk here
    means the *writer* has a bug, and the safe response to our own bug is to keep
    polling at the normal rate. Zero and negatives are rejected for the same
    reason they look tempting: they are not "poll fast", they are "spin".
    """
    try:
        value = int(token)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value > _MAX_INTERVAL_MIN:
        return None
    return value


def read_watchlist_entries(path) -> list[WatchEntry]:
    """Parse a watch-list file: `username [interval_minutes]` per line.

    Blank lines, surrounding whitespace and `#` comments are ignored, and
    duplicates collapse — a user listed twice must not get two recorders (§29:
    two processes recording one account double-write the file). The first row for
    a username wins.

    The interval column is optional, so a plain one-name-per-line file — every
    version tiktak published before §58 — parses exactly as it always did.

    Raises FileNotFoundError if the file is absent. That is deliberate and the
    caller must not paper over it: an *empty* list means "record nobody", which
    is a legitimate instruction, while a *missing* file is an absence of
    information and must change nothing.
    """
    text = Path(path).read_text()

    entries: list[WatchEntry] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        username = parts[0]
        if username in seen:
            continue
        seen.add(username)
        entries.append(
            WatchEntry(
                username=username,
                interval_min=_parse_interval(parts[1]) if len(parts) > 1 else None,
            )
        )
    return entries


def read_watchlist(path) -> list[str]:
    """The watch-list as plain usernames.

    Still the parser for the *stop-now command file*, which has no interval
    column and must not grow one — "end @a's current recording" carries no
    schedule. Keeping this signature is also what leaves §37's supervisor and its
    tests untouched by §58.
    """
    return [e.username for e in read_watchlist_entries(path)]


@dataclass
class ReconcileResult:
    spawned: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    respawned: list[str] = field(default_factory=list)
    reaped: list[str] = field(default_factory=list)
    # Died too fast and is serving a backoff — wanted, deliberately not running.
    deferred: list[str] = field(default_factory=list)
    # Wanted and ready, but this pass's spawn budget was already spent.
    queued: list[str] = field(default_factory=list)

    def is_noop(self) -> bool:
        return not (
            self.spawned
            or self.stopped
            or self.respawned
            or self.reaped
            or self.deferred
        )


class RecorderSupervisor:
    """Owns one worker per watched username and keeps reality matching the file."""

    def __init__(
        self,
        watchlist_path,
        spawn_worker,
        stop_now_path=None,
        poll_interval: int = 5,
        *,
        fast_exit_seconds: int = FAST_EXIT_SECONDS,
        backoff_base: int = BACKOFF_BASE_SECONDS,
        backoff_max: int = BACKOFF_MAX_SECONDS,
        max_spawns_per_pass: int = MAX_SPAWNS_PER_PASS,
        health_path=None,
        now=time.monotonic,
        jitter=None,
    ):
        self.watchlist_path = Path(watchlist_path)
        self.stop_now_path = Path(stop_now_path) if stop_now_path else None
        self.spawn_worker = spawn_worker
        self.poll_interval = poll_interval

        self.fast_exit_seconds = fast_exit_seconds
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_spawns_per_pass = max_spawns_per_pass
        self.health_path = Path(health_path) if health_path else None
        # Injected so tests can drive time and randomness directly. A backoff
        # test that really slept would be slow and flaky, and one that could not
        # fix the jitter could not assert the schedule at all.
        self._now = now
        self._jitter = jitter or (lambda: random.uniform(0.5, 1.0))

        self.workers: dict[str, object] = {}
        # When each live worker was spawned, so a death can be classified as
        # "flapping" or "ordinary". Without a lifetime there is no difference
        # between a monitor that ran all night and one that never started, and
        # the supervisor treats both the same way — which is the bug.
        self._spawned_at: dict[str, float] = {}
        # Consecutive fast exits per user. Reset the moment a monitor proves it
        # can run, so the escalation cannot accumulate across healthy days.
        self._fast_exits: dict[str, int] = {}
        # Earliest monotonic time each user may be respawned.
        self._retry_after: dict[str, float] = {}
        self._respawns_total = 0
        # Counted separately from respawns, because the guard deliberately
        # decouples them: while a backoff is being served a monitor can die over
        # and over with no respawn at all. Reporting only respawns would show a
        # working guard and hide the failure it is working around.
        self._deaths_total = 0
        # Users we've asked to leave. They stay tracked until they actually exit:
        # the stop is cooperative, so a worker mid-broadcast keeps running until
        # it has finished writing its file. We must not ask twice, and must not
        # treat "still here" as "failed to stop".
        self._stopping: set[str] = set()

    # ── inputs ────────────────────────────────────────────────────────────────

    def _read_desired(self) -> list[str] | None:
        """The watch-list, or None meaning "no information — change nothing"."""
        try:
            return read_watchlist(self.watchlist_path)
        except FileNotFoundError:
            logger.warning(
                f"[!] Watch-list {self.watchlist_path} is missing — "
                "keeping the current monitors unchanged."
            )
            return None
        except OSError as e:
            logger.warning(
                f"[!] Watch-list {self.watchlist_path} could not be read ({e}) — "
                "keeping the current monitors unchanged."
            )
            return None

    def _consume_stop_now(self) -> list[str]:
        """Read and delete the force-stop command file.

        A command, not state — so it is consumed. Leaving it in place would
        re-interrupt every subsequent recording for those users, and they would
        never record again.
        """
        if self.stop_now_path is None:
            return []
        try:
            users = read_watchlist(self.stop_now_path)
        except FileNotFoundError:
            return []
        except OSError as e:
            logger.warning(f"[!] Could not read {self.stop_now_path}: {e}")
            return []

        try:
            self.stop_now_path.unlink()
        except OSError as e:
            logger.error(
                f"[!] Could not consume {self.stop_now_path} ({e}) — "
                "force-stop may repeat."
            )
        return users

    # ── the loop ──────────────────────────────────────────────────────────────

    def reconcile(self) -> ReconcileResult:
        result = ReconcileResult()

        # Force-stops are independent of the watch-list: "end @a's current
        # recording" is not the same instruction as "stop watching @a".
        for user in self._consume_stop_now():
            worker = self.workers.get(user)
            if worker is None:
                logger.warning(
                    f"[!] Force-stop for @{user}, who has no monitor — ignoring."
                )
                continue
            logger.info(f"Force-stopping the current recording for @{user}")
            worker.stop_recording_now()

        desired = self._read_desired()
        if desired is None:
            return result

        now = self._now()
        # Users whose monitor died during this pass and is wanted back. Tracked
        # so the spawn loop below can report a replacement as a respawn rather
        # than a fresh start — the two look identical at the point of spawning,
        # and only one of them is a symptom.
        replacing: set[str] = set()

        # A monitor that has been up longer than the fast-exit window has proved
        # it can run. Clear its escalation here, while it is still alive, rather
        # than when it eventually dies: otherwise an account that flapped once
        # this morning carries that penalty into every restart for the rest of
        # the day, and the guard quietly degrades the accounts it protects.
        for user, worker in self.workers.items():
            if self._fast_exits.get(user) and worker.is_alive():
                if now - self._spawned_at.get(user, now) >= self.fast_exit_seconds:
                    self._fast_exits.pop(user, None)

        # Reap the dead first, so a user who died can be respawned in the same
        # pass rather than waiting for the next one.
        for user, worker in list(self.workers.items()):
            if worker.is_alive():
                continue
            worker.join(timeout=5)
            del self.workers[user]
            self._stopping.discard(user)
            lifetime = now - self._spawned_at.pop(user, now)

            if user not in desired:
                # Retired on request, not a death. Deliberately not counted:
                # every watch-list edit would otherwise inflate the death rate
                # the status page is meant to alarm on.
                logger.info(f"Monitor for @{user} exited after {lifetime:.0f}s.")
                result.reaped.append(user)
                self._fast_exits.pop(user, None)
                self._retry_after.pop(user, None)
                continue

            # Nobody told this one to stop and it is still wanted: it crashed.
            self._deaths_total += 1
            if lifetime >= self.fast_exit_seconds:
                # An ordinary death after real work. Replace it at once —
                # delaying costs recordings and buys nothing.
                self._fast_exits.pop(user, None)
                self._retry_after.pop(user, None)
                replacing.add(user)
                logger.warning(
                    f"[!] Monitor for @{user} died after {lifetime:.0f}s — respawning it."
                )
                continue

            # Died during start-up, which is the storm's signature. Escalate.
            strikes = self._fast_exits.get(user, 0) + 1
            self._fast_exits[user] = strikes
            delay = self._backoff_delay(strikes)
            self._retry_after[user] = now + delay
            logger.warning(
                f"[!] Monitor for @{user} exited after only {lifetime:.0f}s "
                f"(strike {strikes}) — holding it back for {delay:.0f}s before "
                "retrying. Repeated fast exits mean start-up is failing, not the "
                "recording."
            )
            result.deferred.append(user)

        # Everyone wanted, missing, and off backoff. Computed before spawning
        # anything, because whether to stagger depends on how big this batch is —
        # a decision that cannot be made one user at a time.
        ready = [
            user
            for user in desired
            if user not in self.workers
            and not (
                (retry_at := self._retry_after.get(user)) is not None and now < retry_at
            )
        ]

        # Stagger only when enough monitors are starting together to form a
        # volley. The count includes users still queued behind the spawn budget:
        # a 119-account cold start arrives as six batches of 20, and every one of
        # those batches is a volley, so judging by this pass's slice alone would
        # leave the tail of the roster unstaggered.
        stagger = len(ready) > STAGGER_BATCH_THRESHOLD

        budget = self.max_spawns_per_pass
        for user in desired:
            if user in self.workers:
                continue
            retry_at = self._retry_after.get(user)
            if retry_at is not None and now < retry_at:
                if user not in result.deferred:
                    result.deferred.append(user)
                continue
            if budget <= 0:
                result.queued.append(user)
                continue

            is_replacement = (
                self._retry_after.pop(user, None) is not None or user in replacing
            )
            logger.info(
                f"{'Respawning' if is_replacement else 'Starting'} monitor for @{user}"
                f"{'' if stagger else ' (polling immediately — no batch to spread)'}"
            )
            self.workers[user] = self.spawn_worker(user, stagger=stagger)
            self._spawned_at[user] = now
            budget -= 1
            if is_replacement:
                self._respawns_total += 1
                result.respawned.append(user)
            else:
                result.spawned.append(user)

        for user in list(self.workers):
            if user in desired or user in self._stopping:
                continue
            logger.info(
                f"Stopping monitor for @{user} (it will finish any recording first)"
            )
            self.workers[user].request_stop()
            self._stopping.add(user)
            result.stopped.append(user)

        self._write_health(result, desired)
        return result

    # ── backoff ───────────────────────────────────────────────────────────────

    def _backoff_delay(self, strikes: int) -> float:
        """Exponential with a ceiling, then jittered downwards.

        Jitter matters more here than in a typical retry loop: when the cause is
        upstream — a WAF block, a DNS outage — all 119 monitors fail at the same
        instant and would otherwise carry identical schedules forever, turning a
        continuous flood into a periodic one of the same size.
        """
        raw = min(self.backoff_base * (2 ** (strikes - 1)), self.backoff_max)
        return raw * self._jitter()

    def retry_delay_for(self, user) -> float:
        """Seconds still to wait before `user` may be respawned. 0 if ready.

        Public because it is the honest way to test and to report the guard:
        the alternative is asserting on log lines, which is what made the last
        outage unobservable.
        """
        retry_at = self._retry_after.get(user)
        if retry_at is None:
            return 0
        return max(0, retry_at - self._now())

    # ── telemetry ─────────────────────────────────────────────────────────────

    def _write_health(self, result: ReconcileResult, desired: list[str]) -> None:
        """Publish supervision state for tiktak's status endpoint.

        The status page reported `healthy` / `in_sync: true` throughout the
        storm because the only thing it could see was the parent process, which
        was alive and extremely busy. Nothing outside this loop knows that a
        monitor died, so nothing outside this loop can report it.

        Written to a file for the same reasons the watch-list is a file, and
        wrapped because an unwritable path must never stop supervision — a
        measurement that can kill what it measures is worse than none.
        """
        if self.health_path is None:
            return
        try:
            payload = {
                "ts": time.time(),
                "watched": len(desired),
                "monitors_alive": sum(1 for w in self.workers.values() if w.is_alive()),
                "respawns_total": self._respawns_total,
                "deaths_total": self._deaths_total,
                "backing_off": {
                    user: round(self.retry_delay_for(user), 1)
                    for user in self._retry_after
                },
                "queued": len(result.queued),
            }
            tmp = self.health_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.health_path)
        except Exception as e:
            logger.warning(f"[!] Could not write health file {self.health_path}: {e}")

    def run_forever(self, reload_event=None) -> None:
        """Reconcile on a timer, or immediately when signalled.

        `reload_event` is set by the SIGHUP handler so a watch-list change lands
        in milliseconds instead of on the next poll.
        """
        while True:
            try:
                result = self.reconcile()
                if not result.is_noop():
                    logger.info(
                        f"Reconciled: +{result.spawned} -{result.stopped} "
                        f"respawned={result.respawned} reaped={result.reaped}"
                    )
            except Exception:
                # A supervisor that dies takes every monitor with it. Never let
                # one bad pass end the loop.
                logger.error("Error in supervisor reconcile pass", exc_info=True)

            if reload_event is not None:
                reload_event.wait(self.poll_interval)
                reload_event.clear()
            else:
                time.sleep(self.poll_interval)

    def stop_all(self) -> None:
        for user, worker in self.workers.items():
            worker.request_stop()
            self._stopping.add(user)
