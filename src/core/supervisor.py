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

import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger_manager import logger


def read_watchlist(path) -> list[str]:
    """Parse a watch-list file: one username per line.

    Blank lines, surrounding whitespace and `#` comments are ignored, and
    duplicates collapse — a user listed twice must not get two recorders.

    Raises FileNotFoundError if the file is absent. That is deliberate and the
    caller must not paper over it: an *empty* list means "record nobody", which
    is a legitimate instruction, while a *missing* file is an absence of
    information and must change nothing.
    """
    text = Path(path).read_text()

    users: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in users:
            users.append(line)
    return users


@dataclass
class ReconcileResult:
    spawned: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    respawned: list[str] = field(default_factory=list)
    reaped: list[str] = field(default_factory=list)

    def is_noop(self) -> bool:
        return not (self.spawned or self.stopped or self.respawned or self.reaped)


class RecorderSupervisor:
    """Owns one worker per watched username and keeps reality matching the file."""

    def __init__(
        self,
        watchlist_path,
        spawn_worker,
        stop_now_path=None,
        poll_interval: int = 5,
    ):
        self.watchlist_path = Path(watchlist_path)
        self.stop_now_path = Path(stop_now_path) if stop_now_path else None
        self.spawn_worker = spawn_worker
        self.poll_interval = poll_interval

        self.workers: dict[str, object] = {}
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
                logger.warning(f"[!] Force-stop for @{user}, who has no monitor — ignoring.")
                continue
            logger.info(f"Force-stopping the current recording for @{user}")
            worker.stop_recording_now()

        desired = self._read_desired()
        if desired is None:
            return result

        # Reap the dead first, so a user who died can be respawned in the same
        # pass rather than waiting for the next one.
        for user, worker in list(self.workers.items()):
            if worker.is_alive():
                continue
            worker.join(timeout=5)
            del self.workers[user]
            self._stopping.discard(user)

            if user in desired:
                # Nobody told us this one to stop, and it's still wanted: it
                # crashed. Respawn just this user — the whole point, versus
                # restarting all N monitors to heal one.
                logger.warning(f"[!] Monitor for @{user} died — respawning it.")
                self.workers[user] = self.spawn_worker(user)
                result.respawned.append(user)
            else:
                logger.info(f"Monitor for @{user} exited.")
                result.reaped.append(user)

        for user in desired:
            if user not in self.workers:
                logger.info(f"Starting monitor for @{user}")
                self.workers[user] = self.spawn_worker(user)
                result.spawned.append(user)

        for user in list(self.workers):
            if user in desired or user in self._stopping:
                continue
            logger.info(f"Stopping monitor for @{user} (it will finish any recording first)")
            self.workers[user].request_stop()
            self._stopping.add(user)
            result.stopped.append(user)

        return result

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
