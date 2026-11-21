import json
import os
import random
import time
import zlib
from http.client import HTTPException
from pathlib import Path
from threading import Thread

from requests import RequestException

from core.supervisor import (
    BLIND_ERROR, BLIND_LOGIN_REQUIRED, BLIND_NO_ANSWER, BLIND_WAF, read_watchlist_entries,
)
from core.tiktok_api import TikTokAPI
from utils.logger_manager import logger
from utils.recorder_config import RecorderConfig
from utils.utils import read_session_cookies
from utils.video_management import VideoManagement
from utils.custom_exceptions import LiveNotFound, UserLiveError, TikTokRecorderError
from utils.enums import Mode, Error, TimeOut, TikTokError

# Upper bound on the one-off delay before a monitor's first poll.
#
# Every monitor starts within ~30s of the recorder starting and nothing offsets
# their timers, so each §58 tier fires as one synchronised volley. Measured on
# prod 2026-08-30, minutes after the §76 deploy: the first 20 requests of a
# volley returned 200 and every one after them returned 403, recovering after a
# few minutes of quiet. A short-window burst threshold, not a ban — so the fix
# is to never present a burst.
#
# The cap binds before the poll interval for anything on a slow tier. A 60-minute
# account spread across a full hour could sit idle for an hour after a restart,
# and 🚨 a missed live is unrecoverable, so the spread is bounded by how long the
# recorder may stay blind rather than by the interval. Five minutes still thins
# 94 cold accounts to one request every ~3.2s, which is far inside budget.
#
# 🚨 tiktak §140 (2026-11-20): it was not inside budget for long. Every later
# poll inherited the stagger's phase, so the whole 20-minute tier (145 monitors)
# stayed inside 5 of its 20 minutes for as long as the recorder ran, and TikTok
# 403'd 15–28 % of the lookups there — the same few accounts, poll after poll,
# for up to 21 hours. The limit is a rate over minutes, not a burst over seconds.
# The stagger now only spreads the FIRST poll; every later poll waits for the
# account's own wall-clock slot (_recheck_delay_s), which covers the interval.
INITIAL_POLL_JITTER_MAX_S = 300


class TikTokRecorder:
    def __init__(self, config: RecorderConfig):
        self.tiktok = TikTokAPI(
            proxy=config.proxy,
            cookies=config.cookies,
            events_file=config.events_file,
            user=config.user,
        )

        self.url = config.url
        self.user = config.user
        self.room_id = config.room_id
        self.mode = config.mode
        self.automatic_interval = config.automatic_interval
        # Where to look up *this* monitor's own recheck interval (§58). Re-read at
        # every poll boundary rather than fixed here, so re-tiering an account
        # lands without respawning its monitor — a respawn truncates whatever it
        # is recording.
        self.watchlist_path = config.watchlist_path
        self.stagger_first_poll = getattr(config, "stagger_first_poll", True)
        self.duration = config.duration
        self.output = config.output
        self.bitrate = config.bitrate
        self.ffmpeg_path = config.ffmpeg_path
        self.use_telegram = config.use_telegram
        self._proxy = config.proxy
        self._cookies = config.cookies
        # §100: the owner's session, for this monitor only, when its watch-list
        # row says `cookies`. Starts anonymous; _apply_session() decides per poll.
        self._base_cookies = config.cookies
        self._session_cookies_path = getattr(config, "session_cookies_path", None)
        self._session_on = False
        self._events_file = config.events_file
        self._stop_event = config.stop_event
        self._stop_now_event = config.stop_now_event
        self._deadline = getattr(config, "deadline", None)
        self._blind = getattr(config, "blind", None)
        self._fleet_session = getattr(config, "fleet_session", None)

    # §94 — how long each kind of work may take before the supervisor may call
    # this monitor stuck. Generous on purpose: ending a working monitor costs a
    # poll (or truncates a recording), so a false "stuck" must be rare. A poll is
    # a handful of curl_cffi calls with 30s timeouts; a stall is bounded by the
    # stream's 60s read timeout plus a re-check; a conversion is a remux of a
    # file that can reach 5 GB.
    POLL_BUDGET_S = 600
    STALL_BUDGET_S = 600
    CONVERT_BUDGET_S = 2 * 3600
    # How often a waiting monitor re-reads its own interval: the hot tier, so a
    # re-tier to hot costs at most one hot poll of delay.
    RETIER_CHECK_S = 5 * 60
    # §140 — the shortest wait to the next slot, so an account never polls twice
    # in quick succession (a quarter of the interval for very short tiers).
    SLOT_GAP_MAX_S = 60
    # §140 — after this many session retries in a row that found nothing to
    # record, no retry for the cooldown, so a false "login required" on an
    # offline account cannot send the owner's session on every poll. A real
    # login-required live records on the first retry and never counts.
    SESSION_RETRY_STRIKES = 3
    SESSION_RETRY_COOLDOWN_S = 3600
    # ...and across ALL monitors at most this many retries per window. Real
    # login-required lives came from 3 accounts in 14 days; six in ten minutes
    # covers several at once and caps an event that hits every monitor.
    FLEET_SESSION_RETRY_BUDGET = 6
    FLEET_SESSION_RETRY_WINDOW_S = 600
    # §104 — pause after a stream pass that ended with zero bytes: 2, 4, 8, 16,
    # then 30 s. A pass that wrote bytes reconnects at once and resets it.
    EMPTY_PASS_BACKOFF_START_S = 2
    EMPTY_PASS_BACKOFF_CAP_S = 30
    # One log line for the first empty pass of a run, then every Nth.
    EMPTY_PASS_LOG_EVERY = 10
    # §104 — this much continuous silence ends the recording. A bound far above any
    # normal tail (the longest measured, 2026-10-01, was 348 s). Without it a room
    # whose room/info stays "live" holds its capture forever with a healthy
    # watchdog promise, and tiktak's sweep cannot take it: a 0-byte capture is "not
    # FLV" to it. It is NOT a pause detector — a stream that resumes sooner is
    # recorded into the same file.
    SILENCE_FLOOR_S = 15 * 60

    def _promise(self, seconds: float) -> None:
        """Publish "I will act again within `seconds`" for the supervisor.

        getattr, not the attribute: hand-built test doubles skip __init__, and
        an unsupervised run has no deadline at all. Both mean "no promise".
        """
        deadline = getattr(self, "_deadline", None)
        if deadline is not None:
            deadline.value = time.time() + seconds

    @staticmethod
    def _room_sidecar_path(output: str) -> Path:
        """`TK_<user>_<ts>_flv.mp4` → `TK_<user>_<ts>.room.json`."""
        return Path(output.replace("_flv.mp4", ".room.json"))

    @staticmethod
    def _capture_taken(output: str, out_file) -> bool:
        """True when `output` no longer names the file this monitor holds open.

        tiktak's ingest sweep treats a capture with no write for 300 s as an
        orphan, remuxes it and unlinks (or renames) it. During a broadcast's last
        minutes — TikTok still says "live", the stream gives nothing — a healthy
        monitor looks exactly like that. Had the stream resumed, every later byte
        would have gone to the unlinked file. An open handle keeps its inode from
        being recycled, so a path that names a different inode is a different file.
        """
        try:
            return os.stat(output).st_ino != os.fstat(out_file.fileno()).st_ino
        except OSError:
            return True

    def _pause_after_empty_pass(self, seconds: float) -> None:
        """Wait out an empty pass, but wake at once for a force-stop."""
        event = self._stop_now_event
        if event is not None:
            event.wait(seconds)
        else:
            time.sleep(seconds)

    def _report_pass(
        self, pass_bytes: int, seconds: float, empty_run: int, error: str | None = None
    ) -> None:
        """One line per stream pass: bytes, seconds, HTTP status, Content-Length.

        🚨 §104: a pass that ends cleanly with no bytes used to log nothing, so a
        recording's tail of ~100 empty passes (5.8 min on 2026-10-01) was
        invisible. Passes with bytes always log; an empty run logs its first pass
        and every Nth, so a long one cannot flood the log.
        """
        if pass_bytes == 0 and empty_run != 1 and empty_run % self.EMPTY_PASS_LOG_EVERY:
            return
        status = getattr(self.tiktok, "last_stream_status", None)
        length = getattr(self.tiktok, "last_stream_length", None)
        run = f", {empty_run} empty passes in a row" if empty_run > 1 else ""
        raised = f", ended by {error}" if error else ""
        logger.info(
            f"Stream pass ended: {pass_bytes} bytes in {seconds:.1f}s "
            f"(HTTP {status}, Content-Length {length if length is not None else 'unknown'}"
            f"{run}{raised})"
        )

    def should_stop(self) -> bool:
        """Retire this monitor at the next poll boundary (never mid-recording)."""
        return self._stop_event is not None and self._stop_event.is_set()

    def should_stop_now(self) -> bool:
        """End the *current recording* immediately (but still finalize it)."""
        return self._stop_now_event is not None and self._stop_now_event.is_set()

    def _apply_session(self) -> None:
        """Poll with the owner's session iff this monitor's row says `cookies`.

        tiktak §100. An age-restricted live answers 4003110 to an anonymous
        room-info request and "login required" on the live page, so the monitor
        sees a room id and never records (@meena_kpsr, 2026-10-01). The session
        is per monitor, never in the shared cookies.json, so the other monitors'
        ~1,000 polls an hour stay anonymous.

        Called only at a poll boundary, so a switch never lands mid-recording.
        An unreadable watch-list or an unlisted user is no instruction and
        changes nothing. A flagged row with no usable session file stays
        anonymous and says so, every poll, until the file is fixed.
        """
        if not self.watchlist_path:
            return
        try:
            entry = next((e for e in read_watchlist_entries(self.watchlist_path)
                          if e.username == self.user), None)
        except (OSError, ValueError):
            return
        if entry is None or entry.session == self._session_on:
            return

        cookies = dict(self._base_cookies or {})
        if entry.session:
            session = read_session_cookies(self._session_cookies_path)
            if not session:
                logger.warning(
                    f"[!] @{self.user} is marked for the owner's session, but "
                    f"{self._session_cookies_path!r} has no TikTok cookies — "
                    "polling anonymously"
                )
                return
            cookies.update(session)

        self.tiktok = TikTokAPI(
            proxy=self._proxy, cookies=cookies,
            events_file=self._events_file, user=self.user,
        )
        self._cookies = cookies
        self._session_on = entry.session
        logger.info(
            f"@{self.user}: "
            f"{'polling with' if entry.session else 'no longer using'} the owner's session"
        )

    def _poll_interval_minutes(self) -> int:
        """This monitor's recheck interval, re-read from the watch-list (§58).

        tiktak tiers accounts by their live history — 5 minutes for the 17 that
        actually stream, 60 for the 81 that never have — and publishes the number
        in the watch-list, because the tier is derived from a `lives` table this
        repo cannot and should not see. We only obey the number.

        🚨 **Every failure resolves to the configured default, never slower.** A
        missing file, a garbled row, an account not listed, a filesystem blip: all
        of them mean "no instruction", and the safe reading of no instruction is
        the normal poll rate. Biasing the other way would silently park a real
        streamer on an hourly poll, and a live we did not poll for is gone — there
        is no equivalent of re-walking a page to pick it up later.
        """
        if not self.watchlist_path:
            return self.automatic_interval
        try:
            for entry in read_watchlist_entries(self.watchlist_path):
                if entry.username == self.user:
                    return entry.interval_min or self.automatic_interval
        except (OSError, ValueError):
            # Includes FileNotFoundError. The supervisor treats a missing
            # watch-list as "change nothing"; for a monitor the equivalent is
            # "keep polling as before".
            return self.automatic_interval
        # Listed nowhere: this monitor is mid-retirement. Keep it at the default
        # until it actually exits rather than changing its rate on the way out.
        return self.automatic_interval

    def _wait_for_next_poll(self, seconds: float) -> None:
        """Wait out the recheck delay, but wake at once if asked to retire.

        🚨 This must not be a bare `time.sleep`. `automatic_mode()` reads the stop
        flag only at the loop top, so a plain sleep makes a cooperative stop (§37,
        de-listing an account) take up to a full interval to land. At the old
        global 5 minutes that was invisible; at §58's 60-minute cold tier it would
        leave a retiring monitor alive for an hour, with the status page showing
        "1 retiring" the whole time — §38b's cry-wolf failure, on a routine
        opt-out.

        Unsupervised runs (single-user manual mode) have no event and still sleep.

        The wait runs in slices of at most RETIER_CHECK_S, re-reading this
        monitor's interval between them, so a re-tier lands mid-wait. On
        2026-09-24 @yumehime555's first stream ended 05:41 and her monitor began
        a 60-minute cold wait; tiktak made her hot two minutes later, she went
        live again at 05:42:50, and we joined 26 minutes in. A stream that ends
        is often about to restart — that is exactly when the tier changes.
        """
        self._promise(seconds + self.POLL_BUDGET_S)
        if self._stop_event is None:
            time.sleep(seconds)
            return
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= seconds:
                return
            if self._stop_event.wait(min(seconds - elapsed, self.RETIER_CHECK_S)):
                return
            # 🚨 Cut only a wait longer than any slot wait of the current tier
            # (§140). A slot wait runs up to interval + gap; cut to the bare
            # interval it would end just before the slot, the next slot would
            # then be under `gap` away, and the account would poll twice a
            # minute apart, every cycle. A re-tier to a shorter interval still
            # cuts the wait to that interval, as §94 needs. That poll lands off
            # the new tier's slot, so the next wait can be as short as the gap
            # (60 s): one early extra poll per re-tier, accepted to keep §94's
            # "a promotion to hot polls within five minutes".
            interval_s = self._poll_interval_minutes() * TimeOut.ONE_MINUTE
            if seconds > interval_s + self._slot_gap_s(interval_s):
                seconds = interval_s

    def _slot_offset_s(self, interval_s: float) -> float:
        """Where inside each interval this account polls (§140).

        crc32 of the username, not hash(): hash() is salted per process, so its
        slot would move at every restart. The same name gives the same offset in
        every process, so a restart re-forms no herd.
        """
        return zlib.crc32(self.user.encode("utf-8")) / 2**32 * interval_s

    def _slot_gap_s(self, interval_s: float) -> float:
        return min(self.SLOT_GAP_MAX_S, interval_s / 4)

    def _recheck_delay_s(self, interval_min: float) -> float:
        """Seconds until this account's next poll slot, at least one gap away.

        tiktak §140. The old wait was the bare interval, so each cycle lasted the
        interval plus the poll's own time: the phases drifted, and they kept
        whatever bunching the start-up stagger left. A wall-clock slot ignores
        both. Result is in [gap, interval + gap). An unsupervised run has no herd
        and keeps the plain interval.
        """
        interval_s = interval_min * TimeOut.ONE_MINUTE
        if self.watchlist_path is None:
            return interval_s
        gap = self._slot_gap_s(interval_s)
        delay = (self._slot_offset_s(interval_s) - time.time()) % interval_s
        if delay < gap:
            delay += interval_s
        return delay

    def _setup(self):
        """Resolve user/room data and validate prerequisites via network calls."""
        self.check_country_blacklisted()

        if self.mode == Mode.FOLLOWERS:
            self.sec_uid = self.tiktok.get_sec_uid()
            if self.sec_uid is None:
                raise TikTokRecorderError("Failed to retrieve sec_uid.")

            logger.info("Followers mode activated\n")
        else:
            if self.url:
                self.user, self.room_id = self.tiktok.get_room_and_user_from_url(
                    self.url
                )

            if not self.user:
                self.user = self.tiktok.get_user_from_room_id(self.room_id)

            if not self.room_id:
                self.room_id = self.tiktok.get_room_id_from_user(self.user)

            logger.info(f"USERNAME: {self.user}" + ("\n" if not self.room_id else ""))
            if self.room_id:
                # Deliberately does NOT probe liveness here. This line used to
                # append "\n" when the room was offline, which cost a full
                # is_room_alive() request per monitor start — a network call made
                # solely to decide a log message's trailing whitespace. The
                # caller re-resolves liveness on the very next statement anyway,
                # so it was also a duplicate. At 119 monitors that was 119 wasted
                # requests per start, spent against the same WAF budget whose
                # refusals caused the 2026-08-29 respawn storm.
                logger.info(f"ROOM_ID:  {self.room_id}\n")

        # If proxy was used for the initial checks, switch to a direct connection
        # for the actual stream download to avoid proxy bottlenecks
        if self._proxy:
            self.tiktok = TikTokAPI(proxy=None, cookies=self._cookies)

    def run(self):
        """
        Resolves prerequisites and runs the recorder in the selected mode.

        If the mode is MANUAL, it checks if the user is currently live and
        if so, starts recording.

        If the mode is AUTOMATIC, it continuously checks if the user is live
        and if not, waits for the specified timeout before rechecking.
        If the user is live, it starts recording.

        if the mode is FOLLOWERS, it continuously checks the followers of
        the authenticated user. If any follower is live, it starts recording
        their live stream in a separate process.
        """
        if self.mode == Mode.AUTOMATIC:
            # NOT preceded by _setup(). Automatic mode runs its own setup from
            # inside the retry loop, so a network failure there is retried like
            # any other poll error instead of killing the monitor process.
            #
            # This is the 2026-08-29 respawn storm. _setup()'s room-id call
            # raised UserLiveError(WAF_BLOCKED) — a class automatic_mode()
            # already handles as "wait and try again" — but raised out here it
            # escaped run(), record_user() logged it, and the process exited.
            # The supervisor respawned all 119 monitors every 5s: ~63,000
            # requests/hour at TikTok, pihole's rate limiter wedged, load 38.
            # A retryable condition must not be raised where nothing retries.
            self.automatic_mode()
            return

        # Manual and followers mode keep the fail-fast setup. Both are one-shot
        # foreground invocations with a human reading the error; there is no
        # supervisor to turn a hard exit into a hot loop.
        self._setup()

        if self.mode == Mode.MANUAL:
            self.manual_mode()

        elif self.mode == Mode.FOLLOWERS:
            self.followers_mode()

    def manual_mode(self):
        if not self.tiktok.is_room_alive(self.room_id):
            raise UserLiveError(f"@{self.user}: {TikTokError.USER_NOT_CURRENTLY_LIVE}")

        self.start_recording(self.user, self.room_id)

    def _record_if_live(self):
        """One poll's check-and-record, retried once with the owner's session if
        the live needs a login (tiktak §140).

        §100's `cookies` flag is set by a human, and nothing told one to: from
        2026-11-15 one watched account's lives answered "Live is private, login
        required" and four were missed. Retrying only on that answer, only when
        a session file is configured, keeps §100's rule that the owner's session
        goes out for lives that need it and nothing else. The next poll's
        _apply_session() puts the client back to what the row says.
        """
        try:
            self.manual_mode()
        except UserLiveError as ex:
            restricted = bool(ex.args) and ex.args[0] is TikTokError.LIVE_RESTRICTION
            if not restricted:
                # "In a row" means in a row: any poll that reaches this check
                # without a restriction ends the run of fruitless retries (review
                # round 3 — three spread over weeks must not arm the cooldown
                # before a real live). A poll whose room-id lookup is WAF-blocked
                # never gets here and neither counts nor resets.
                self._fruitless_session_retries = 0
                raise
            if not self._use_session_for_this_poll():
                raise
            try:
                self.manual_mode()
            except (UserLiveError, LiveNotFound) as again:
                # 🚨 The session found nothing to record. LIVE_RESTRICTION is raised
                # for any room/info 4003110 whose page has no stream, and that path
                # decides liveness by check_alive, which §39 caught saying alive
                # for ended rooms — so a restriction is not proof of a live. But
                # a cooldown after ONE such retry cuts the start of a real live
                # that begins inside it (review round 2), and over 14 days of
                # logs "login required" fired only inside real lives. So: hold
                # back only after several fruitless retries in a row. A strike is
                # "not live" or the same wall again; a WAF block or a missing URL
                # says nothing about whether the session is needed.
                fruitless = isinstance(again, UserLiveError) and (
                    self._is_not_live(again)
                    or (again.args and again.args[0] is TikTokError.LIVE_RESTRICTION)
                )
                if fruitless:
                    self._fruitless_session_retries = (
                        getattr(self, "_fruitless_session_retries", 0) + 1
                    )
                if fruitless and self._fruitless_session_retries >= self.SESSION_RETRY_STRIKES:
                    self._fruitless_session_retries = 0
                    self._session_retry_after = (
                        time.monotonic() + self.SESSION_RETRY_COOLDOWN_S
                    )
                    logger.warning(
                        f"[!] @{self.user}: {self.SESSION_RETRY_STRIKES} session retries "
                        "in a row found nothing to record — no session retry for "
                        f"{self.SESSION_RETRY_COOLDOWN_S // 60} min"
                    )
                raise
            self._fruitless_session_retries = 0
        else:
            self._fruitless_session_retries = 0

    def _note_poll(self, ex) -> None:
        """Publish how this poll ended, for the supervisor's health file (§140).

        Clean (resets the run): a recording (`ex` None) or "not live". Anything
        else extends the run of polls that could have missed a live. Never
        raises: the measurement must not cost the poll.
        """
        blind = getattr(self, "_blind", None)
        if blind is None:
            return
        if ex is not None and getattr(self, "_poll_found_live", False):
            return  # this poll found and recorded a live; how it ended is not blindness
        try:
            code = self._blind_code(ex)
            if code == 0:
                blind[0], blind[1], blind[2] = 0, 0, 0   # polls first: a reader sees "clean"
                return
            # The count last: a reader that sees polls >= 1 must find its start
            # and reason already written (the array has no lock).
            if blind[0] < 1:
                blind[1] = time.time()
            blind[2] = code
            blind[0] += 1
        except Exception:
            pass

    @staticmethod
    def _is_not_live(ex) -> bool:
        """The clean "not live" answer: the enum from is_room_alive(), or
        manual_mode()'s f-string around it."""
        what = ex.args[0] if ex.args else ""
        return str(TikTokError.USER_NOT_CURRENTLY_LIVE) in str(what)

    @classmethod
    def _blind_code(cls, ex) -> int:
        """0 clean, else a BLIND_REASONS code."""
        if ex is None:
            return 0
        if isinstance(ex, UserLiveError):
            what = ex.args[0] if ex.args else ""
            if what is TikTokError.WAF_BLOCKED:
                return BLIND_WAF
            if what is TikTokError.LIVE_RESTRICTION:
                return BLIND_LOGIN_REQUIRED
            if cls._is_not_live(ex):
                return 0
            return BLIND_NO_ANSWER
        if isinstance(ex, LiveNotFound):
            return BLIND_NO_ANSWER
        return BLIND_ERROR

    def _use_session_for_this_poll(self) -> bool:
        """Switch to the owner's session for a login-required live; False if not possible."""
        if self.watchlist_path is None or getattr(self, "_session_on", False):
            return False
        if time.monotonic() < getattr(self, "_session_retry_after", 0.0):
            return False
        # A fault here must not replace the poll's own answer (the login wall)
        # with an "error", nor spend a unit of the fleet budget: build first,
        # take the budget last.
        try:
            session = read_session_cookies(getattr(self, "_session_cookies_path", None))
            if not session:
                return False
            cookies = dict(self._base_cookies or {})
            cookies.update(session)
            client = TikTokAPI(
                proxy=self._proxy, cookies=cookies,
                events_file=self._events_file, user=self.user,
            )
        except Exception as ex:
            logger.error(f"@{self.user}: could not build the session client ({ex!r})")
            return False
        if not self._fleet_allows_session():
            logger.warning(
                f"[!] @{self.user}: login-required live, but the fleet's session "
                f"budget ({self.FLEET_SESSION_RETRY_BUDGET} per "
                f"{self.FLEET_SESSION_RETRY_WINDOW_S // 60} min) is spent — polling anonymously"
            )
            return False
        self.tiktok = client
        self._cookies = cookies
        self._session_on = True
        self._session_for_poll = True
        logger.info(
            f"@{self.user}: this live needs a login — retrying this poll with the "
            "owner's session"
        )
        return True

    def _fleet_allows_session(self) -> bool:
        """Take one retry from the fleet-wide window, or False if it is spent.

        Fails open: on a broken shared array the per-monitor strikes still bound
        the session, and a guard fault must not cost a real live.
        """
        fleet = getattr(self, "_fleet_session", None)
        if fleet is None:
            return True
        try:
            with fleet.get_lock():
                now = time.time()
                if now - fleet[0] >= self.FLEET_SESSION_RETRY_WINDOW_S:
                    fleet[0], fleet[1] = now, 0
                if fleet[1] >= self.FLEET_SESSION_RETRY_BUDGET:
                    return False
                fleet[1] += 1
                return True
        except Exception:
            return True

    def _end_poll_session(self) -> None:
        """Drop a session that _use_session_for_this_poll() turned on, at the
        next poll boundary, whatever the watch-list says.

        🚨 Not left to _apply_session(): that changes nothing on an unreadable
        watch-list or an unlisted (retiring) user, so the owner's session would
        stay on for poll after poll of an account whose row never said
        `cookies`. A row that does say `cookies` turns it back on right after.
        """
        if not getattr(self, "_session_for_poll", False):
            return
        cookies = dict(self._base_cookies or {})
        self.tiktok = TikTokAPI(
            proxy=self._proxy, cookies=cookies,
            events_file=self._events_file, user=self.user,
        )
        self._cookies = cookies
        self._session_on = False
        # Last: if building the client raised, the flag stays set and the next
        # poll boundary tries again rather than keeping the session for good.
        self._session_for_poll = False

    def _stagger_first_poll(self):
        """Wait a random fraction of this monitor's interval before polling.

        Applied once, to the FIRST poll only. 🚨 §140: later polls do not
        inherit this offset any more — each waits for the account's own
        wall-clock slot (_recheck_delay_s). The stagger spans at most 5 minutes,
        so a 20-minute tier that inherited it stayed inside one 5-minute band
        for as long as the recorder ran, and TikTok 403'd its core.

        🚨 Only in supervised (watch-list) mode. A single-account CLI run has no
        herd, and making someone wait up to five minutes for
        `-user someone -mode automatic` would be a bug wearing a safeguard's
        clothes.

        The wait goes through _wait_for_next_poll, so it is interruptible: §37's
        stop is cooperative and read at poll boundaries, and an uninterruptible
        initial wait would make every watch-list removal take up to the cap and
        would hang stop_all() on shutdown.
        """
        if self.watchlist_path is None:
            return

        # 🚨 A lone respawn is not staggered. @shellykimm was recording when the
        # recorder restarted on 2026-08-30; her replacement drew a 292s stagger
        # and the account went unwatched for nearly five minutes, for nothing —
        # the other 118 monitors had kept their offsets. The supervisor decides
        # this, because only it can see how many are starting together.
        if not getattr(self, "stagger_first_poll", True):
            return

        interval_s = self._poll_interval_minutes() * TimeOut.ONE_MINUTE
        spread = min(interval_s, INITIAL_POLL_JITTER_MAX_S)
        delay = random.uniform(0, spread)

        logger.info(
            f"Staggering @{self.user}'s first poll by {delay:.0f}s so this tier "
            "does not poll in lockstep"
        )
        self._wait_for_next_poll(delay)

    def automatic_mode(self):
        # The stop flag is read HERE, at the poll boundary — never inside
        # start_recording(), which blocks for the whole broadcast. So a monitor
        # told to stop mid-stream finishes writing its file and exits after.
        #
        # Setup happens inside the loop rather than before it (see run()), but
        # still only once: it resolves things that do not change between polls,
        # and re-running its country probe every poll would add one request per
        # account per cycle to a budget this subsystem is already short of. The
        # flag flips only on success, so a monitor whose setup is failing keeps
        # retrying it and one whose setup succeeded never pays for it again.
        self._stagger_first_poll()

        setup_done = False

        while not self.should_stop():
            self._promise(self.POLL_BUDGET_S)
            self._poll_found_live = False
            try:
                # 🚨 Its own guard: a fault in the session switch must cost
                # the session, never the poll. Raised into the catch-all
                # below, it would skip this poll and every one after it.
                try:
                    self._end_poll_session()
                    self._apply_session()
                except Exception as ex:
                    logger.error(f"@{self.user}: session switch failed ({ex!r}) — polling as before")
                if not setup_done:
                    # _setup() has already resolved the room id, so re-resolving
                    # it here would spend a second request on an answer we hold.
                    # One duplicate per monitor start is 119 needless requests
                    # per recorder start, against the budget whose exhaustion is
                    # the whole reason this loop was restructured.
                    self._setup()
                    setup_done = True
                else:
                    self.room_id = self.tiktok.get_room_id_from_user(self.user)

                self._record_if_live()
                self._note_poll(None)
                self._error_streak = 0

            except (UserLiveError, LiveNotFound) as ex:
                self._error_streak = 0
                self._note_poll(ex)
                logger.info(ex)
                interval = self._poll_interval_minutes()
                delay = self._recheck_delay_s(interval)
                slot = f" (the {interval}-minute tier's slot)" if self.watchlist_path else ""
                logger.info(f"Waiting {delay / 60:.1f} minutes before recheck{slot}\n")
                self._wait_for_next_poll(delay)

            except Exception as ex:
                # Any other error during the poll/record cycle is transient from
                # the monitor's point of view and must NOT escape this loop. In
                # multi-user automatic mode each user runs in its own process
                # (see main.run_recordings) and the parent only joins children —
                # it never respawns a dead one. So an uncaught exception here
                # silently stops monitoring this user until the entire recorder
                # is restarted. The previous handler caught only (ConnectionError,
                # RequestException, HTTPException), which still misses curl_cffi
                # transport errors (the is_room_alive path uses curl_cffi, whose
                # exceptions don't derive from requests.RequestException),
                # JSON-decode errors, etc. Catch broadly, log, and retry.
                self._note_poll(ex)
                logger.error(
                    f"Recoverable error in automatic mode for @{self.user}, "
                    f"retrying after delay: {ex}"
                )
                # One quick retry for a transient fault; from the second error in
                # a row, the account's own slot (§140), so a monitor whose polls
                # keep failing does not poll every 2 minutes off its slot and put
                # back the load the slots spread out.
                # A fault at the end of a recording (remux, disk) is not a failing
                # poll: the next check of a stream that may restart stays quick.
                if getattr(self, "_poll_found_live", False):
                    self._error_streak = 0
                else:
                    self._error_streak = getattr(self, "_error_streak", 0) + 1
                delay = TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE
                if self._error_streak >= 2:
                    delay = self._recheck_delay_s(self._poll_interval_minutes())
                # Interruptible (§37's stop must not wait it out); it promises
                # the §94 deadline itself.
                self._wait_for_next_poll(delay)

        logger.info(f"Monitor for @{self.user} stopping as requested.")

    def followers_mode(self):
        active_recordings = {}  # follower -> Thread

        while True:
            try:
                followers = self.tiktok.get_followers_list(self.sec_uid)

                for follower in followers:
                    if follower in active_recordings:
                        if not active_recordings[follower].is_alive():
                            logger.info(f"Recording of @{follower} finished.")
                            del active_recordings[follower]
                        else:
                            continue

                    try:
                        room_id = self.tiktok.get_room_id_from_user(follower)

                        if not room_id or not self.tiktok.is_room_alive(room_id):
                            continue

                        logger.info(f"@{follower} is live. Starting recording...")

                        thread = Thread(
                            target=self.start_recording,
                            args=(follower, room_id),
                            daemon=True,
                        )
                        thread.start()
                        active_recordings[follower] = thread

                        time.sleep(2.5)

                    except TikTokRecorderError as e:
                        logger.error(f"Error while processing @{follower}: {e}")
                        continue

                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing @{follower}: {e}",
                            exc_info=True,
                        )
                        continue

                print()
                logger.info(
                    f"Waiting {self.automatic_interval} minutes for the next check..."
                )
                time.sleep(self.automatic_interval * TimeOut.ONE_MINUTE)

            except (UserLiveError, LiveNotFound) as ex:
                logger.info(ex)
                logger.info(
                    f"Waiting {self.automatic_interval} minutes before recheck\n"
                )
                time.sleep(self.automatic_interval * TimeOut.ONE_MINUTE)

            except (ConnectionError, RequestException, HTTPException):
                logger.error(Error.CONNECTION_CLOSED_AUTOMATIC)
                time.sleep(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)

    def _build_output_path(self, user: str) -> str:
        filename = (
            f"TK_{user}_{time.strftime('%Y.%m.%d_%H-%M-%S', time.localtime())}_flv.mp4"
        )
        if self.output:
            return str(Path(self.output) / filename)
        return filename

    def _write_room_sidecar(self, output: str, room_id) -> None:
        """Record when the broadcast began, beside the capture, for ingest.

        `TK_<user>_<ts>_flv.mp4` gets `TK_<user>_<ts>.room.json`, named for the
        *converted* mp4 so ingest finds it under the name it ingests — including
        an orphaned `_flv` capture that ingest converts itself. `room_created_at`
        is None when room/info did not say (the WAF page-scrape path): unknown
        must reach the page as unknown, never as a zero-minute join.

        🚨 Bookkeeping only. Every failure is logged and swallowed — the
        recording this describes matters more than the note about it.
        """
        created = getattr(self.tiktok, "last_room_created_at", None)
        if isinstance(created, bool) or not isinstance(created, int) or created <= 0:
            created = None
        joined = int(time.time())
        sidecar = self._room_sidecar_path(output)
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        try:
            tmp.write_text(json.dumps({
                "room_id": str(room_id),
                "room_created_at": created,
                "joined_at": joined,
            }))
            os.replace(tmp, sidecar)
        except OSError as ex:
            logger.warning(f"Could not write room sidecar {sidecar}: {ex}")
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        if created is not None:
            logger.info(
                f"Joined @{self.user}'s broadcast {(joined - created) / 60:.0f} "
                "minutes after it started"
            )

    def start_recording(self, user, room_id):
        """
        Start recording live
        """
        live_url = self.tiktok.get_live_url(room_id, user=user)
        if not live_url:
            raise LiveNotFound(TikTokError.RETRIEVE_LIVE_URL)
        # §140: the live is found and about to be recorded, so this poll is
        # clean now — not when the broadcast ends hours later, which would list
        # an account being recorded as blind for its whole live. The flag keeps
        # an error at the END of the recording (a remux, a disk fault) from
        # opening a blind run for an account that was just recorded.
        self._note_poll(None)
        self._poll_found_live = True

        # Every FLV the room offers, best first; `live_url` is the first. A
        # hand-built test double has no list here, and then there is one stream.
        streams = getattr(self.tiktok, "last_stream_candidates", None)
        if not isinstance(streams, list) or not streams:
            streams = [("stream", live_url)]
        stream_index = 0
        logger.info(
            f"Streams on offer for @{user}: "
            + ", ".join(label for label, _ in streams)
        )

        output = self._build_output_path(user)
        self._write_room_sidecar(output, room_id)

        if self.duration:
            logger.info(f"Started recording for {self.duration} seconds ")
        else:
            logger.info("Started recording...")

        buffer_size = 512 * 1024  # 512 KB buffer
        buffer = bytearray()

        logger.info("[PRESS CTRL + C ONCE TO STOP]")
        with open(output, "wb") as out_file:
            stop_recording = False
            adopted = False
            empty_run = 0
            last_bytes_at = time.monotonic()
            received_any = False
            paused_at = None
            while not stop_recording:
                self._promise(self.STALL_BUDGET_S)
                # Before any request, so a monitor in a long backoff notices the
                # sweep taking its capture without spending one more.
                if self._capture_taken(output, out_file):
                    logger.warning(self._adopted_message(output, 0))
                    adopted = True
                    break
                pass_started = time.monotonic()
                pass_bytes = 0
                pass_error = None
                # The status belongs to THIS pass. Left over from the last one, a
                # connect that fails would be reported as "HTTP 200".
                self.tiktok.last_stream_status = None
                self.tiktok.last_stream_length = None
                try:
                    if not self.tiktok.is_room_alive(room_id):
                        status = getattr(self.tiktok, "last_room_status", None)
                        if status != TikTokAPI.ROOM_PAUSED:
                            # The value is named because it was not on
                            # 2026-10-05, when a recording ended in the middle
                            # of a broadcast and nothing said which status did it.
                            shown = status if isinstance(status, int) else "unknown"
                            logger.info(
                                f"User is no longer live (room status {shown}). "
                                "Stopping recording."
                            )
                            break
                        # A pause is not an end. The file stays open and the
                        # stream is asked as usual: while it gives nothing, the
                        # §104 backoff paces the asking and the silence floor
                        # bounds it.
                        if paused_at is None:
                            paused_at = time.monotonic()
                            logger.info(
                                f"@{user} paused the broadcast (room status 3) — "
                                "keeping the recording open."
                            )
                    elif paused_at is not None:
                        logger.info(
                            f"@{user} resumed after "
                            f"{time.monotonic() - paused_at:.0f} s."
                        )
                        paused_at = None

                    start_time = time.time()
                    for chunk in self.tiktok.download_live_stream(live_url):
                        pass_bytes += len(chunk)
                        buffer.extend(chunk)
                        if len(buffer) >= buffer_size:
                            if self._capture_taken(output, out_file):
                                # The buffer is lost with the old file: it has no
                                # FLV header, so it cannot start the next one.
                                logger.warning(self._adopted_message(output, len(buffer)))
                                buffer.clear()
                                adopted = True
                                stop_recording = True
                                break
                            out_file.write(buffer)
                            buffer.clear()
                            # Bytes are arriving: this monitor is working (§94).
                            self._promise(self.STALL_BUDGET_S)

                        # Force-stop: break out of the download loop rather than
                        # being killed. Falling through leaves the `finally` to
                        # flush, the `with` to close, and convert_flv_to_mp4() to
                        # run — so the partial recording is kept and is a real
                        # mp4. A SIGKILL here would instead leave raw FLV bytes
                        # under a .mp4 name for ingest to misfile.
                        if self.should_stop_now():
                            logger.info(
                                f"Stop requested for @{user} — ending the recording "
                                "now and converting what we have."
                            )
                            stop_recording = True
                            break

                        elapsed_time = time.time() - start_time
                        if self.duration and elapsed_time >= self.duration:
                            stop_recording = True
                            break

                except ConnectionError as ex:
                    pass_error = type(ex).__name__
                    if self.mode == Mode.AUTOMATIC:
                        logger.error(Error.CONNECTION_CLOSED_AUTOMATIC)
                        time.sleep(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)

                except (RequestException, HTTPException) as ex:
                    pass_error = type(ex).__name__
                    logger.warning(f"Network hiccup, retrying: {ex}")
                    time.sleep(2)

                except KeyboardInterrupt:
                    pass_error = "KeyboardInterrupt"
                    logger.info("Recording stopped by user.")
                    stop_recording = True

                except Exception as ex:
                    pass_error = type(ex).__name__
                    logger.error(
                        f"Unexpected error during recording: {ex}",
                        exc_info=True,
                    )
                    stop_recording = True

                finally:
                    if buffer:
                        if self._capture_taken(output, out_file):
                            # Same loss as at a flush: report what was measured.
                            logger.warning(self._adopted_message(output, len(buffer)))
                            buffer.clear()
                            adopted = True
                            stop_recording = True
                        else:
                            out_file.write(buffer)
                            buffer.clear()
                    out_file.flush()

                # §104 — every pass that did not break out above ends here,
                # clean or raised. Prod runs Mode.MANUAL (no `-mode` on the
                # command line), where the builtin-ConnectionError handler is
                # silent and sleeps nothing, so a pass that raises must reach
                # the report and the backoff too or it can spin unseen.
                empty_run = 0 if pass_bytes else empty_run + 1
                self._report_pass(
                    pass_bytes, time.monotonic() - pass_started, empty_run, pass_error
                )
                # The chunk loop is the only other place a force-stop is read, so
                # an empty run never saw it — and the event's wait() returns at
                # once when set, which would turn the backoff back into a spin.
                if not stop_recording and self.should_stop_now():
                    logger.info(
                        f"Stop requested for @{user} — ending the recording "
                        "now and converting what we have."
                    )
                    stop_recording = True
                if pass_bytes:
                    last_bytes_at = time.monotonic()
                    received_any = True
                # 🚨 Only while the capture is empty. A reconnect to the same URL
                # appends a second FLV of the same rendition, which the remux
                # copes with; another rendition could change the codec in the
                # middle of the file.
                if (
                    not pass_bytes
                    and not received_any
                    and not stop_recording
                    and len(streams) > 1
                ):
                    stream_index = (stream_index + 1) % len(streams)
                    label, live_url = streams[stream_index]
                    logger.info(
                        f"@{user}: no data yet — trying stream "
                        f"{stream_index + 1}/{len(streams)} ({label})."
                    )
                if not pass_bytes and not stop_recording:
                    self._pause_after_empty_pass(min(
                        self.EMPTY_PASS_BACKOFF_START_S * 2 ** min(empty_run - 1, 5),
                        self.EMPTY_PASS_BACKOFF_CAP_S,
                    ))
                    silent_for = time.monotonic() - last_bytes_at
                    if silent_for >= self.SILENCE_FLOOR_S:
                        logger.warning(
                            f"@{user}: the stream has been silent for "
                            f"{silent_for / 60:.0f} min while the room still reads "
                            "live — ending this recording. The monitor polls again "
                            "at once."
                        )
                        break

        # A force-stop is one-shot: it ends *this* recording. If the user is
        # still in the watch-list, the monitor keeps polling — without this the
        # flag would cut every future recording too.
        if self._stop_now_event is not None:
            self._stop_now_event.clear()

        logger.info(f"Recording finished: {Path(output).resolve()}\n")
        if adopted:
            return  # already warned; the sweep archived the file, nothing to convert
        try:
            # One call, not exists() then getsize(): the sweep can act between two.
            size = os.path.getsize(output)
        except OSError:
            # With the pass-top check this means the sweep won a race in the last
            # instants. 🚨 Do not call convert: open(path, "ab") re-creates the
            # missing file as a 0-byte stub and ffmpeg then fails on it.
            logger.warning(
                f"Capture {Path(output).name} was already adopted by ingest — "
                "nothing left to convert."
            )
            return
        if size == 0:
            logger.info(
                f"Capture {Path(output).name} holds no bytes — removing it and "
                "its room sidecar instead of converting."
            )
            for stub in (Path(output), self._room_sidecar_path(output)):
                try:
                    stub.unlink()
                except OSError:
                    pass
            return
        self._promise(self.CONVERT_BUDGET_S)
        VideoManagement.convert_flv_to_mp4(output, self.bitrate, self.ffmpeg_path)

    @staticmethod
    def _adopted_message(output: str, lost_bytes: int) -> str:
        return (
            f"Capture {Path(output).name} was adopted while open (tiktak ingest "
            f"took it) — ending this recording; {lost_bytes} buffered bytes are "
            "lost with it. A new file starts on the next poll."
        )

    def check_country_blacklisted(self):
        is_blacklisted = self.tiktok.is_country_blacklisted()
        if not is_blacklisted:
            return False

        if self.room_id is None:
            raise TikTokRecorderError(TikTokError.COUNTRY_BLACKLISTED)

        if self.mode == Mode.AUTOMATIC:
            raise TikTokRecorderError(TikTokError.COUNTRY_BLACKLISTED_AUTO_MODE)

        elif self.mode == Mode.FOLLOWERS:
            raise TikTokRecorderError(TikTokError.COUNTRY_BLACKLISTED_FOLLOWERS_MODE)

        return is_blacklisted
