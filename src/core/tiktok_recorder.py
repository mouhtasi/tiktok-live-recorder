import json
import os
import random
import time
from http.client import HTTPException
from pathlib import Path
from threading import Thread

from requests import RequestException

from core.supervisor import read_watchlist_entries
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
            seconds = min(seconds, self._poll_interval_minutes() * TimeOut.ONE_MINUTE)

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

    def _stagger_first_poll(self):
        """Wait a random fraction of this monitor's interval before polling.

        Applied once, never per poll. Every later poll inherits the offset, so a
        monitor that starts 137s late stays 137s out of step with its tier for
        as long as it lives — the herd is broken up for one delay, paid once,
        rather than latency added to every cycle forever.

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
            try:
                # 🚨 Its own guard: a fault in the session switch must cost
                # the session, never the poll. Raised into the catch-all
                # below, it would skip this poll and every one after it.
                try:
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

                self.manual_mode()

            except (UserLiveError, LiveNotFound) as ex:
                logger.info(ex)
                interval = self._poll_interval_minutes()
                logger.info(f"Waiting {interval} minutes before recheck\n")
                self._wait_for_next_poll(interval * TimeOut.ONE_MINUTE)

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
                logger.error(
                    f"Recoverable error in automatic mode for @{self.user}, "
                    f"retrying after delay: {ex}"
                )
                delay = TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE
                self._promise(delay + self.POLL_BUDGET_S)
                time.sleep(delay)

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
