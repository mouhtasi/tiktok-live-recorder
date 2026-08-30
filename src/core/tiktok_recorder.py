import time
from http.client import HTTPException
from pathlib import Path
from threading import Thread

from requests import RequestException

from core.supervisor import read_watchlist_entries
from core.tiktok_api import TikTokAPI
from utils.logger_manager import logger
from utils.recorder_config import RecorderConfig
from utils.video_management import VideoManagement
from utils.custom_exceptions import LiveNotFound, UserLiveError, TikTokRecorderError
from utils.enums import Mode, Error, TimeOut, TikTokError


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
        self.duration = config.duration
        self.output = config.output
        self.bitrate = config.bitrate
        self.ffmpeg_path = config.ffmpeg_path
        self.use_telegram = config.use_telegram
        self._proxy = config.proxy
        self._cookies = config.cookies
        self._stop_event = config.stop_event
        self._stop_now_event = config.stop_now_event

    def should_stop(self) -> bool:
        """Retire this monitor at the next poll boundary (never mid-recording)."""
        return self._stop_event is not None and self._stop_event.is_set()

    def should_stop_now(self) -> bool:
        """End the *current recording* immediately (but still finalize it)."""
        return self._stop_now_event is not None and self._stop_now_event.is_set()

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
        """
        if self._stop_event is not None:
            self._stop_event.wait(seconds)
        else:
            time.sleep(seconds)

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
        setup_done = False

        while not self.should_stop():
            try:
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
                time.sleep(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)

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

    def start_recording(self, user, room_id):
        """
        Start recording live
        """
        live_url = self.tiktok.get_live_url(room_id, user=user)
        if not live_url:
            raise LiveNotFound(TikTokError.RETRIEVE_LIVE_URL)

        output = self._build_output_path(user)

        if self.duration:
            logger.info(f"Started recording for {self.duration} seconds ")
        else:
            logger.info("Started recording...")

        buffer_size = 512 * 1024  # 512 KB buffer
        buffer = bytearray()

        logger.info("[PRESS CTRL + C ONCE TO STOP]")
        with open(output, "wb") as out_file:
            stop_recording = False
            while not stop_recording:
                try:
                    if not self.tiktok.is_room_alive(room_id):
                        logger.info("User is no longer live. Stopping recording.")
                        break

                    start_time = time.time()
                    for chunk in self.tiktok.download_live_stream(live_url):
                        buffer.extend(chunk)
                        if len(buffer) >= buffer_size:
                            out_file.write(buffer)
                            buffer.clear()

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

                except ConnectionError:
                    if self.mode == Mode.AUTOMATIC:
                        logger.error(Error.CONNECTION_CLOSED_AUTOMATIC)
                        time.sleep(TimeOut.CONNECTION_CLOSED * TimeOut.ONE_MINUTE)

                except (RequestException, HTTPException) as ex:
                    logger.warning(f"Network hiccup, retrying: {ex}")
                    time.sleep(2)

                except KeyboardInterrupt:
                    logger.info("Recording stopped by user.")
                    stop_recording = True

                except Exception as ex:
                    logger.error(
                        f"Unexpected error during recording: {ex}",
                        exc_info=True,
                    )
                    stop_recording = True

                finally:
                    if buffer:
                        out_file.write(buffer)
                        buffer.clear()
                    out_file.flush()

        # A force-stop is one-shot: it ends *this* recording. If the user is
        # still in the watch-list, the monitor keeps polling — without this the
        # flag would cut every future recording too.
        if self._stop_now_event is not None:
            self._stop_now_event.clear()

        logger.info(f"Recording finished: {Path(output).resolve()}\n")
        VideoManagement.convert_flv_to_mp4(output, self.bitrate, self.ffmpeg_path)

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
