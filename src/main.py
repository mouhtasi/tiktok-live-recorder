import sys
import os
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROC_TITLE_PREFIX = "tiktok-live-recorder"


def record_user(config):
    from core.tiktok_recorder import TikTokRecorder
    from utils.logger_manager import logger

    # Forked workers inherit the parent's command line, so without this every
    # monitor looks identical in `ps` and you cannot tell which process belongs
    # to which account — nor reliably distinguish ours from any other project's.
    _set_proc_title(f"{PROC_TITLE_PREFIX} [@{config.user}]")

    try:
        TikTokRecorder(config).run()
    except Exception as e:
        logger.error(f"{e}", exc_info=True)


def _set_proc_title(title):
    try:
        from setproctitle import setproctitle

        setproctitle(title)
    except ImportError:
        pass  # cosmetic only — never fail a recording over a process name


def _build_config(args, mode, cookies, user=None):
    from utils.recorder_config import RecorderConfig

    return RecorderConfig(
        url=args.url,
        user=user,
        room_id=args.room_id,
        mode=mode,
        automatic_interval=args.automatic_interval,
        cookies=cookies,
        proxy=args.proxy,
        output=args.output,
        duration=args.duration,
        use_telegram=args.telegram,
        bitrate=args.bitrate,
        ffmpeg_path=args.ffmpeg_path,
    )


def run_supervised(args, mode, cookies):
    """Automatic mode driven by a watch-list file.

    Same process model as run_recordings() — one monitor process per user — but
    the parent now *supervises* instead of only join()ing: it respawns a monitor
    that died (otherwise that user is silently unmonitored until the whole
    recorder is restarted) and it adds/removes individual monitors as the
    watch-list changes, so nobody else's recording is disturbed.
    """
    import signal
    import threading

    from core.supervisor import RecorderSupervisor
    from utils.logger_manager import logger

    reload_event = threading.Event()

    def _on_sighup(_signum, _frame):
        # Apply a watch-list change in milliseconds rather than on the next poll.
        reload_event.set()

    signal.signal(signal.SIGHUP, _on_sighup)

    def spawn(username):
        stop_event = multiprocessing.Event()
        stop_now_event = multiprocessing.Event()
        config = _build_config(args, mode, cookies, user=username)
        config.stop_event = stop_event
        config.stop_now_event = stop_now_event

        proc = multiprocessing.Process(
            target=record_user, args=(config,), name=f"{PROC_TITLE_PREFIX}[@{username}]"
        )
        proc.start()
        return _Worker(username, proc, stop_event, stop_now_event)

    supervisor = RecorderSupervisor(
        watchlist_path=args.watchlist,
        stop_now_path=args.stop_now_file,
        spawn_worker=spawn,
    )

    logger.info(f"Supervising watch-list {args.watchlist}")
    try:
        supervisor.run_forever(reload_event=reload_event)
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C detected — asking monitors to finish and exit.")
        supervisor.stop_all()
        for worker in supervisor.workers.values():
            worker.join(timeout=30)


class _Worker:
    """Handle for one per-user monitor process.

    Stopping is cooperative, never a kill: the worker holds the output file open
    and the bytes on disk are raw FLV until convert_flv_to_mp4() runs at the end,
    so terminating it mid-stream would leave an unconverted file behind.
    """

    def __init__(self, username, proc, stop_event, stop_now_event):
        self.username = username
        self.proc = proc
        self._stop_event = stop_event
        self._stop_now_event = stop_now_event

    def is_alive(self):
        return self.proc.is_alive()

    def request_stop(self):
        """Exit at the next poll boundary — finishing any recording in flight."""
        self._stop_event.set()

    def stop_recording_now(self):
        """End the current recording immediately (still flushed and converted)."""
        self._stop_now_event.set()

    def join(self, timeout=None):
        self.proc.join(timeout)

    @property
    def pid(self):
        return self.proc.pid


def run_recordings(args, mode, cookies):
    if isinstance(args.user, list):
        processes = []
        for user in args.user:
            config = _build_config(args, mode, cookies, user=user)
            p = multiprocessing.Process(target=record_user, args=(config,))
            p.start()
            processes.append(p)
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[!] Ctrl-C detected.")
            try:
                for p in processes:
                    p.join()
            except KeyboardInterrupt:
                print("\n[!] Forcefully terminating all processes.")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
    else:
        config = _build_config(args, mode, cookies, user=args.user)
        record_user(config)


def main():
    from utils.args_handler import validate_and_parse_args
    from utils.utils import read_cookies
    from utils.logger_manager import logger
    from utils.custom_exceptions import TikTokRecorderError
    from utils.dependencies import check_ffmpeg
    from check_updates import check_updates

    try:
        # validate and parse command line arguments
        args, mode = validate_and_parse_args()

        # check ffmpeg binary (supports custom path via -ffmpeg-path)
        check_ffmpeg(args.ffmpeg_path or "ffmpeg")

        # check for updates
        if args.update_check is True:
            logger.info("Checking for updates...\n")
            if check_updates():
                exit()
        else:
            logger.info("Skipped update check\n")

        # read cookies from the config file
        cookies = read_cookies()

        # run the recordings based on the parsed arguments
        if args.watchlist:
            run_supervised(args, mode, cookies)
        else:
            run_recordings(args, mode, cookies)

    except TikTokRecorderError as ex:
        logger.error(f"Application Error: {ex}")

    except Exception as ex:
        logger.critical(f"Generic Error: {ex}", exc_info=True)


if __name__ == "__main__":
    # print the banner
    from utils.utils import banner

    banner()

    # check and install dependencies
    from utils.dependencies import check_and_install_dependencies

    check_and_install_dependencies()

    # set up signal handling for graceful shutdown
    multiprocessing.freeze_support()

    # run
    main()
