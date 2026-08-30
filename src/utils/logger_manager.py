import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Where a per-monitor log file lives, and how large it may get. Each monitor
# process owns its own file (see use_worker_log_file), so the ceiling is
# per-account: at ~119 accounts the worst case is roughly 119 * 2 MB. Kept small
# deliberately — with the setup-retry fix in place a healthy monitor logs almost
# nothing, and the box this runs on has under 50 GB free.
WORKER_LOG_DIR = os.environ.get("TIKTOK_RECORDER_LOG_DIR", "logs")
WORKER_LOG_MAX_BYTES = int(os.environ.get("TIKTOK_RECORDER_LOG_MAX_BYTES", 1024 * 1024))
WORKER_LOG_BACKUPS = int(os.environ.get("TIKTOK_RECORDER_LOG_BACKUPS", 1))

# A username reaches us from the watch-list file, so it is input rather than a
# constant. Anything outside this set is replaced before it becomes a filename.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]")
# Collapse runs of dots as well. A single dot is legitimate in a TikTok handle
# (@mei._.108), but ".." in a filename reads as traversal even where it cannot
# act as one, and a log path is a thing people paste into commands.
_DOT_RUN = re.compile(r"\.{2,}")


class MaxLevelFilter(logging.Filter):
    """
    Filter that only allows log records up to a specified maximum level.
    """

    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


class LoggerManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance.logger = None
            cls._instance.setup_logger()
        return cls._instance

    def setup_logger(self):
        if self.logger is None:
            self.logger = logging.getLogger("logger")
            self.logger.setLevel(logging.DEBUG)

            fmt_datefmt = "%Y-%m-%d %H:%M:%S"

            # 1) Console handler for INFO and WARNING, on stdout.
            #
            # The max level is WARNING, not INFO, and that single word is the
            # 2026-08-29 outage. With INFO here, and the handler below starting
            # at ERROR, WARNING fell between the two and reached no console at
            # all. The supervisor reports every monitor death with
            # logger.warning, so its 2,244 "died — respawning it" lines appeared
            # zero times in the 620 MB stdout log tiktak captures — the storm
            # was diagnosed hours later from `ps` and request_log instead.
            #
            # The two handlers are one partition of the level range. Widening
            # this half requires the other half to start exactly where this one
            # stops, or records duplicate.
            #
            # The stream is now named explicitly. It defaulted to sys.stderr
            # despite the comment here reading "(stdout)", so the documented
            # split did not exist.
            info_handler = logging.StreamHandler(sys.stdout)
            info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(
                logging.Formatter("[*] %(asctime)s - %(message)s", fmt_datefmt)
            )
            info_handler.addFilter(MaxLevelFilter(logging.WARNING))
            self.logger.addHandler(info_handler)

            # 2) Console ERROR handler (stderr) — the other half of the
            #    partition; starts one level above the filter above.
            error_handler = logging.StreamHandler(sys.stderr)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(
                logging.Formatter("[!] %(asctime)s - %(message)s", fmt_datefmt)
            )
            self.logger.addHandler(error_handler)

            # 3) File handler — DEBUG level, includes full stack traces
            #    Rotates at 5 MB, keeps 3 backups
            file_handler = RotatingFileHandler(
                "tiktok-recorder.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", fmt_datefmt
                )
            )
            self.logger.addHandler(file_handler)


def use_worker_log_file(username, log_dir=None):
    """Give this monitor process its own rotating log file.

    Every monitor is a separate OS process that inherits the parent's handlers,
    so without this all ~119 of them hold a RotatingFileHandler open on one
    path. Rotation is a rename: when one process rolls the file, the other 118
    keep writing to a descriptor that no longer has a name, and the next roll
    overwrites the backup the previous one just made. On prod this left
    tiktok-recorder.log.1/.2/.3 at **1,123 bytes each** — the log destroyed its
    own contents under exactly the load that produced them, which is why the
    respawn storm had to be reconstructed from `ps` and request_log.

    The console handlers are deliberately left alone. They write to the
    inherited stdout/stderr, which tiktak redirects into one file, and a short
    line written to an O_APPEND descriptor is atomic on POSIX — so the shared
    stream stays coherent and remains the single place to read the narrative.
    The per-user file carries the detail, without any writer but this one.

    Safe to call more than once; the previous rotating handler is replaced.
    """
    logger_ = LoggerManager().logger
    directory = Path(log_dir if log_dir is not None else WORKER_LOG_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    # The username comes from the watch-list, so it is untrusted input on a path.
    # Collapse anything that could traverse or escape before it becomes a name.
    safe = _UNSAFE_IN_FILENAME.sub("_", str(username))
    safe = _DOT_RUN.sub(".", safe).strip(".") or "unknown"
    path = directory / f"monitor-{safe}.log"

    for handler in list(logger_.handlers):
        if isinstance(handler, RotatingFileHandler):
            logger_.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        path,
        maxBytes=WORKER_LOG_MAX_BYTES,
        backupCount=WORKER_LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger_.addHandler(handler)
    return str(path)


logger = LoggerManager().logger
