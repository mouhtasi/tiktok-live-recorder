"""
Tests for the log plumbing that hid the 2026-08-29 respawn storm.

## Why this file exists

The storm ran for hours and the operator's grep found nothing. Two independent
plumbing defects caused that, and neither is in the recording logic:

**1. WARNING reached no console handler.** ``LoggerManager`` installs a stdout
handler at INFO with ``MaxLevelFilter(logging.INFO)`` — i.e. *exactly* INFO —
and a stderr handler at ERROR. WARNING falls between them and is dropped by
both. The supervisor announces every respawn with ``logger.warning``, so the
single most diagnostic line in the system — "Monitor for @X died — respawning
it" — appeared **0 times** in the 620 MB stdout log tiktak captures, and 2,244
times only in a file nobody greps.

**2. 119 processes shared one RotatingFileHandler.** Each monitor is its own OS
process and each inherits a handler pointing at the same path. Rotation renames
the file underneath the other 118 writers, so the surviving backups on prod were
**1,123 bytes each** — the evidence destroyed itself under load, precisely when
there was most of it.

These are not cosmetic. Every other fix in this change set is unverifiable
without them: a backoff that silently fails to engage looks exactly like a
backoff that works.
"""

import io
import logging
from pathlib import Path

import pytest

from utils.logger_manager import LoggerManager


@pytest.fixture
def fresh_logger(tmp_path, monkeypatch):
    """A LoggerManager built from scratch, independent of the import-time one."""
    # logging.getLogger("logger") is a process-global singleton, and
    # setup_logger() *adds* handlers rather than replacing them. Dropping the
    # LoggerManager instance alone therefore leaves the import-time handlers
    # attached and stacks a second set on top — every record would then print
    # twice, and the double-print assertion below would fail for a reason that
    # exists only in the fixture.
    logging.getLogger("logger").handlers.clear()
    LoggerManager._instance = None
    # Keep the shared file handler out of the repo working tree.
    monkeypatch.chdir(tmp_path)
    mgr = LoggerManager()
    yield mgr.logger
    logging.getLogger("logger").handlers.clear()
    LoggerManager._instance = None


def _capture_console(logger):
    """Redirect the console handlers into buffers and return a reader.

    Deliberately not `capsys`. StreamHandler resolves its stream once, at
    construction, so the capture has to be installed on the handlers themselves;
    swapping sys.stdout/sys.stderr underneath an already-built handler does
    nothing, and pytest's replacement stream makes emit() fail outright — which
    surfaces as a "--- Logging error ---" traceback that happens to contain the
    probe string, i.e. a test that fails while telling you nothing true.
    """
    buffers = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            buf = io.StringIO()
            handler.setStream(buf)
            buffers.append(buf)
    return lambda: "".join(b.getvalue() for b in buffers)


def _accepting_handlers(logger, level):
    """Handlers that would actually emit a record at `level`."""
    record = logging.LogRecord(
        name="logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg="probe",
        args=(),
        exc_info=None,
    )
    return [
        h
        for h in logger.handlers
        if record.levelno >= h.level and all(f.filter(record) for f in h.filters)
    ]


# --- defect 1: the level gap ------------------------------------------------


@pytest.mark.parametrize(
    "level",
    [logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
    ids=["info", "warning", "error", "critical"],
)
def test_every_level_reaches_at_least_one_console_handler(fresh_logger, level):
    """No level may be silently dropped by the console pair.

    WARNING was the gap. It is parametrised across all four levels rather than
    asserted for WARNING alone, because the defect is a *coverage* one: the two
    handlers' ranges must tile without a hole, and a test naming only the hole
    we already found would not notice a new one opening at CRITICAL.
    """
    console = [
        h
        for h in _accepting_handlers(fresh_logger, level)
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert console, (
        f"level {logging.getLevelName(level)} reaches no console handler — "
        "it will be invisible in the log tiktak captures"
    )


def test_the_supervisors_respawn_warning_is_not_swallowed(fresh_logger):
    """The exact line whose absence cost the 2026-08-29 diagnosis."""
    read = _capture_console(fresh_logger)
    fresh_logger.warning("[!] Monitor for @lulu83245 died — respawning it.")

    assert "died — respawning it" in read()


def test_warning_and_error_do_not_double_print(fresh_logger):
    """Widening the stdout filter must not make ERROR print twice.

    The stdout handler's max-level filter and the stderr handler's min level are
    two halves of one partition; raising the former without respecting the
    latter turns every traceback into two, which is its own way of making a log
    unreadable.
    """
    read = _capture_console(fresh_logger)
    fresh_logger.error("a single error")

    assert read().count("a single error") == 1


# --- defect 2: shared rotation ----------------------------------------------


def test_a_worker_gets_its_own_log_file(fresh_logger, tmp_path):
    """Each monitor process must own its file, or rotation eats the evidence.

    119 processes rotating one path is what reduced prod's backups to 1,123
    bytes. The fix is not "rotate less" — it is that no two processes may share
    a rotating handler.
    """
    from utils.logger_manager import use_worker_log_file

    before = _rotating_paths(fresh_logger)
    use_worker_log_file("alice", log_dir=tmp_path)
    after = _rotating_paths(fresh_logger)

    assert len(after) == 1, "a worker must end up with exactly one rotating file"
    assert after != before, "the worker kept the shared file handler"
    assert "alice" in after[0], f"worker file is not per-user: {after[0]}"


def test_two_workers_never_share_a_file(fresh_logger, tmp_path):
    from utils.logger_manager import use_worker_log_file

    use_worker_log_file("alice", log_dir=tmp_path)
    alice = _rotating_paths(fresh_logger)[0]
    use_worker_log_file("bob", log_dir=tmp_path)
    bob = _rotating_paths(fresh_logger)[0]

    assert alice != bob


def test_a_worker_still_logs_to_the_console(fresh_logger, tmp_path):
    """Repointing the file must not detach the worker from the main stream.

    The per-user file is for detail. The narrative — including every traceback —
    must stay in the stdout/stderr tiktak captures, or the fix trades one
    invisible failure for another.
    """
    from utils.logger_manager import use_worker_log_file

    read = _capture_console(fresh_logger)
    use_worker_log_file("alice", log_dir=tmp_path)
    fresh_logger.error("worker blew up")

    assert "worker blew up" in read()


def test_worker_log_filenames_cannot_escape_the_log_directory(fresh_logger, tmp_path):
    """Usernames come from a watch-list file, so they are input, not constants.

    A name containing a path separator must not be able to steer the log file
    somewhere else on disk.
    """
    from utils.logger_manager import use_worker_log_file

    use_worker_log_file("../../etc/passwd", log_dir=tmp_path)
    path = Path(_rotating_paths(fresh_logger)[0]).resolve()

    # Containment is the property that matters, and it is asserted after
    # resolve() so a surviving ".." would have to actually escape to fail.
    assert path.parent == tmp_path.resolve(), f"log file escaped its directory: {path}"
    assert ".." not in path.name, f"traversal survived into the filename: {path.name}"


def _rotating_paths(logger):
    from logging.handlers import RotatingFileHandler

    return [
        h.baseFilename for h in logger.handlers if isinstance(h, RotatingFileHandler)
    ]
