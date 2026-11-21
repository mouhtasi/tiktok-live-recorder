"""
Tests for reporting a monitor that polls on time and never gets an answer
(tiktak §140).

On 2026-11-20 five monitors were 403'd on 90 %+ of their polls, up to 65 polls
in a row (21 hours). Every check stayed green: each monitor was alive, kept its
§94 deadline, and logged a line every 20 minutes. "Alive and on time" is not
"watching". Only the monitor knows how each poll ended, so it publishes its run
of polls with no clean answer, and the supervisor puts the blind ones in its
health file for tiktak's /status.

A clean answer is "not live" or a recording. Everything else — a WAF block, a
login-required live, an error, a live with no stream — is a poll that could
have missed a live.
"""

import json
import multiprocessing
import time
from unittest.mock import Mock, patch

import pytest

from core.supervisor import BLIND_REASONS, RecorderSupervisor
from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import LiveNotFound, UserLiveError
from utils.enums import TikTokError


def _recorder(blind=True):
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "alice"
    rec._blind = multiprocessing.RawArray("d", 3) if blind else None
    return rec


def _state(rec):
    return list(rec._blind)


NOT_LIVE = UserLiveError("@alice: The user is not hosting a live stream at the moment.")


def test_a_waf_poll_starts_a_run_and_a_clean_answer_ends_it():
    rec = _recorder()
    with patch("core.tiktok_recorder.time.time", return_value=1000.0):
        rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))
    with patch("core.tiktok_recorder.time.time", return_value=2200.0):
        rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))
    polls, since, reason = _state(rec)
    assert (polls, since) == (2, 1000.0)        # since = the FIRST blocked poll
    assert BLIND_REASONS[int(reason)] == "waf"

    rec._note_poll(NOT_LIVE)
    assert _state(rec) == [0, 0, 0]


def test_the_api_s_own_not_live_is_clean():
    """is_room_alive() raises the enum member, manual_mode() an f-string; both clean."""
    rec = _recorder()
    rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))
    rec._note_poll(UserLiveError(TikTokError.USER_NOT_CURRENTLY_LIVE))
    assert _state(rec)[0] == 0


def test_a_recording_is_clean():
    rec = _recorder()
    rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))
    rec._note_poll(None)
    assert _state(rec)[0] == 0


class _StopRecording(BaseException):
    pass


def test_the_poll_is_clean_once_the_stream_is_found():
    """Review round 1: marked clean only when the broadcast ENDED, an account
    recorded for three hours after one blocked poll read as blind the whole
    time, and /status warned about the account it was recording."""
    rec = _recorder()
    rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))
    rec.tiktok = Mock()
    rec.tiktok.get_live_url.return_value = "https://example/stream.flv"
    seen = []

    def after_url(*a, **k):
        seen.append(_state(rec)[0])
        raise _StopRecording()

    # Anything start_recording() touches after the URL stops the test there.
    type(rec.tiktok).last_stream_candidates = property(after_url)
    with pytest.raises(_StopRecording):
        rec.start_recording("alice", "room-1")
    assert seen == [0]


@pytest.mark.parametrize("ex, reason", [
    (UserLiveError(TikTokError.LIVE_RESTRICTION), "login_required"),
    (UserLiveError(TikTokError.ACCOUNT_PRIVATE), "no_answer"),
    (LiveNotFound(TikTokError.RETRIEVE_LIVE_URL), "no_answer"),
    (ConnectionError("Remote end closed connection"), "error"),
])
def test_every_other_ending_is_blind(ex, reason):
    rec = _recorder()
    rec._note_poll(ex)
    polls, _, code = _state(rec)
    assert polls == 1
    assert BLIND_REASONS[int(code)] == reason


def test_an_unsupervised_run_records_nothing():
    rec = _recorder(blind=False)
    rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))  # must not raise


def test_a_broken_counter_never_breaks_the_poll():
    """A measurement that can kill what it measures is worse than none."""
    rec = _recorder()
    rec._blind = Mock()
    rec._blind.__getitem__ = Mock(side_effect=RuntimeError("boom"))
    rec._note_poll(UserLiveError(TikTokError.WAF_BLOCKED))


# --- wired into the poll loop --------------------------------------------------


def test_automatic_mode_counts_each_poll(tmp_path):
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)
    rec._blind = multiprocessing.RawArray("d", 3)
    rec._session_on = False
    outcomes = iter([UserLiveError(TikTokError.WAF_BLOCKED),
                     UserLiveError(TikTokError.WAF_BLOCKED),
                     RuntimeError("tikrec down")])
    seen = []

    def poll():
        ex = next(outcomes)
        raise ex

    def wait(seconds):
        seen.append(int(rec._blind[0]))
        if len(seen) == 3:
            stop.set()

    with patch.object(TikTokRecorder, "_record_if_live", side_effect=poll), \
         patch.object(TikTokRecorder, "_apply_session"), \
         patch.object(TikTokRecorder, "_stagger_first_poll"), \
         patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=wait), \
         patch("core.tiktok_recorder.time.sleep", side_effect=wait):
        rec.automatic_mode()

    assert seen == [1, 2, 3]


# --- the supervisor publishes it ---------------------------------------------------


class _Worker:
    def __init__(self, username, state=None):
        self.username = username
        self._state = state

    def is_alive(self):
        return True

    def blind_state(self):
        return self._state

    def request_stop(self):
        pass

    def join(self, timeout=None):
        pass


def test_the_health_file_lists_blind_monitors(tmp_path):
    (tmp_path / "users.txt").write_text("alice\nbob\ncarol\n")
    states = {
        "alice": {"polls": 65, "since": 1_000.0, "reason": "waf"},
        "bob": None,                      # answering normally
    }

    class _OldWorker(_Worker):
        blind_state = None                # an older monitor without the method

    def spawn(username, stagger=True):
        if username == "carol":
            return _OldWorker(username)
        return _Worker(username, states.get(username))

    sup = RecorderSupervisor(watchlist_path=tmp_path / "users.txt",
                             spawn_worker=spawn, health_path=tmp_path / "h.json")
    sup.reconcile()
    health = json.loads((tmp_path / "h.json").read_text())
    assert health["blind"] == [{"user": "alice", "polls": 65, "since": 1_000.0,
                                "reason": "waf"}]


def _blocked_child(blind):
    blind[0], blind[1], blind[2] = 7, 1234.5, 1
    time.sleep(3600)


def test_a_real_child_s_run_reaches_the_parent(tmp_path):
    """Across a real process boundary, through the shared array."""
    from main import _Worker as RealWorker

    (tmp_path / "users.txt").write_text("alice\n")
    procs = []

    def spawn(username, stagger=True):
        blind = multiprocessing.RawArray("d", 3)
        proc = multiprocessing.Process(target=_blocked_child, args=(blind,))
        proc.start()
        procs.append(proc)
        return RealWorker(username, proc, multiprocessing.Event(),
                          multiprocessing.Event(), None, blind)

    sup = RecorderSupervisor(watchlist_path=tmp_path / "users.txt",
                             spawn_worker=spawn, health_path=tmp_path / "h.json")
    try:
        sup.reconcile()
        end = time.time() + 10
        blind = []
        while time.time() < end and not blind:
            time.sleep(0.2)
            sup.reconcile()
            blind = json.loads((tmp_path / "h.json").read_text()).get("blind")
        assert blind == [{"user": "alice", "polls": 7, "since": 1234.5, "reason": "waf"}]
    finally:
        for p in procs:
            p.kill()
            p.join(5)


# --- review round 2 -------------------------------------------------------------


class _TornArray(list):
    """Reads the count as 30, then the writer's reset lands: since and code are 0."""

    def __init__(self):
        super().__init__([30, 0.0, 0])
        self._reads = 0

    def __getitem__(self, i):
        if i == 0:
            self._reads += 1
            return 30 if self._reads == 1 else 0
        return super().__getitem__(i)


def test_a_torn_read_is_dropped_not_published():
    """Without the re-read the health file said {polls: 30, since: 0,
    reason: no_answer}: blind since 1970."""
    from main import _Worker as RealWorker
    w = RealWorker("alice", Mock(), Mock(), Mock(), None, _TornArray())
    assert w.blind_state() is None


def test_an_error_after_the_recording_is_not_blindness(tmp_path):
    """A live recorded for two hours whose remux then raised was listed blind."""
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)
    rec._blind = multiprocessing.RawArray("d", 3)
    rec._session_on = False

    def recorded_then_failed():
        rec._note_poll(None)            # what start_recording() does at the URL
        rec._poll_found_live = True
        raise OSError("remux failed")

    def wait(seconds):
        stop.set()

    with patch.object(TikTokRecorder, "_record_if_live", side_effect=recorded_then_failed), \
         patch.object(TikTokRecorder, "_apply_session"), \
         patch.object(TikTokRecorder, "_stagger_first_poll"), \
         patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=wait), \
         patch("core.tiktok_recorder.time.sleep", side_effect=wait):
        rec.automatic_mode()
    assert int(rec._blind[0]) == 0


def test_the_error_wait_is_interruptible(tmp_path):
    """The catch-all's 2-minute pause used time.sleep, so a retiring monitor
    waited it out; it now goes through the stop-aware wait."""
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)
    rec._session_on = False
    waits = []

    def wait(seconds):
        waits.append(seconds)
        stop.set()

    with patch.object(TikTokRecorder, "_record_if_live", side_effect=RuntimeError("x")), \
         patch.object(TikTokRecorder, "_apply_session"), \
         patch.object(TikTokRecorder, "_stagger_first_poll"), \
         patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=wait), \
         patch("core.tiktok_recorder.time.sleep", side_effect=lambda s: stop.set()) as slept:
        rec.automatic_mode()
    assert waits == [120]
    slept.assert_not_called()


def test_repeated_errors_wait_for_the_slot(tmp_path):
    """Review round 3 (pre-existing): a monitor whose polls keep failing polled
    every 2 minutes off its slot. One quick retry, then the slot."""
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 20\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, interval=20, user="alice", stop_event=stop)
    rec._session_on = False
    waits = []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            stop.set()

    now = 1_763_700_000.0
    with patch("core.tiktok_recorder.time.time", return_value=now), \
         patch.object(TikTokRecorder, "_record_if_live", side_effect=RuntimeError("x")), \
         patch.object(TikTokRecorder, "_apply_session"), \
         patch.object(TikTokRecorder, "_stagger_first_poll"), \
         patch.object(TikTokRecorder, "_wait_for_next_poll", side_effect=wait), \
         patch("core.tiktok_recorder.time.sleep", side_effect=lambda s: stop.set()):
        rec.automatic_mode()
        slot = rec._recheck_delay_s(20)
    assert waits == [120, slot, slot]
