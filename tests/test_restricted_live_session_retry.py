"""
Tests for retrying a login-required live with the owner's session (tiktak §140).

§100 lets tiktak mark an account `cookies` so its monitor polls with the owner's
session. A human has to set that flag, and nothing tells them to. One watched
account had 17 archived lives; from 2026-11-15 her lives answered "Live is private,
login required" to the anonymous client, and four of them (40 min to 3 h) were
missed before anyone looked. Her row was never flagged.

Now an anonymous poll that hits the restriction retries THAT poll once with the
session, when a session file is configured. The next poll goes back to whatever
the watch-list row says, so the session is used only while such a live is on.

🚨 The direction that protects the owner's account: no other answer — not
live, WAF, a private ACCOUNT — ever loads the session.
"""

from unittest.mock import Mock, patch

import pytest

from core.tiktok_recorder import TikTokRecorder
from utils.custom_exceptions import UserLiveError
from utils.enums import TikTokError

COOKIES_TXT = (
    "# Netscape HTTP Cookie File\n"
    ".tiktok.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tSECRET\n"
)


def _recorder(tmp_path, row="alice 5", session_file=True, supervised=True):
    (tmp_path / "users.txt").write_text(row + "\n")
    if session_file:
        (tmp_path / "cookies.txt").write_text(COOKIES_TXT)
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "alice"
    rec.room_id = "room-1"
    rec.watchlist_path = (tmp_path / "users.txt") if supervised else None
    rec._proxy = None
    rec._events_file = None
    rec._base_cookies = {"tt-target-idc": ""}
    rec._cookies = rec._base_cookies
    rec._session_on = False
    rec._session_cookies_path = str(tmp_path / "cookies.txt")
    rec.tiktok = Mock(name="anonymous client")
    return rec


def _restricted():
    return UserLiveError(TikTokError.LIVE_RESTRICTION)


def test_a_restricted_live_is_retried_with_the_session(tmp_path):
    rec = _recorder(tmp_path)
    session_client = Mock(name="session client")
    calls = []

    def manual_mode():
        calls.append(rec.tiktok)
        if rec.tiktok is not session_client:
            raise _restricted()

    with patch("core.tiktok_recorder.TikTokAPI", return_value=session_client) as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=manual_mode):
        rec._record_if_live()

    assert api.call_args.kwargs["cookies"]["sessionid"] == "SECRET"
    assert calls[-1] is session_client
    assert len(calls) == 2
    assert rec._session_on is True


def test_the_next_poll_goes_back_to_anonymous(tmp_path):
    """The session is for this live only; an unflagged row drops it again."""
    rec = _recorder(tmp_path, row="alice 5")
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    assert rec._session_on is True

    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    assert "sessionid" not in api.call_args.kwargs["cookies"]
    assert rec._session_on is False


def test_the_retry_happens_once_per_poll(tmp_path):
    """A live the owner cannot see either must not loop inside one poll."""
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode",
                      side_effect=[_restricted(), _restricted()]) as mm, \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    assert mm.call_count == 2


def test_a_monitor_already_on_the_session_does_not_retry(tmp_path):
    rec = _recorder(tmp_path, row="alice 5 cookies")
    rec._session_on = True
    with patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=_restricted()) as mm, \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    api.assert_not_called()
    assert mm.call_count == 1


def test_no_session_file_means_no_retry(tmp_path):
    rec = _recorder(tmp_path, session_file=False)
    with patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=_restricted()) as mm, \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    api.assert_not_called()
    assert mm.call_count == 1


@pytest.mark.parametrize("message", [
    TikTokError.WAF_BLOCKED,
    TikTokError.ACCOUNT_PRIVATE,
    "@alice: The user is not hosting a live stream at the moment.",
])
def test_no_other_answer_loads_the_session(tmp_path, message):
    """🚨 The owner's session goes out only for a live that needs it."""
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode",
                      side_effect=UserLiveError(message)), \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    api.assert_not_called()
    assert rec._session_on is False


def test_an_unsupervised_run_does_not_retry(tmp_path):
    """A CLI run uses cookies.json as its user configured it."""
    rec = _recorder(tmp_path, supervised=False)
    with patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=_restricted()), \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    api.assert_not_called()


def test_automatic_mode_uses_the_retry(tmp_path):
    """Wired in: the poll loop records through _record_if_live, not manual_mode."""
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)

    def poll():
        stop.set()

    with patch.object(TikTokRecorder, "_record_if_live", side_effect=poll) as rif, \
         patch.object(TikTokRecorder, "_apply_session"), \
         patch.object(TikTokRecorder, "_stagger_first_poll"):
        rec.automatic_mode()
    assert rif.call_count == 1


# --- review round 1 -------------------------------------------------------------


NOT_LIVE = UserLiveError("@alice: The user is not hosting a live stream at the moment.")


def _fruitless_poll(rec, clock):
    """One poll: anonymous restriction, then the session says not live."""
    with patch("core.tiktok_recorder.time.monotonic", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), NOT_LIVE]), \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    rec._end_poll_session()
    clock["t"] += 5 * 60
    return api.call_count


def test_repeated_fruitless_retries_hold_the_session_back(tmp_path):
    """🚨 LIVE_RESTRICTION is raised for any 4003110 with no stream on the page,
    and that path decides liveness by check_alive (§39: alive for ended rooms).
    A false restriction on an offline account must not send the session on
    every poll: after three fruitless retries in a row, none for an hour."""
    rec = _recorder(tmp_path)
    clock = {"t": 1000.0}
    assert [_fruitless_poll(rec, clock) for _ in range(3)] == [1, 1, 1]

    with patch("core.tiktok_recorder.time.monotonic", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode", side_effect=_restricted()), \
         pytest.raises(UserLiveError):
        rec._record_if_live()
    api.assert_not_called()                          # held back

    clock["t"] += 61 * 60                            # past the hour: allowed again
    with patch("core.tiktok_recorder.time.monotonic", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    assert rec._session_on is True


def test_one_fruitless_retry_does_not_cost_the_next_live(tmp_path):
    """Review round 2: a cooldown armed by ONE fruitless retry would leave a
    live that starts five minutes later unrecorded for up to 55 minutes — the
    very loss this retry exists to stop."""
    rec = _recorder(tmp_path)
    clock = {"t": 1000.0}
    _fruitless_poll(rec, clock)

    with patch("core.tiktok_recorder.time.monotonic", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI") as api, \
         patch.object(TikTokRecorder, "manual_mode",
                      side_effect=[_restricted(), None]) as mm:
        rec._record_if_live()                         # no raise: it recorded
    assert api.call_count == 1 and mm.call_count == 2
    assert rec._session_on is True


def test_a_recorded_retry_resets_the_strikes(tmp_path):
    rec = _recorder(tmp_path)
    clock = {"t": 1000.0}
    _fruitless_poll(rec, clock)
    _fruitless_poll(rec, clock)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    assert rec._fruitless_session_retries == 0


def test_a_retry_that_records_sets_no_cooldown(tmp_path):
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    assert getattr(rec, "_session_retry_after", 0.0) == 0.0


def test_the_poll_session_ends_even_when_the_watchlist_is_unreadable(tmp_path):
    """🚨 _apply_session() changes nothing on an unreadable watch-list or an
    unlisted (retiring) user, so it cannot be what ends a per-poll session."""
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    (tmp_path / "users.txt").unlink()

    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._end_poll_session()
        rec._apply_session()
    assert "sessionid" not in api.call_args.kwargs["cookies"]
    assert rec._session_on is False


def test_a_flagged_row_keeps_its_session_after_a_poll_retry(tmp_path):
    """The per-poll session ends, and the row's own `cookies` turns it back on."""
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    (tmp_path / "users.txt").write_text("alice 5 cookies\n")
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._end_poll_session()
        rec._apply_session()
    assert api.call_args.kwargs["cookies"]["sessionid"] == "SECRET"
    assert rec._session_on is True


def test_the_loop_ends_the_poll_session_before_the_row_is_applied(tmp_path):
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)
    order = []

    with patch.object(TikTokRecorder, "_end_poll_session",
                      side_effect=lambda: order.append("end")), \
         patch.object(TikTokRecorder, "_apply_session",
                      side_effect=lambda: order.append("apply")), \
         patch.object(TikTokRecorder, "_record_if_live", side_effect=lambda: stop.set()), \
         patch.object(TikTokRecorder, "_stagger_first_poll"):
        rec.automatic_mode()
    assert order == ["end", "apply"]


# --- review round 3 -------------------------------------------------------------


def test_a_clean_poll_between_retries_resets_the_strikes(tmp_path):
    """Three fruitless retries weeks apart are not "in a row"; armed, the
    cooldown would cut the start of her next live."""
    rec = _recorder(tmp_path)
    clock = {"t": 1000.0}
    _fruitless_poll(rec, clock)
    _fruitless_poll(rec, clock)
    with patch.object(TikTokRecorder, "manual_mode", side_effect=NOT_LIVE), \
         pytest.raises(UserLiveError):
        rec._record_if_live()                       # an ordinary offline poll
    _fruitless_poll(rec, clock)
    assert rec._fruitless_session_retries == 1
    assert getattr(rec, "_session_retry_after", 0.0) == 0.0


def test_a_waf_block_on_the_retry_is_not_a_strike(tmp_path):
    rec = _recorder(tmp_path)
    for _ in range(4):
        with patch("core.tiktok_recorder.TikTokAPI"), \
             patch.object(TikTokRecorder, "manual_mode",
                          side_effect=[_restricted(), UserLiveError(TikTokError.WAF_BLOCKED)]), \
             pytest.raises(UserLiveError):
            rec._record_if_live()
        rec._end_poll_session()
    assert getattr(rec, "_fruitless_session_retries", 0) == 0
    assert getattr(rec, "_session_retry_after", 0.0) == 0.0


def test_the_fleet_budget_caps_every_monitor_together(tmp_path):
    """🚨 Review round 3: an event answering 4003110 everywhere (a WAF episode)
    would send the owner's session from every monitor; per-monitor strikes do
    not bound the fleet. One shared window does."""
    import multiprocessing

    fleet = multiprocessing.Array("d", 2)
    recs = []
    for i in range(8):
        d = tmp_path / f"m{i}"
        d.mkdir()
        rec = _recorder(d)
        rec.user = f"user{i}"
        (d / "users.txt").write_text(f"user{i} 5\n")
        rec._fleet_session = fleet
        recs.append(rec)

    clock = {"t": 50_000.0}
    with patch("core.tiktok_recorder.time.time", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI"):
        allowed = [r._use_session_for_this_poll() for r in recs]
    assert allowed == [True] * 6 + [False] * 2

    clock["t"] += 601                               # a new window
    with patch("core.tiktok_recorder.time.time", lambda: clock["t"]), \
         patch("core.tiktok_recorder.TikTokAPI"):
        assert recs[7]._use_session_for_this_poll() is True


def test_a_failed_client_build_keeps_the_poll_session_flag(tmp_path):
    """If building the anonymous client raises, the next boundary must try
    again — not keep the owner's session for the rest of the monitor's life."""
    rec = _recorder(tmp_path)
    with patch("core.tiktok_recorder.TikTokAPI"), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=[_restricted(), None]):
        rec._record_if_live()
    with patch("core.tiktok_recorder.TikTokAPI", side_effect=RuntimeError("init")), \
         pytest.raises(RuntimeError):
        rec._end_poll_session()
    assert rec._session_for_poll is True
    with patch("core.tiktok_recorder.TikTokAPI"):
        rec._end_poll_session()
    assert rec._session_on is False and rec._session_for_poll is False


def test_a_session_client_that_will_not_build_keeps_the_poll_s_answer(tmp_path):
    """Round 4: the build fault replaced the login wall with an "error" and
    spent a unit of the fleet budget."""
    import multiprocessing

    rec = _recorder(tmp_path)
    rec._fleet_session = multiprocessing.Array("d", 2)
    with patch("core.tiktok_recorder.TikTokAPI", side_effect=RuntimeError("init")), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=_restricted()), \
         pytest.raises(UserLiveError) as raised:
        rec._record_if_live()
    assert raised.value.args[0] is TikTokError.LIVE_RESTRICTION
    assert rec._fleet_session[1] == 0
    assert rec._session_on is False
