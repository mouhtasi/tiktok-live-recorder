"""
Tests for per-account session cookies (tiktak §100).

@meena_kpsr's lives are age-restricted. Anonymously the room-info API answers
4003110 and the live page says "login required", so the monitor saw a room id
and never recorded. With the owner's logged-in session the same request returns
the stream (measured 2026-10-01, while she was live).

`cookies.json` is shared by every monitor, so putting the session there would
send it on ~1,000 polls an hour for 167 accounts. Instead tiktak marks single
rows `username interval cookies`, and only those monitors load the session from
`-session-cookies`. The tests below pin the two directions that matter: a
flagged monitor gets the session, and an unflagged one never does.
"""

from unittest.mock import patch

from core.supervisor import read_watchlist_entries
from core.tiktok_recorder import TikTokRecorder
from utils.utils import read_session_cookies

COOKIES_TXT = (
    "# Netscape HTTP Cookie File\n"
    ".tiktok.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tSECRET\n"
    ".tiktok.com\tTRUE\t/\tTRUE\t1893456000\ttt-target-idc\tuseast2a\n"
    ".example.com\tTRUE\t/\tFALSE\t1893456000\tother\tnope\n"
)


# --- the watch-list flag ------------------------------------------------------


def test_the_cookies_flag_is_read_from_the_third_column(tmp_path):
    p = tmp_path / "users.txt"
    p.write_text("alice 5 cookies\nbob 60\ncarol\n")
    entries = {e.username: e for e in read_watchlist_entries(p)}
    assert entries["alice"].session is True
    assert entries["alice"].interval_min == 5
    assert entries["bob"].session is False
    assert entries["carol"].session is False


# --- loading the session ------------------------------------------------------


def test_only_tiktok_cookies_are_loaded(tmp_path):
    p = tmp_path / "cookies.txt"
    p.write_text(COOKIES_TXT)
    assert read_session_cookies(str(p)) == {"sessionid": "SECRET", "tt-target-idc": "useast2a"}


def test_a_missing_session_file_loads_nothing(tmp_path):
    assert read_session_cookies(str(tmp_path / "absent.txt")) == {}
    assert read_session_cookies(None) == {}


# --- the monitor --------------------------------------------------------------


def _recorder(tmp_path, row, session_file=True):
    (tmp_path / "users.txt").write_text(row + "\n")
    if session_file:
        (tmp_path / "cookies.txt").write_text(COOKIES_TXT)
    rec = TikTokRecorder.__new__(TikTokRecorder)
    rec.user = "alice"
    rec.watchlist_path = tmp_path / "users.txt"
    rec._proxy = None
    rec._events_file = None
    rec._base_cookies = {"sessionid_ss": "", "tt-target-idc": ""}
    rec._cookies = rec._base_cookies
    rec._session_on = False
    rec._session_cookies_path = str(tmp_path / "cookies.txt")
    rec.tiktok = "anonymous client"
    return rec


def test_a_flagged_monitor_switches_to_the_session(tmp_path):
    rec = _recorder(tmp_path, "alice 5 cookies")
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    assert api.call_args.kwargs["cookies"]["sessionid"] == "SECRET"
    assert rec._session_on is True


def test_an_unflagged_monitor_never_loads_the_session(tmp_path):
    """🚨 The direction that protects the owner's account."""
    rec = _recorder(tmp_path, "alice 5")
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    api.assert_not_called()
    assert rec.tiktok == "anonymous client"
    assert "sessionid" not in rec._cookies


def test_removing_the_flag_switches_back_to_anonymous(tmp_path):
    rec = _recorder(tmp_path, "alice 5 cookies")
    with patch("core.tiktok_recorder.TikTokAPI"):
        rec._apply_session()
    (tmp_path / "users.txt").write_text("alice 5\n")
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    assert "sessionid" not in api.call_args.kwargs["cookies"]
    assert rec._session_on is False


def test_an_unreadable_watchlist_changes_nothing(tmp_path):
    """No instruction is not an instruction to drop the session mid-run."""
    rec = _recorder(tmp_path, "alice 5 cookies")
    with patch("core.tiktok_recorder.TikTokAPI"):
        rec._apply_session()
    (tmp_path / "users.txt").unlink()
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    api.assert_not_called()
    assert rec._session_on is True


def test_a_failing_session_switch_does_not_skip_the_poll(tmp_path):
    """🚨 Raised into automatic_mode()'s catch-all, a fault here would turn every
    poll into an error backoff — the monitor alive, and blind to every live."""
    import threading
    from test_adaptive_poll_interval import _make_recorder

    (tmp_path / "users.txt").write_text("alice 5 cookies\n")
    stop = threading.Event()
    rec = _make_recorder(tmp_path, user="alice", stop_event=stop)
    polled = []

    def poll():
        polled.append(True)
        stop.set()

    with patch.object(TikTokRecorder, "_apply_session", side_effect=RuntimeError("boom")), \
         patch.object(TikTokRecorder, "manual_mode", side_effect=poll), \
         patch.object(TikTokRecorder, "_stagger_first_poll"):
        rec.automatic_mode()

    assert polled == [True]


def test_a_flagged_monitor_without_a_session_file_stays_anonymous(tmp_path):
    rec = _recorder(tmp_path, "alice 5 cookies", session_file=False)
    with patch("core.tiktok_recorder.TikTokAPI") as api:
        rec._apply_session()
    api.assert_not_called()
    assert rec._session_on is False
