from dataclasses import dataclass

from utils.enums import Mode


@dataclass
class RecorderConfig:
    mode: Mode
    url: str | None = None
    user: str | None = None
    room_id: str | None = None
    automatic_interval: int = 5
    cookies: dict | None = None
    proxy: str | None = None
    output: str | None = None
    duration: int | None = None
    use_telegram: bool = False
    bitrate: str | None = None
    ffmpeg_path: str | None = None
    # Set by the supervisor to retire this worker. `stop_event` is read at the
    # poll boundary, so a worker that is mid-broadcast finishes writing its file
    # first. `stop_now_event` additionally interrupts the download loop — used to
    # cut a recording short on request, and it still flushes + converts what it
    # has. Neither is a kill.
    stop_event: object | None = None
    stop_now_event: object | None = None
    # The watch-list this monitor re-reads its own poll interval from (§58). None
    # in single-user runs, which just use `automatic_interval`.
    watchlist_path: str | None = None
    # Optional request-event sink, appended to by TikTokAPI on every outbound
    # call. None (the default) disables it entirely — see TikTokAPI._get().
    events_file: str | None = None
    # tiktak §100: a Netscape cookies.txt holding the owner's session, loaded only
    # by monitors whose watch-list row says `cookies`. None: nobody gets it.
    session_cookies_path: str | None = None
    # Whether this monitor delays its first poll by a random fraction of its
    # interval. Set by the supervisor, True only when enough monitors are
    # starting together to form a volley. 🚨 A lone respawn sets it False: there
    # is no herd to break up, and delaying it blinds the one account most likely
    # to have been recording a moment ago.
    stagger_first_poll: bool = True
    # §94: a shared multiprocessing.Value holding the wall-clock time by which
    # this monitor promises to act again. The supervisor ends a live worker that
    # is past it — the only way to see a monitor that is alive but stuck.
    deadline: object | None = None
