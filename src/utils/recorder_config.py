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
