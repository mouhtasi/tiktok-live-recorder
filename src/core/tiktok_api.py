import html
import json
import re

from http_utils.http_client import HttpClient
from utils.enums import StatusCode, TikTokError
from utils.logger_manager import logger
from utils.custom_exceptions import (
    UserLiveError,
    TikTokRecorderError,
    LiveNotFound,
    TikRecUnavailableError,
)


class TikTokAPI:
    def __init__(self, proxy, cookies):
        self.BASE_URL = "https://www.tiktok.com"
        self.WEBCAST_URL = "https://webcast.tiktok.com"
        self.API_URL = "https://www.tiktok.com/api-live/user/room/"
        self.EULER_API = "https://tiktok.eulerstream.com"
        self.TIKREC_API = "https://tikrec.com"

        self.http_client = HttpClient(proxy, cookies).req
        self._http_client_stream = HttpClient(proxy, cookies).req_stream

    def _is_authenticated(self) -> bool:
        response = self.http_client.get(f"{self.BASE_URL}/foryou")
        response.raise_for_status()

        content = response.text
        return "login-title" not in content

    def is_country_blacklisted(self) -> bool:
        """
        Checks if the user is in a blacklisted country that requires login
        """
        response = self.http_client.get(f"{self.BASE_URL}/live", allow_redirects=False)

        return response.status_code == StatusCode.REDIRECT

    def _check_alive_flag(self, room_id: str) -> bool:
        """The legacy liveness probe: webcast/room/check_alive's `alive` flag.

        Kept ONLY as a WAF-time fallback for is_room_alive(). As of 2026-07-15
        this flag is unreliable on its own — TikTok returns alive:true for
        long-ended rooms (room status 4) — so it must never be the sole signal.
        """
        data = self.http_client.get(
            f"{self.WEBCAST_URL}/webcast/room/check_alive/"
            f"?aid=1988&region=CH&room_ids={room_id}&user_is_login=true"
        ).json()

        if "data" not in data or len(data["data"]) == 0:
            return False

        return data["data"][0].get("alive", False)

    def is_room_alive(self, room_id: str) -> bool:
        """Return True only if the room is CURRENTLY broadcasting.

        Liveness is decided by room/info's `status` field (2 == ongoing/live,
        4 == ended), NOT by webcast/room/check_alive's `alive` flag. On
        2026-07-15 that flag began answering alive:true for rooms that had
        ENDED months earlier (e.g. a room created 2025-04-05, status 4). Because
        get_room_id_from_user() deliberately returns an offline user's *last*
        room_id, trusting `alive` made the recorder start phantom recordings of
        stale rooms — writing keepalive `pong` frames to disk forever and
        reporting every watched account as "live". room `status` cleanly
        separates a genuine live (2) from every stale room (4).

        Under a WAF challenge (status_code 4003110) room/info can't give us a
        status; we fall back to the legacy `alive` flag so get_live_url()'s
        page-scrape WAF path can still fire, rather than newly going blind.
        """
        if not room_id:
            raise UserLiveError(TikTokError.USER_NOT_CURRENTLY_LIVE)

        data = self.http_client.get(
            f"{self.WEBCAST_URL}/webcast/room/info/?aid=1988&room_id={room_id}"
        ).json()

        if data.get("status_code") == 4003110:  # WAF block — no status available
            return self._check_alive_flag(room_id)

        return (data.get("data") or {}).get("status") == 2

    def get_sec_uid(self):
        """
        Returns the sec_uid of the authenticated user.
        """
        response = self.http_client.get(f"{self.BASE_URL}/foryou")

        sec_uid = re.search('"secUid":"(.*?)",', response.text)
        if sec_uid:
            sec_uid = sec_uid.group(1)

        return sec_uid

    def get_user_from_room_id(self, room_id) -> str:
        """
        Given a room_id, I get the username
        """
        data = self.http_client.get(
            f"{self.WEBCAST_URL}/webcast/room/info/?aid=1988&room_id={room_id}"
        ).json()

        if "Follow the creator to watch their LIVE" in json.dumps(data):
            raise UserLiveError(TikTokError.ACCOUNT_PRIVATE_FOLLOW)

        if "This account is private" in data:
            raise UserLiveError(TikTokError.ACCOUNT_PRIVATE)

        display_id = data.get("data", {}).get("owner", {}).get("display_id")
        if display_id is None:
            raise TikTokRecorderError(TikTokError.USERNAME_ERROR)

        return display_id

    def get_room_and_user_from_url(self, live_url: str):
        """
        Given a url, get user and room_id.
        """
        response = self.http_client.get(live_url, allow_redirects=False)
        content = response.text

        if response.status_code == StatusCode.REDIRECT:
            raise UserLiveError(TikTokError.COUNTRY_BLACKLISTED)

        if response.status_code == StatusCode.MOVED:  # MOBILE URL
            matches = re.findall("com/@(.*?)/live", content)
            if len(matches) < 1:
                raise LiveNotFound(TikTokError.INVALID_TIKTOK_LIVE_URL)

            user = matches[0]

        # https://www.tiktok.com/@<username>/live
        match = re.match(r"https?://(?:www\.)?tiktok\.com/@([^/]+)/live", live_url)
        if match:
            user = match.group(1)

        room_id = self.get_room_id_from_user(user)

        return user, room_id

    def _old_get_room_id_from_user(self, user: str) -> str:
        params = {"uniqueId": user, "giftInfo": "false"}

        response = self.http_client.get(
            f"{self.EULER_API}/webcast/room_info",
            params=params,
            headers={"x-api-key": ""},
        )

        if response.status_code != 200:
            raise UserLiveError(TikTokError.ROOM_ID_ERROR)

        data = response.json()

        room_id = data.get("data", {}).get("room_info", {}).get("id")
        if not room_id:
            raise UserLiveError(TikTokError.ROOM_ID_ERROR)

        return room_id

    def _tikrec_get_room_id_signed_url(self, user: str) -> str:
        try:
            response = self.http_client.get(
                f"{self.TIKREC_API}/tiktok/room/api/sign",
                params={"unique_id": user},
            )
            response.raise_for_status()
        except Exception as e:
            raise TikRecUnavailableError(
                f"tikrec signing service is unreachable: {e}"
            ) from e

        try:
            data = response.json()
        except ValueError as e:
            raise TikRecUnavailableError(
                "tikrec signing service returned an invalid response "
                "(expected JSON, got something else — the service may be down)."
            ) from e

        signed_path = data.get("signed_path")
        if not signed_path:
            raise TikRecUnavailableError(
                "tikrec signing service did not return a signed_path "
                "(the service may be down or overloaded)."
            )

        return f"{self.BASE_URL}{signed_path}"

    def _direct_get_room_id_from_user(self, user: str) -> str | None:
        """Resolve a room_id straight from TikTok's public api-live endpoint,
        skipping the tikrec signing hop.

        Used as a fallback when tikrec is unavailable. This endpoint currently
        answers *unsigned* (no X-Bogus/_signature needed), which is why it works
        as a drop-in — but it is more prone to WAF "Please wait" challenges under
        heavy automated polling from a datacenter IP, so it stays a fallback
        rather than the primary path. It returns a roomId even when the user is
        offline (liveness is decided separately by is_room_alive), matching the
        tikrec path's contract.
        """
        response = self.http_client.get(
            self.API_URL,
            params={"aid": "1988", "sourceType": "54", "uniqueId": user},
        )
        content = response.text

        if not content or "Please wait" in content:
            raise UserLiveError(TikTokError.WAF_BLOCKED)

        try:
            data = response.json()
        except ValueError as e:
            raise UserLiveError(TikTokError.ROOM_ID_ERROR) from e

        return (data.get("data") or {}).get("user", {}).get("roomId")

    def _signed_get_room_id_from_user(self, user: str) -> str | None:
        """Resolve a room_id through the tikrec-signed URL (the primary path).

        Raises TikRecUnavailableError if the tikrec path fails to produce a
        usable answer — including when the *signed fetch* comes back non-JSON,
        which means the signed URL itself was bad, not that TikTok is down. A
        WAF block is a different thing (our IP, not tikrec) and propagates as
        UserLiveError.
        """
        signed_url = self._tikrec_get_room_id_signed_url(user)

        response = self.http_client.get(signed_url)
        content = response.text

        if not content or "Please wait" in content:
            raise UserLiveError(TikTokError.WAF_BLOCKED)

        try:
            data = response.json()
        except ValueError as e:
            raise TikRecUnavailableError(
                "tikrec's signed URL returned an invalid response "
                "(expected JSON, got something else)."
            ) from e

        return (data.get("data") or {}).get("user", {}).get("roomId")

    def get_room_id_from_user(self, user: str) -> str | None:
        """Given a username, get the room_id.

        Primary path is the tikrec signing service. tikrec is a free third-party
        SPOF that has taken all recording down before (Cloudflare 522, 2026-07-12),
        so *any* failure to resolve a room_id through it — unreachable, junk
        response, no signed_path, or a signed fetch that yields no roomId — falls
        back to TikTok's public api-live endpoint, which currently answers
        unsigned. The fallback is more WAF-prone under heavy automated load, so
        it stays a fallback rather than the primary path.

        Note this deliberately does NOT use upstream's _old_get_room_id_from_user():
        that calls eulerstream with an empty API key and answers HTTP 401
        ("requires the Webcast Premium add-on"), so as a fallback it is a no-op
        that would leave us blind exactly when tikrec is down.
        """
        try:
            room_id = self._signed_get_room_id_from_user(user)
            if room_id:
                return room_id
            reason = "tikrec resolved no roomId for this user"
        except TikRecUnavailableError as e:
            reason = str(e)

        logger.warning(
            f"[!] tikrec did not resolve a room-id for @{user} ({reason}). "
            "Falling back to the direct TikTok api-live endpoint — recording "
            "continues but may be more WAF-prone under heavy load."
        )
        return self._direct_get_room_id_from_user(user)

    def get_followers_list(self, sec_uid) -> list:
        """
        Returns all followers for the authenticated user by paginating
        """
        followers = []
        cursor = 0
        has_more = True

        ms_token = self.http_client.get(
            f"{self.BASE_URL}/api/user/list/?"
            "WebIdLastTime=1747672102&aid=1988&app_language=it-IT&app_name=tiktok_web&"
            "browser_language=it-IT&browser_name=Mozilla&browser_online=true&"
            "browser_platform=Linux%20x86_64&"
            "browser_version=5.0%20%28X11%3B%20Linux%20x86_64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F140.0.0.0%20Safari%2F537.36&"
            "channel=tiktok_web&cookie_enabled=true&count=5&data_collection_enabled=true&"
            "device_id=7506194516308166166&device_platform=web_pc&focus_state=true&"
            "from_page=user&history_len=3&is_fullscreen=false&is_page_visible=true&"
            "maxCursor=0&minCursor=0&odinId=7246312836442604570&os=linux&priority_region=IT&"
            "referer=&region=IT&root_referer=https%3A%2F%2Fwww.tiktok.com%2Flive&scene=21&"
            "screen_height=1080&screen_width=1920&tz_name=Europe%2FRome&user_is_login=true&"
            "verifyFp=verify_mh4yf0uq_rdjp1Xwt_OoTk_4Jrf_AS8H_sp31opbnJFre&webcast_language=it-IT&"
            "msToken=GphHoLvRR4QxA5AWVwDkrs3AbumoK5H8toE8LVHtj6cce3ToGdXhMfvDWzOXG-0GXUWoaGVHrwGNA4k_NnjuFFnHgv2S5eMjsvtkAhwMPa13xLmvP7tumx0KreFjPwTNnOj-BvAkPdO5Zrev3hoFBD9lHVo=&X-Bogus=&X-Gnarly="
        ).cookies["msToken"]

        while has_more:
            url = (
                "https://www.tiktok.com/api/user/list/?"
                "WebIdLastTime=1747672102&aid=1988&app_language=it-IT&app_name=tiktok_web"
                "&browser_language=it-IT&browser_name=Mozilla&browser_online=true"
                "&browser_platform=Linux%20x86_64&browser_version=5.0%20%28X11%3B%20Linux%20x86_64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F140.0.0.0%20Safari%2F537.36&channel=tiktok_web&"
                "cookie_enabled=true&count=5&data_collection_enabled=true&device_id=7506194516308166166"
                "&device_platform=web_pc&focus_state=true&from_page=user&history_len=3&"
                f"is_fullscreen=false&is_page_visible=true&maxCursor={cursor}&minCursor={cursor}&"
                "odinId=7246312836442604570&os=linux&priority_region=IT&referer=&"
                "region=IT&scene=21&screen_height=1080&screen_width=1920"
                "&tz_name=Europe%2FRome&user_is_login=true&"
                f"secUid={sec_uid}&verifyFp=verify_mh4yf0uq_rdjp1Xwt_OoTk_4Jrf_AS8H_sp31opbnJFre&"
                f"webcast_language=it-IT&msToken={ms_token}&X-Bogus=&X-Gnarly="
            )

            response = self.http_client.get(url)

            if response.status_code != StatusCode.OK:
                raise TikTokRecorderError("Failed to retrieve followers list.")

            if not response.content:
                raise TikTokRecorderError("Empty response from TikTok followers API.")

            data = response.json()
            user_list = data.get("userList", [])

            for user in user_list:
                username = user.get("user", {}).get("uniqueId")
                if username:
                    followers.append(username)

            has_more = data.get("hasMore", False)
            new_cursor = data.get("minCursor", 0)

            if new_cursor == cursor:
                break

            cursor = new_cursor

        if not followers:
            raise TikTokRecorderError("Followers list is empty.")

        return followers

    def _get_stream_url_from_page(self, user: str) -> str | None:
        """
        Fallback: fetch the live page HTML and extract the stream URL directly.
        Used when the webcast API returns status code 4003110 (WAF/access restriction).
        """
        try:
            live_page_url = f"{self.BASE_URL}/@{user}/live"
            response = self.http_client.get(live_page_url)
            content = response.text

            flv_matches = re.findall(r'https?://[^\s"\'<>]+\.flv[^\s"\'<>]*', content)
            if flv_matches:
                # Prefer original (_or4) or SD quality
                for url in flv_matches:
                    url = html.unescape(url.rstrip("\\"))
                    if "_or4" in url or "_sd" in url:
                        logger.info(f"Found stream URL from page: {url[:80]}...")
                        return url
                return html.unescape(flv_matches[0].rstrip("\\"))

            hls_matches = re.findall(r'https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*', content)
            if hls_matches:
                return html.unescape(hls_matches[0].rstrip("\\"))

            return None
        except Exception as e:
            logger.warning(f"Failed to extract stream URL from page: {e}")
            return None

    def get_live_url(self, room_id: str, user: str = None) -> str | None:
        """
        Return the cdn (flv or m3u8) of the streaming.
        If the API returns status code 4003110 and a username is provided,
        falls back to scraping the live page directly.
        """
        data = self.http_client.get(
            f"{self.WEBCAST_URL}/webcast/room/info/?aid=1988&room_id={room_id}"
        ).json()

        if "This account is private" in data:
            raise UserLiveError(TikTokError.ACCOUNT_PRIVATE)

        status_code = data.get("status_code", 0)

        if status_code == 4003110:
            if user:
                logger.info(
                    "API blocked by WAF (4003110). Trying fallback: extract stream URL from live page..."
                )
                fallback_url = self._get_stream_url_from_page(user)
                if fallback_url:
                    return fallback_url

            raise UserLiveError(TikTokError.LIVE_RESTRICTION)

        stream_url = data.get("data", {}).get("stream_url", {})

        sdk_data_str = (
            stream_url.get("live_core_sdk_data", {})
            .get("pull_data", {})
            .get("stream_data")
        )
        if not sdk_data_str:
            logger.warning(
                "No SDK stream data found. Falling back to legacy URLs. Consider contacting the developer to update the code."
            )
            return (
                stream_url.get("flv_pull_url", {}).get("FULL_HD1")
                or stream_url.get("flv_pull_url", {}).get("HD1")
                or stream_url.get("flv_pull_url", {}).get("SD2")
                or stream_url.get("flv_pull_url", {}).get("SD1")
                or stream_url.get("rtmp_pull_url", "")
            )

        # Extract stream options
        sdk_data = json.loads(sdk_data_str).get("data", {})
        qualities = (
            stream_url.get("live_core_sdk_data", {})
            .get("pull_data", {})
            .get("options", {})
            .get("qualities", [])
        )
        if not qualities:
            logger.warning("No qualities found in the stream data. Returning None.")
            return None
        level_map = {q["sdk_key"]: q["level"] for q in qualities}

        best_level = -1
        best_flv = None
        for sdk_key, entry in sdk_data.items():
            level = level_map.get(sdk_key, -1)
            stream_main = entry.get("main", {})
            if level > best_level:
                best_level = level
                best_flv = stream_main.get("flv")

        return best_flv

    def download_live_stream(self, live_url: str):
        """Generator that returns the live stream for a given room_id."""
        stream = self._http_client_stream.get(live_url, stream=True)
        for chunk in stream.iter_content(chunk_size=4096):
            if chunk:
                yield chunk
