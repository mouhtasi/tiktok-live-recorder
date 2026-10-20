"""Driver for test_forkserver_preload.py: one scenario per fresh interpreter.

The forkserver is a per-interpreter singleton whose preload list is fixed when
it starts, so each scenario needs its own process — as prod's recorder has.
Run with the test's tmp_path as cwd: the logger opens tiktok-recorder.log
relative to it. Prints one JSON line.

Usage: python forkserver_driver.py SCENARIO preload|none [extra args]
"""

import json
import multiprocessing
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

PRELOADED_MARKER = "core.tiktok_recorder"


def native_thread_count():
    if os.path.isdir("/proc/self/task"):
        return len(os.listdir("/proc/self/task"))
    out = subprocess.run(
        ["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout
    return len(out.strip().splitlines()) - 1


def private_dirty_kb():
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith("Private_Dirty:"):
                return int(line.split()[1])
    return None


# --- children: each reports whether it inherited the recorder ----------------


def child_tls(url, queue):
    preloaded = PRELOADED_MARKER in sys.modules
    from http_utils.http_client import HttpClient

    client = HttpClient()
    curl = client.req.get(url, verify=False, timeout=10)
    plain = client.req_stream.get(url, verify=False, timeout=10)
    queue.put(
        {
            "preloaded": preloaded,
            "curl_status": curl.status_code,
            "curl_body": curl.text,
            "requests_status": plain.status_code,
        }
    )


def child_logs(name, queue):
    preloaded = PRELOADED_MARKER in sys.modules
    from utils.logger_manager import logger, use_worker_log_file

    use_worker_log_file(name, log_dir="logs")
    logger.debug(f"marker-{name}")
    for handler in logger.handlers:
        handler.flush()
    main_log = os.path.abspath("tiktok-recorder.log")
    fds = None
    if os.path.isdir("/proc/self/fd"):
        fds = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}") == main_log:
                    fds += 1
            except OSError:
                pass
    queue.put({"preloaded": preloaded, "main_log_fds": fds})


def child_random(queue):
    import random

    preloaded = PRELOADED_MARKER in sys.modules
    # The draw core.tiktok_recorder makes for the first-poll jitter.
    queue.put({"preloaded": preloaded, "draw": random.uniform(0, 300)})


def child_memory(queue):
    preloaded = PRELOADED_MARKER in sys.modules
    import gc

    import core.tiktok_recorder  # noqa: F401 — what record_user() imports
    from http_utils.http_client import HttpClient

    client = HttpClient()  # noqa: F841
    gc.collect()
    queue.put({"preloaded": preloaded, "private_dirty_kb": private_dirty_kb()})


# --- scenarios ----------------------------------------------------------------


def run_children(target, arg_sets):
    queue = multiprocessing.Queue()
    procs = [
        multiprocessing.Process(target=target, args=(*args, queue)) for args in arg_sets
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    return results


def serve_https(cert, key):
    import http.server
    import ssl
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok-from-test-server"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"https://127.0.0.1:{server.server_address[1]}/"


def main():
    scenario, mode = sys.argv[1], sys.argv[2]

    if scenario == "import":
        # No fork at all: what importing the preload module leaves running.
        import threading

        import gc
        from utils import forkserver_preload

        print(
            json.dumps(
                {
                    "python_threads": threading.active_count(),
                    "native_threads": native_thread_count(),
                    "loaded": forkserver_preload.LOADED,
                    "modules": list(forkserver_preload.MODULES),
                    "freeze_count": gc.get_freeze_count(),
                }
            )
        )
        return

    # Linux 3.14+ defaults to forkserver; macOS defaults to spawn, so force it.
    multiprocessing.set_start_method("forkserver", force=True)
    preload_enabled = False
    if mode == "preload":
        import main as recorder_main
        from utils.logger_manager import logger

        preload_enabled = recorder_main._preload_forkserver(logger)

    if scenario == "tls":
        url = serve_https(sys.argv[3], sys.argv[4])
        results = run_children(child_tls, [(url,), (url,), (url,)])
    elif scenario == "logs":
        results = run_children(child_logs, [("alice",), ("bob",)])
    elif scenario == "random":
        results = run_children(child_random, [()] * 4)
    elif scenario == "memory":
        results = run_children(child_memory, [()] * 3)
    else:
        raise SystemExit(f"unknown scenario {scenario}")

    print(json.dumps({"preload_enabled": preload_enabled, "children": results}))


if __name__ == "__main__":
    main()
