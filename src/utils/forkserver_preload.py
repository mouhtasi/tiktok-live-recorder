"""Imported once, by the forkserver, so every monitor shares one copy (tiktak §56).

Every monitor is a fork of multiprocessing's forkserver (the default start
method on Linux since Python 3.14). Without a preload the forkserver has
imported almost nothing, and ``record_user()`` imports the recorder inside each
child, so each of ~190 monitors built its own private ~20 MB copy of the same
modules: 4.4 GB on prod on 2026-10-15, at the host's RAM wall.

Importing them here, in the forkserver, puts them in pages the children inherit
copy-on-write. That alone saved only 30% (21.5 -> 14.9 MB per child, measured
on prod), because CPython writes to the objects it inherited: the cyclic GC
writes a header field on every tracked object each time it collects, and one
write copies the whole 4 KB page. ``gc.freeze()`` moves everything imported so
far into a permanent generation the GC never scans, so those pages stay shared:
6.8 MB per child, 4.5 MB of it private.

🚨 This module must never raise. The stdlib forkserver skips a preload module
only on ImportError; any other exception escapes ``forkserver.main()``, the
forkserver dies, and every later ``Process.start()`` fails — no monitor could
start or be respawned. So each import is guarded, and a failure here falls back
to today's behaviour: the child imports what it needs itself.
"""

import gc

# What record_user() pulls in. core.tiktok_recorder brings most of the graph
# (core.tiktok_api, http_utils, utils.*, requests); curl_cffi is imported
# lazily inside HttpClient, so it is named here or it would still be private
# per child.
#
# 🚨 Never add setproctitle. On macOS its import starts two native threads,
# and every TLS request in a child forked from that forkserver then hung
# forever — found by test_preloaded_children_complete_tls_requests, 2026-10-16.
# It is a few KB of C, so preloading it saves nothing; record_user() imports it
# after the fork, where a thread it starts harms only that one monitor.
MODULES = (
    "curl_cffi",
    "requests",
    "utils.logger_manager",
    "http_utils.http_client",
    "core.tiktok_api",
    "core.tiktok_recorder",
    # Imported after the fork on a real monitor's first minute (a sys.modules
    # diff on prod, 2026-10-16): unpickling its stop events, distro and
    # argparse via the recorder's helpers, netrc on each requests call.
    "multiprocessing.synchronize",
    "multiprocessing.sharedctypes",
    "ctypes",
    "sysconfig",
    "platform",
    "distro",
    "argparse",
    "gettext",
    "shlex",
    "netrc",
)


def _preload(names):
    loaded = []
    for name in names:
        try:
            __import__(name)
            loaded.append(name)
        except BaseException:  # noqa: BLE001 — see the module docstring
            pass
    try:
        gc.collect()
        gc.freeze()
    except BaseException:  # noqa: BLE001
        pass
    return loaded


LOADED = _preload(MODULES)
