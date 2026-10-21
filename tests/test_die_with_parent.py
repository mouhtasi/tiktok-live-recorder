"""
A monitor dies with main.py (tiktak, 2026-10-21).

If main.py is killed (a cgroup cap kill can pick it), its monitors keep
recording, and tiktak's watchdog starts a new recorder beside them: two
monitors per account. tiktak now ends the old process group before a heal;
this is the second layer, so an orphan cannot outlive its recorder even if
that step is missed.

🚨 The first version used PR_SET_PDEATHSIG and passed its fake-libc tests. On
prod it did nothing: the forkserver hands its alive pipe to every child
(multiprocessing/forkserver.py), so it outlives main.py while any monitor
lives, and a death signal tied to the forkserver never fires. So these tests
kill a real parent, and the control proves the orphan survives without the
guard; otherwise a broken guard and a harmless setup look the same.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DRIVER = Path(__file__).with_name("parent_death_driver.py")


def _orphan(mode):
    # Read one line, not to EOF: the orphan inherits stdout and holds it open.
    driver = subprocess.Popen([sys.executable, str(DRIVER), mode],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pid = int(driver.stdout.readline())
    driver.wait(timeout=30)
    driver.stdout.close()
    return pid


def _alive(pid):
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _gone_within(pid, seconds):
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _cleanup(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_without_the_guard_a_monitor_outlives_main():
    pid = _orphan("noguard")
    try:
        time.sleep(2)
        assert _alive(pid), "the control must show the orphan, or the next test proves nothing"
    finally:
        _cleanup(pid)


def test_with_the_guard_a_monitor_ends_when_main_is_killed():
    pid = _orphan("guard")
    try:
        assert _gone_within(pid, 5)
    finally:
        _cleanup(pid)
