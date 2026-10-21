"""Driver for test_die_with_parent: start one forkserver child, print its pid,
then SIGKILL this process the way a cgroup cap kill ends main.py.

Usage: python parent_death_driver.py guard|noguard
"""
import multiprocessing
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def target(guard):
    if guard:
        import main

        main._die_with_parent()
    time.sleep(30)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process(target=target, args=(sys.argv[1] == "guard",))
    p.start()
    print(p.pid, flush=True)
    time.sleep(1)  # let the child install its watch before the parent dies
    os.kill(os.getpid(), signal.SIGKILL)
