"""Run a luka module with its output appended to a log file.

For Windows scheduled tasks started with pythonw.exe, which opens no console window
(so nothing flashes on screen every 15 minutes) but also has nowhere to print.

    pythonw scripts/run_logged.py <log file> <module> [args...]
    pythonw scripts/run_logged.py markets.log luka.ingest.markets
"""
import contextlib
import os
import runpy
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Windows starts a missed task the moment the laptop wakes, often before Wi-Fi has
# reconnected, and the first DNS lookup fails. Wait for DNS before running.
NETWORK_PROBE_HOST = "api.elections.kalshi.com"
NETWORK_WAIT_S = 180


def wait_for_network(timeout_s: float = NETWORK_WAIT_S, step_s: float = 10) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            socket.getaddrinfo(NETWORK_PROBE_HOST, 443)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(step_s)


def main() -> int:
    log_name, module, *args = sys.argv[1:]
    os.chdir(ROOT)  # so .env is found and the luka package imports
    sys.path.insert(0, ROOT)
    code = 0
    with open(os.path.join(ROOT, log_name), "a", encoding="utf-8") as log:
        log.write(f"---- {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC\n")
        if not wait_for_network():
            log.write(f"no network after {NETWORK_WAIT_S}s, skipped\nexit 1\n")
            return 1
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            sys.argv = [module, *args]
            try:
                runpy.run_module(module, run_name="__main__")
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            except Exception:
                traceback.print_exc()
                code = 1
        log.write(f"exit {code}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
