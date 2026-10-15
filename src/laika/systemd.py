"""Tell systemd how Laika is doing (sd_notify), with the standard library only.

Under a `Type=notify` unit with `WatchdogSec=`, systemd restarts Laika when
the pings stop. Everything here is a no-op when `NOTIFY_SOCKET` is not set
(a laptop, `--sim`, a manual start) and on platforms without Unix sockets.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Callable, Mapping

log = logging.getLogger(__name__)

PING_FRACTION = 3  # systemd advises pinging at half the timeout or faster; a third leaves slack
SEND_TIMEOUT_S = 0.2


class Notifier:
    """Sends `READY=1`, `WATCHDOG=1` and `STOPPING=1`. No method ever raises."""

    def __init__(
        self,
        address: str | None,
        watchdog_usec: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.address = address if address and hasattr(socket, "AF_UNIX") else None
        self.watchdog_interval_s = (
            watchdog_usec / 1_000_000 / PING_FRACTION if watchdog_usec > 0 else None
        )
        self._clock = clock
        self._last_ping: float | None = None
        self._warned = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Notifier:
        env = os.environ if env is None else env
        usec = 0
        # WATCHDOG_PID names the process systemd expects the pings from.
        pid = env.get("WATCHDOG_PID")
        if not pid or pid == str(os.getpid()):
            try:
                usec = int(env.get("WATCHDOG_USEC", "0"))
            except ValueError:
                usec = 0
        return cls(env.get("NOTIFY_SOCKET"), usec)

    @property
    def enabled(self) -> bool:
        return self.address is not None

    def ready(self) -> bool:
        return self._send("READY=1")

    def stopping(self) -> bool:
        return self._send("STOPPING=1")

    def status(self, text: str) -> bool:
        return self._send("STATUS=" + text.replace("\n", " "))

    def watchdog(self) -> bool:
        """Ping the watchdog if a ping is due. Cheap enough to call on every monitor tick."""
        interval = self.watchdog_interval_s
        if not self.enabled or interval is None:
            return False
        now = self._clock()
        if self._last_ping is not None and now - self._last_ping < interval:
            return False
        sent = self._send("WATCHDOG=1")
        if sent:
            self._last_ping = now
        return sent

    def _send(self, message: str) -> bool:
        if self.address is None:
            return False
        address = self.address
        if address.startswith("@"):  # abstract namespace
            address = "\0" + address[1:]
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | getattr(socket, "SOCK_CLOEXEC", 0)) as sock:
                sock.settimeout(SEND_TIMEOUT_S)
                sock.connect(address)
                sock.sendall(message.encode("utf-8"))
        except OSError as error:
            if not self._warned:
                self._warned = True
                log.warning("cannot notify systemd (%s): %s", message, error)
            return False
        return True
