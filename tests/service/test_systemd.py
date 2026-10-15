import os
import socket
import tempfile
import unittest
from pathlib import Path

from laika.systemd import Notifier

POSIX = hasattr(socket, "AF_UNIX") and os.name == "posix"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class NoopTests(unittest.TestCase):
    def test_without_notify_socket_nothing_happens(self):
        notifier = Notifier.from_env({})
        self.assertFalse(notifier.enabled)
        self.assertFalse(notifier.ready())
        self.assertFalse(notifier.watchdog())
        self.assertFalse(notifier.stopping())

    def test_watchdog_interval_from_environment(self):
        notifier = Notifier.from_env({"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "5000000"})
        self.assertAlmostEqual(notifier.watchdog_interval_s, 5 / 3)

    def test_watchdog_for_another_pid_is_ignored(self):
        env = {"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "5000000", "WATCHDOG_PID": "1"}
        self.assertIsNone(Notifier.from_env(env).watchdog_interval_s)
        env["WATCHDOG_PID"] = str(os.getpid())
        self.assertIsNotNone(Notifier.from_env(env).watchdog_interval_s)

    def test_bad_usec_means_no_watchdog(self):
        self.assertIsNone(Notifier.from_env({"WATCHDOG_USEC": "abc"}).watchdog_interval_s)

    def test_unreachable_socket_does_not_raise(self):
        notifier = Notifier("/nonexistent/notify.sock", 3_000_000)
        self.assertFalse(notifier.ready())
        self.assertFalse(notifier.watchdog())


@unittest.skipUnless(POSIX, "needs Unix datagram sockets")
class SocketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "notify")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(self.server.close)
        self.server.bind(self.path)
        self.server.settimeout(2)

    def received(self) -> str:
        return self.server.recv(1024).decode()

    def test_messages_reach_the_socket(self):
        notifier = Notifier(self.path)
        self.assertTrue(notifier.ready())
        self.assertEqual(self.received(), "READY=1")
        self.assertTrue(notifier.stopping())
        self.assertEqual(self.received(), "STOPPING=1")
        self.assertTrue(notifier.status("hi\nthere"))
        self.assertEqual(self.received(), "STATUS=hi there")

    def test_watchdog_pings_at_a_third_of_the_timeout(self):
        clock = FakeClock()
        notifier = Notifier(self.path, 6_000_000, clock)
        self.assertTrue(notifier.watchdog())
        self.assertEqual(self.received(), "WATCHDOG=1")
        clock.now += 1.9  # the interval is 2 s
        self.assertFalse(notifier.watchdog())
        clock.now += 0.2
        self.assertTrue(notifier.watchdog())
        self.assertEqual(self.received(), "WATCHDOG=1")

    def test_no_watchdog_ping_when_systemd_has_no_watchdog(self):
        self.assertFalse(Notifier(self.path, 0).watchdog())

    def test_abstract_namespace_address(self):
        if os.uname().sysname != "Linux":
            self.skipTest("abstract sockets are Linux only")
        name = "laika-test-" + os.path.basename(self.tmp.name)
        abstract = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(abstract.close)
        abstract.bind("\0" + name)
        abstract.settimeout(2)
        self.assertTrue(Notifier("@" + name).ready())
        self.assertEqual(abstract.recv(1024), b"READY=1")


if __name__ == "__main__":
    unittest.main()
