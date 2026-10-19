import tempfile
import threading
import time
import unittest
from pathlib import Path

from laika.config import Config, LedsConfig, validate_leds
from laika.leds import (
    HEARTBEAT_PERIOD_S,
    NullLed,
    StatusLeds,
    SysfsLed,
    activity_pattern,
    open_led,
    pattern_value,
    status_pattern,
)
from tests.agent.test_agent import AgentTestCase, text


class FakeLed:
    def __init__(self, fail=False):
        self.writes = []
        self.closed = False
        self.fail = fail

    def set(self, on):
        if self.fail:
            raise OSError("no led")
        self.writes.append(on)

    def close(self):
        self.closed = True


def on_fraction(name, seconds=2.8, step=0.005):
    samples = [pattern_value(name, i * step) for i in range(int(seconds / step))]
    return sum(samples) / len(samples)


class PatternTests(unittest.TestCase):
    def test_a_heartbeat_has_two_thumps_then_rest(self):
        thumps = 0
        previous = False
        for i in range(int(HEARTBEAT_PERIOD_S / 0.005)):
            now = pattern_value("heartbeat", i * 0.005)
            thumps += 1 if now and not previous else 0
            previous = now
        self.assertEqual(thumps, 2)
        self.assertLess(on_fraction("heartbeat"), 0.3)
        self.assertFalse(pattern_value("heartbeat", 1.0))

    def test_the_blinks_have_different_rhythms(self):
        def changes(name):
            values = [pattern_value(name, i * 0.01) for i in range(200)]
            return sum(1 for a, b in zip(values, values[1:]) if a != b)

        self.assertGreater(changes("fast"), changes("slow") * 3)
        self.assertGreater(changes("alarm"), changes("slow") * 3)
        self.assertTrue(all(pattern_value("solid", t) for t in (0, 0.3, 7)))
        self.assertFalse(any(pattern_value("off", t) for t in (0, 0.3, 7)))

    def test_the_status_led_shows_the_most_urgent_thing(self):
        self.assertEqual(status_pattern(stopped=False, battery="ok"), "heartbeat")
        self.assertEqual(status_pattern(stopped=False, battery=None), "heartbeat")
        self.assertEqual(status_pattern(stopped=False, battery="unknown"), "heartbeat")
        self.assertEqual(status_pattern(stopped=False, battery="low"), "slow")
        self.assertEqual(status_pattern(stopped=False, battery="critical"), "alarm")
        self.assertEqual(status_pattern(stopped=True, battery="critical"), "solid")

    def test_the_activity_led_priorities(self):
        self.assertEqual(activity_pattern(thinking=True, learning=True, awake=True), "fast")
        self.assertEqual(activity_pattern(thinking=False, learning=True, awake=True), "slow")
        self.assertEqual(activity_pattern(thinking=False, learning=False, awake=True), "solid")
        self.assertEqual(activity_pattern(thinking=False, learning=False, awake=False), "off")


class DriverTests(unittest.TestCase):
    def test_statusleds_writes_only_on_a_change_and_follows_the_sources(self):
        status, activity = FakeLed(), FakeLed()
        state = {"thinking": False, "awake": False, "battery": "ok"}
        leds = StatusLeds(
            status, activity,
            thinking=lambda: state["thinking"], awake=lambda: state["awake"], battery=lambda: state["battery"],
        )
        leds.tick(0.0)  # heartbeat thump on, activity off
        leds.tick(0.01)  # still on: no new write
        self.assertEqual(status.writes, [True])
        self.assertEqual(activity.writes, [False])
        leds.tick(0.15)  # between the thumps
        self.assertEqual(status.writes, [True, False])
        state["thinking"] = True
        leds.tick(1.01)  # fast blink: phase 0.01 of 0.2 is on
        self.assertEqual(activity.writes, [False, True])
        state["thinking"], state["awake"] = False, True
        leds.tick(1.5)
        self.assertEqual(activity.writes[-1], True)  # solid while awake
        state["awake"] = False
        leds.tick(1.6)
        self.assertEqual(activity.writes[-1], False)

    def test_a_failing_source_or_led_never_breaks_the_loop(self):
        def broken():
            raise RuntimeError("boom")

        bad = FakeLed(fail=True)
        leds = StatusLeds(bad, FakeLed(), thinking=broken, battery=broken)
        leds.tick(0.0)
        leds.tick(0.1)
        self.assertEqual(leds.patterns(), ("heartbeat", "off"))

    def test_the_thread_runs_and_close_gives_the_leds_back(self):
        status, activity = FakeLed(), FakeLed()
        leds = StatusLeds(status, activity)
        leds.start()
        deadline = time.time() + 5
        while not status.writes and time.time() < deadline:
            time.sleep(0.01)
        leds.close()
        self.assertTrue(status.writes)
        self.assertTrue(status.closed and activity.closed)
        self.assertFalse(leds._thread.is_alive())

    def test_sysfs_takes_the_led_over_and_puts_it_back(self):
        folder = Path(tempfile.mkdtemp()) / "PWR"
        folder.mkdir()
        (folder / "trigger").write_text("none [default-on] heartbeat mmc0")
        (folder / "brightness").write_text("1")
        led = SysfsLed(folder)
        self.assertEqual((folder / "trigger").read_text(), "none")
        led.set(False)
        self.assertEqual((folder / "brightness").read_text(), "0")
        led.set(True)
        self.assertEqual((folder / "brightness").read_text(), "1")
        led.close()
        self.assertEqual((folder / "trigger").read_text(), "default-on")

    def test_open_led_falls_back_to_a_null_led(self):
        empty = Path(tempfile.mkdtemp())
        self.assertIsInstance(open_led("none"), NullLed)
        self.assertIsInstance(open_led("pi_red", empty), NullLed)  # no such LED here
        self.assertIsInstance(open_led("hat"), NullLed)  # robot_hat is not installed off the robot

    def test_open_led_finds_the_pi_led_by_either_name(self):
        root = Path(tempfile.mkdtemp())
        (root / "led1").mkdir()
        (root / "led1" / "trigger").write_text("none [input] default-on")
        (root / "led1" / "brightness").write_text("0")
        led = open_led("pi_red", root)
        self.assertIsInstance(led, SysfsLed)
        led.close()
        self.assertEqual((root / "led1" / "trigger").read_text(), "input")


class ConfigTests(unittest.TestCase):
    def test_defaults_validate_and_bad_values_are_named(self):
        validate_leds(Config().leds)
        for settings in ({"status": "blue"}, {"activity": "pi_red"}, {"enabled": 1}, {"status": "hat", "activity": "hat"}):
            with self.subTest(settings), self.assertRaises(ValueError):
                validate_leds(LedsConfig(**settings))
        validate_leds(LedsConfig(status="none", activity="none"))


class ThinkingTests(AgentTestCase):
    def test_the_agent_counts_requests_in_flight(self):
        seen = []
        agent = self.build(text("ciao"))
        client = agent.client
        original = client.complete

        def watching(*args, **kwargs):
            seen.append(agent.thinking)
            return original(*args, **kwargs)

        client.complete = watching
        self.assertEqual(agent.thinking, 0)
        agent.respond("ciao")
        self.assertEqual(seen, [1])
        self.assertEqual(agent.thinking, 0)

    def test_it_goes_back_to_zero_when_the_model_is_down(self):
        from laika.ai_client import AIUnavailable

        agent = self.build(AIUnavailable("down"))
        with self.assertRaises(AIUnavailable):
            agent.respond("ciao")
        self.assertEqual(agent.thinking, 0)


if __name__ == "__main__":
    unittest.main()
