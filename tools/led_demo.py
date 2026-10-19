"""Plays with the Robot HAT's user LED: run it on the robot, with the service stopped.

    sudo systemctl stop laika
    sudo python3 tools/led_demo.py            # all patterns once
    sudo python3 tools/led_demo.py heartbeat  # one pattern: blink, heartbeat, morse, countdown
    sudo systemctl start laika

The HAT's user LED is a single on/off LED (no colour, no brightness), driven through SunFounder's
`robot_hat.led.LED` like in the vendor voice examples (`on`, `off`, `blink`, `close`). It is not part of
Laika (she does not use LEDs yet), so this lives outside src/ and imports the vendor package only here.
"""

from __future__ import annotations

import sys
import time

MORSE = {"L": ".-..", "A": ".-", "I": "..", "K": "-.-"}
UNIT = 0.12  # seconds


def blink(led, count: int = 5) -> None:
    led.blink(times=count, delay=0.15)


def heartbeat(led, beats: int = 5) -> None:
    for _ in range(beats):
        for pause in (0.12, 0.12):
            led.on()
            time.sleep(pause)
            led.off()
            time.sleep(0.12)
        time.sleep(0.7)


def morse(led, word: str = "LAIKA") -> None:
    for letter in word:
        for symbol in MORSE[letter]:
            led.on()
            time.sleep(UNIT * (3 if symbol == "-" else 1))
            led.off()
            time.sleep(UNIT)
        time.sleep(UNIT * 2)


def countdown(led, start: int = 5) -> None:
    for number in range(start, 0, -1):
        for _ in range(number):
            led.on()
            time.sleep(0.15)
            led.off()
            time.sleep(0.15)
        time.sleep(0.6)


PATTERNS = {"blink": blink, "heartbeat": heartbeat, "morse": morse, "countdown": countdown}


def main(argv: list[str]) -> int:
    names = argv or list(PATTERNS)
    unknown = [name for name in names if name not in PATTERNS]
    if unknown:
        print(f"unknown pattern {unknown[0]!r}; choose from: {', '.join(PATTERNS)}")
        return 2
    from robot_hat.led import LED  # the SunFounder package, installed on the robot

    led = LED()
    try:
        for name in names:
            print(f"{name}...")
            PATTERNS[name](led)
            led.off()
            time.sleep(0.8)
    finally:
        led.off()
        led.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
