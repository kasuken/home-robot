#!/usr/bin/env python3

from picarx import Picarx
from vilib import Vilib
from picarx.tts import Pico2Wave

from time import sleep
import threading
import readchar
import random

# -----------------------
# Settings
# -----------------------
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
DETECTION_WIDTH_THRESHOLD = 100  # how wide the color blob must be
DRIVE_SPEED = 80
TURN_ANGLE = 30

MANUAL = """
Press keys to control PiCar-X:
  w: forward    a: turn left    s: backward    d: turn right
  space: repeat target          Ctrl+C: quit
"""

# -----------------------
# Init
# -----------------------
px = Picarx()

tts = Pico2Wave()
tts.set_lang("en-US")

current_color = "red"
key = None
lock = threading.Lock()

def say(line: str):
    print(f"[SAY] {line}")
    tts.say(line)

def renew_color_detect():
    """Choose a new target color and start detection."""
    global current_color
    current_color = random.choice(COLORS)
    Vilib.color_detect(current_color)
    say(f"Look for {current_color}!")

def key_scan_thread():
    """Background thread reading keys."""
    global key
    while True:
        k = readchar.readkey()
        # Map special keys before lowercasing
        if k == readchar.key.SPACE:
            mapped = "space"
        elif k == readchar.key.CTRL_C:
            mapped = "quit"
        else:
            mapped = k.lower()

        with lock:
            key = mapped

        if mapped == "quit":
            return
        sleep(0.01)

def car_move(k: str):
    if k == "w":
        px.set_dir_servo_angle(0)
        px.forward(DRIVE_SPEED)
    elif k == "s":
        px.set_dir_servo_angle(0)
        px.backward(DRIVE_SPEED)
    elif k == "a":
        px.set_dir_servo_angle(-TURN_ANGLE)
        px.forward(DRIVE_SPEED)
    elif k == "d":
        px.set_dir_servo_angle(TURN_ANGLE)
        px.forward(DRIVE_SPEED)

def main():
    global key

    # Start camera and web preview
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=False, web=True)
    sleep(0.8)

    print(MANUAL.strip())
    say("Game start!")
    sleep(0.1)
    renew_color_detect()

    # Start keyboard thread (modern style)
    key_thread = threading.Thread(target=key_scan_thread, daemon=True)
    key_thread.start()

    try:
        while True:
            # Check detection: if target color present and wide enough
            if (Vilib.detect_obj_parameter.get("color_n", 0) != 0 and
                Vilib.detect_obj_parameter.get("color_w", 0) > DETECTION_WIDTH_THRESHOLD):
                say("Well done!")
                sleep(0.1)
                renew_color_detect()

            # Take a snapshot of the last key (and clear it)
            with lock:
                k = key
                key = None

            # Handle movement / actions
            if k in ("w", "a", "s", "d"):
                car_move(k)
                sleep(0.5)
                px.stop()
            elif k == "space":
                say(f"Look for {current_color}!")
            elif k == "quit":
                print("\n[INFO] Quit requested.")
                break

            sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            Vilib.camera_close()
        except Exception:
            pass
        px.stop()
        say("Goodbye!")
        sleep(0.2)

if __name__ == "__main__":
    main()
