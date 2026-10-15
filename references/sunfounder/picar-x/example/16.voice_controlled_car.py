from picarx import Picarx
from picarx.stt import Vosk
import time

px = Picarx()
stt = Vosk(language="en-us")


WAKE_WORDS = ["hey robot"]

print('Say "hey robot" to wake me up! Then say: forward / backward / left / right. Say "sleep" to stop listening.')

try:
    while True:
        # --- wait for wake word once ---
        stt.wait_until_heard(WAKE_WORDS)
        print("Wake word detected. Listening for commands... (say 'sleep' to pause)")

        # --- command loop: multiple commands after one wake ---
        while True:
            res = stt.listen(stream=False)
            text = res.get("text", "") if isinstance(res, dict) else str(res)
            text = text.lower().strip()
            if not text:
                continue

            print("Heard:", text)

            if "sleep" in text:
                # pause command mode; go back to wait for wake word
                px.stop(); px.set_dir_servo_angle(0)
                print("Sleeping. Say 'hey robot' to wake me again.")
                break

            elif "forward" in text:
                px.set_dir_servo_angle(0)
                px.forward(30); time.sleep(1); px.stop()

            elif "backward" in text:
                px.set_dir_servo_angle(0)
                px.backward(30); time.sleep(1); px.stop()

            elif "left" in text:
                px.set_dir_servo_angle(-25)
                px.forward(30); time.sleep(1)
                px.stop(); px.set_dir_servo_angle(0)

            elif "right" in text:
                px.set_dir_servo_angle(25)
                px.forward(30); time.sleep(1)
                px.stop(); px.set_dir_servo_angle(0)
            # (ignore other words)

except KeyboardInterrupt:
    pass
finally:
    px.stop(); px.set_dir_servo_angle(0)
    print("Stopped and centered. Bye.")

