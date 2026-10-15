from picarx import Picarx
from picarx.tts import Espeak
import time

# If you want to try Pico2Wave instead of Espeak, uncomment below:
# from picarx.tts import Pico2Wave
# tts = Pico2Wave()
# tts.set_lang('en-US')  # Options: en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

px = Picarx()
tts = Espeak()

# Quick hello (test)
tts.say("Hello! I'm PiCar-X.")

def main():
    try:
        # Forward
        tts.say("Moving forward")
        px.forward(30)
        time.sleep(2)
        px.stop()

        # Backward
        tts.say("Moving backward")
        px.backward(30)
        time.sleep(2)
        px.stop()

        # Turn left
        tts.say("Turning left")
        px.set_dir_servo_angle(-20)
        px.forward(30)
        time.sleep(2)
        px.stop()
        px.set_dir_servo_angle(0)

        # Turn right
        tts.say("Turning right")
        px.set_dir_servo_angle(20)
        px.forward(30)
        time.sleep(2)
        px.stop()
        px.set_dir_servo_angle(0)

    except KeyboardInterrupt:
        # Stop if interrupted
        px.stop()
    finally:
        # Reset to safe state
        px.stop()
        px.set_dir_servo_angle(0)

if __name__ == "__main__":
    main()
