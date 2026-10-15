from picarx import Picarx
import time

# === TTS Configuration ===
# Default: Piper
from picarx.tts import Piper
tts = Piper()
tts.set_model("en_US-amy-low")  # use the voice model you installed

# Optional: switch to OpenAI TTS
# from picarx.tts import OpenAI_TTS
# from secret import OPENAI_API_KEY
# tts = OpenAI_TTS(api_key=OPENAI_API_KEY)
# tts.set_model("gpt-4o-mini-tts")  # low-latency TTS model
# tts.set_voice("alloy")            # choose a voice

# === PiCar-X Setup ===
px = Picarx()

# Quick hello (sanity check)
tts.say("Hello! I'm PiCar-X speaking with Piper.")

def main():
    try:
        # Leg 1
        px.forward(30)
        time.sleep(3)
        px.stop()
        tts.say("Why can't your nose be twelve inches long? Because then it would be a foot!")

        # Leg 2
        px.forward(30)
        time.sleep(3)
        px.stop()
        tts.say("Why did the cow go to outer space? To see the moooon!")

        # Wrap-up
        tts.say("That's all for today. Goodbye, let's go home and sleep.")
        px.backward(30)
        time.sleep(6)
        px.stop()

    except KeyboardInterrupt:
        px.stop()
    finally:
        px.stop()
        px.set_dir_servo_angle(0)

if __name__ == "__main__":
    main()
