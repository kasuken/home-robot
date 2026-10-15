import re
import time
from picarx.llm import Ollama
from picarx.stt import Vosk
from picarx.tts import Piper

stt = Vosk(language="en-us")

tts = Piper()
tts.set_model("en_US-amy-low")

INSTRUCTIONS = (
    "You are a helpful assistant. Answer directly in plain English. "
    "Do NOT include any hidden thinking, analysis, or tags like <think>."
)
WELCOME = "Hello! I'm your voice chatbot. Speak when you're ready."

# --- INIT (set your host IP/model) ---
# If Ollama runs on the same Pi, use "localhost".
# If it runs on another computer in your LAN, replace with that computer's IP.
llm = Ollama(ip="localhost", model="llama3.2:3b")
llm.set_max_messages(20)
llm.set_instructions(INSTRUCTIONS)

def strip_thinking(text: str) -> str:
    """Remove any chain-of-thought sections and stray markers."""
    if not text:
        return ""
    text = re.sub(r"<\s*think[^>]*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"<\s*thinking[^>]*>.*?<\s*/\s*thinking\s*>", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"```(?:\s*thinking)?\s*.*?```", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"\[/?thinking\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+\n", "\n", text).strip()

def main():
    print(WELCOME)
    tts.say(WELCOME)

    try:
        while True:
            print("\n🎤 Listening... (Press Ctrl+C to stop)")

            # Collect final transcript from Vosk
            text = ""
            for result in stt.listen(stream=True):
                if result["done"]:
                    text = result["final"].strip()
                    print(f"[YOU] {text}")
                else:
                    print(f"[YOU] {result['partial']}", end="\r", flush=True)

            if not text:
                print("[INFO] Nothing recognized. Try again.")
                time.sleep(0.1)
                continue

            # Query LLM with streaming
            reply_accum = ""
            response = llm.prompt(text, stream=True)
            for next_word in response:
                if next_word:
                    print(next_word, end="", flush=True)
                    reply_accum += next_word
            print("")

            # Clean and speak only once
            clean = strip_thinking(reply_accum)
            if clean:
                tts.say(clean)
            else:
                tts.say("Sorry, I didn't catch that.")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        tts.say("Goodbye!")
        print("Bye.")

if __name__ == "__main__":
    main()
