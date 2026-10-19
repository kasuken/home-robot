# Status LEDs

`src/laika/leds.py`. Two LEDs show what she is doing, from across the room. Each shows one pattern at a time; the most urgent wins.

| LED (default) | Pattern | Means |
| --- | --- | --- |
| **status**, the Pi's red PWR LED (`pi_red`) | heartbeat: two quick thumps, a pause | she is up and fine |
| | slow blink (1 Hz) | battery LOW |
| | fast stutter (4 Hz) | battery CRITICAL |
| | solid on | emergency stop: wheels are refused until it is lifted |
| **activity**, the Robot HAT's user LED (`hat`) | fast blink (5 Hz) | thinking: a request to the model is in flight |
| | slow blink (1 Hz) | learning a new skill ([learning](learning.md)) |
| | solid on | awake, in a conversation |
| | off | asleep, waiting for "ehi Laika" |

`[leds]` in `config/laika.toml` chooses the LEDs: `hat` (the HAT's user LED), `pi_red` (the Pi's PWR LED), `pi_green` (the Pi's ACT LED) or `none`, for `status` and `activity`, which must differ. `enabled = false` switches both off. Nothing happens in `--sim`.

## How it works

One small thread polls the state (is a model request in flight, is the conversation awake, is a skill being learned, is the safety layer stopped, the battery state) every 20 ms and writes a LED only when its wanted value changes. No other thread ever waits on a LED, a failing LED is logged and ignored, and nothing here can reach the motors or the safety monitor.

- The **Pi's LEDs** go through `/sys/class/leds` (`PWR`, or `led1` on older kernels; `ACT` or `led0`), which needs root: the service is. She saves the LED's normal trigger, sets it to `none`, and **puts it back on a normal stop** (SIGTERM, Ctrl+C). After a hard kill the LED stays as it was left until a reboot or `echo default-on | sudo tee /sys/class/leds/PWR/trigger`.
- The **HAT's user LED** is SunFounder's `robot_hat.led.LED`, wrapped by `hardware.HatLed` (the adapter that may import SunFounder) and imported lazily. It is a single on/off LED (no colour, no brightness).
- A LED that is not there (no such file, no `robot_hat`) is reported once at start (`leds: ... is not available`) and stays off.

## The HAT's charge LED

The red CHG LED on the Robot HAT is, as far as I know, wired to the charger chip and lights by itself while the pack charges; it is not on a GPIO pin, so software cannot drive it (this is unverified: nothing in the vendor snapshot mentions it). That is why the default heartbeat uses the Pi's own red PWR LED instead. Also, she cannot tell whether she is charging (only the pack voltage is read), so the heartbeat does not stop while charging. If you prefer another LED for either job, change `[leds]`.

## Not verified on the robot

Everything above is tested with fake LEDs and a fake sysfs folder. Not tried on the real Pi: the PWR LED's sysfs name and trigger on your OS image, `robot_hat.led.LED` on your installed SunFounder version, and whether the PWR LED stops showing under-voltage once she takes it over. To try the HAT LED alone, see [tools/led_demo.py](../tools/led_demo.py).
